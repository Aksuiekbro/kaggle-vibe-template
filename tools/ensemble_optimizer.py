#!/usr/bin/env python3
"""
Ensemble Optimizer -- Portfolio-optimized model blending.

Treats model ensembling as portfolio optimization: each model is an "asset",
CV scores are "returns", and CV variance is "risk". Finds mathematically
optimal blend weights using Mean-CVaR optimization to maximize expected
score while minimizing private leaderboard shake-up risk.

Uses NVIDIA cufolio for GPU-accelerated optimization when available,
falls back to scipy-based optimization otherwise.

Usage:
    python tools/ensemble_optimizer.py \
        --predictions model1.csv model2.csv model3.csv \
        --cv-scores 0.85 0.83 0.87 \
        --cv-stds 0.01 0.02 0.005 \
        [--target target.csv] \
        [--metric rmse|auc|logloss|mae|accuracy] \
        [--alpha 0.95] \
        [--output blended.csv] \
        [--correlation-penalty 0.1]

Arguments:
    --predictions        CSV files with model predictions (same row order)
    --cv-scores          Mean CV score for each model (higher = better)
    --cv-stds            CV standard deviation for each model (lower = more stable)
    --target             Optional target CSV for direct optimization on holdout
    --metric             Scoring metric (default: rmse)
    --alpha              CVaR confidence level, 0-1 (default: 0.95, higher = more conservative)
    --output             Output blended predictions CSV (default: ensemble_blend.csv)
    --correlation-penalty  Penalty for correlated predictions, 0-1 (default: 0.1)

Concept:
    In portfolio theory, the optimal portfolio maximizes the Sharpe ratio
    (return / risk). Here we adapt this:
    - "Return" = expected CV score improvement from including a model
    - "Risk" = variance of the model's score across folds (shake-up risk)
    - "Correlation" = prediction correlation between models (diversification)
    - "CVaR" = Conditional Value at Risk -- worst-case expected score

    Models with high CV scores but also high variance get lower weights.
    Models that are uncorrelated with others get higher weights (diversification).
    The result: an ensemble that is robust to private LB shake-up.
"""

import argparse
import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(pred_files):
    """Load prediction files and return as list of numpy arrays."""
    predictions = []
    for f in pred_files:
        df = pd.read_csv(f)
        # Use the last column as the prediction column (convention: id, prediction)
        pred_col = df.columns[-1]
        predictions.append(df[pred_col].values.astype(np.float64))
    return predictions


def compute_correlation_matrix(predictions):
    """Compute pairwise Pearson correlation between model predictions."""
    n = len(predictions)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(predictions[i], predictions[j])[0, 1]
            corr[i, j] = r
            corr[j, i] = r
    return corr


def compute_covariance_matrix(cv_stds, corr_matrix):
    """Build covariance matrix from CV stds and prediction correlations."""
    n = len(cv_stds)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = cv_stds[i] * cv_stds[j] * corr_matrix[i, j]
    return cov


def portfolio_cvar_objective(weights, cv_scores, cov_matrix, alpha=0.95,
                              corr_penalty=0.1, corr_matrix=None, maximize=True):
    """
    Mean-CVaR objective function.

    Maximizes: expected_return - lambda * CVaR - penalty * correlation_cost

    For CVaR approximation under normality assumption:
        CVaR_alpha = -mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
    where phi is the standard normal PDF and Phi^{-1} is the inverse CDF.
    """
    from scipy.stats import norm

    # Expected return (weighted CV score)
    expected_return = np.dot(weights, cv_scores)

    # Portfolio variance
    port_variance = weights @ cov_matrix @ weights
    port_std = np.sqrt(max(port_variance, 1e-12))

    # CVaR under normality
    z_alpha = norm.ppf(alpha)
    phi_z = norm.pdf(z_alpha)
    cvar = -expected_return + port_std * phi_z / (1 - alpha)

    # Correlation penalty: penalize weight on highly correlated models
    corr_cost = 0.0
    if corr_matrix is not None and corr_penalty > 0:
        n = len(weights)
        for i in range(n):
            for j in range(i + 1, n):
                corr_cost += weights[i] * weights[j] * abs(corr_matrix[i, j])

    # We want to MINIMIZE this (scipy.minimize convention)
    sign = -1.0 if maximize else 1.0
    return sign * expected_return + cvar + corr_penalty * corr_cost


def optimize_cufolio(cv_scores, cv_stds, corr_matrix, alpha=0.95):
    """
    GPU-accelerated optimization using NVIDIA cufolio.

    cufolio implements Mean-CVaR portfolio optimization on GPU.
    This maps directly to our ensemble problem.
    """
    try:
        import cufolio
    except ImportError:
        raise ImportError("cufolio not available")

    n = len(cv_scores)

    # Build covariance matrix
    cov_matrix = compute_covariance_matrix(cv_stds, corr_matrix)

    # cufolio expects returns as a time-series matrix.
    # We simulate "historical returns" from CV scores and covariance.
    rng = np.random.default_rng(42)
    n_simulations = 1000
    simulated_returns = rng.multivariate_normal(
        mean=cv_scores, cov=cov_matrix, size=n_simulations
    )

    # Use cufolio's Mean-CVaR optimizer
    optimizer = cufolio.MeanCVaROptimizer(
        returns=simulated_returns,
        alpha=alpha,
    )
    weights = optimizer.optimize()

    # Normalize to sum to 1
    weights = np.array(weights)
    weights = np.maximum(weights, 0)
    weights /= weights.sum()

    return weights


def optimize_scipy(cv_scores, cv_stds, corr_matrix, alpha=0.95,
                   corr_penalty=0.1, maximize=True):
    """
    CPU fallback optimization using scipy.

    Uses SLSQP with the Mean-CVaR objective and multi-start initialization.
    """
    from scipy.optimize import minimize

    n = len(cv_scores)
    cov_matrix = compute_covariance_matrix(cv_stds, corr_matrix)

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: each weight in [0, 1]
    bounds = [(0.0, 1.0)] * n

    best_result = None
    best_obj = float("inf")

    # Multi-start to avoid local optima
    rng = np.random.default_rng(42)
    for trial in range(50):
        if trial == 0:
            # Start with equal weights
            w0 = np.ones(n) / n
        elif trial == 1:
            # Start with score-proportional weights
            w0 = cv_scores / cv_scores.sum()
        elif trial == 2:
            # Start with inverse-variance weights
            inv_var = 1.0 / (cv_stds ** 2 + 1e-12)
            w0 = inv_var / inv_var.sum()
        else:
            # Random start
            w0 = rng.dirichlet(np.ones(n))

        result = minimize(
            portfolio_cvar_objective,
            w0,
            args=(cv_scores, cov_matrix, alpha, corr_penalty, corr_matrix, maximize),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    if best_result is None:
        print("WARNING: Optimization did not converge. Using equal weights.",
              file=sys.stderr)
        return np.ones(n) / n

    weights = best_result.x
    weights = np.maximum(weights, 0)
    weights /= weights.sum()

    return weights


def optimize_hillclimb(predictions, target, metric_fn, maximize=True,
                       n_iter=10000):
    """
    Direct hill-climbing on the target metric.

    Used when target values are available for direct optimization.
    """
    n = len(predictions)
    rng = np.random.default_rng(42)

    # Start with equal weights
    best_weights = np.ones(n) / n
    best_pred = sum(w * p for w, p in zip(best_weights, predictions))
    best_score = metric_fn(target, best_pred)

    for _ in range(n_iter):
        # Perturb weights
        new_weights = best_weights.copy()
        idx1, idx2 = rng.choice(n, size=2, replace=False)
        delta = rng.uniform(0.001, 0.05)
        new_weights[idx1] += delta
        new_weights[idx2] -= delta

        # Ensure non-negative and normalized
        new_weights = np.maximum(new_weights, 0)
        if new_weights.sum() < 1e-12:
            continue
        new_weights /= new_weights.sum()

        # Evaluate
        new_pred = sum(w * p for w, p in zip(new_weights, predictions))
        new_score = metric_fn(target, new_pred)

        if (maximize and new_score > best_score) or \
           (not maximize and new_score < best_score):
            best_weights = new_weights
            best_score = new_score

    return best_weights


def get_metric_fn(metric_name):
    """Return (metric_function, maximize_flag) for the given metric name."""
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        from sklearn.metrics import mean_squared_error
        def root_mean_squared_error(y, p):
            return mean_squared_error(y, p) ** 0.5

    from sklearn.metrics import (
        mean_absolute_error,
        log_loss,
        roc_auc_score,
        accuracy_score,
    )

    metrics = {
        "rmse": (root_mean_squared_error, False),
        "mae": (mean_absolute_error, False),
        "logloss": (log_loss, False),
        "auc": (roc_auc_score, True),
        "accuracy": (lambda y, p: accuracy_score(y, (p > 0.5).astype(int)), True),
    }

    if metric_name not in metrics:
        print(f"Unknown metric '{metric_name}'. Available: {list(metrics.keys())}",
              file=sys.stderr)
        sys.exit(1)

    return metrics[metric_name]


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio-optimized ensemble blending (Mean-CVaR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions", nargs="+", required=True,
        help="Paths to prediction CSV files",
    )
    parser.add_argument(
        "--cv-scores", nargs="+", type=float, required=True,
        help="Mean CV score for each model",
    )
    parser.add_argument(
        "--cv-stds", nargs="+", type=float, required=True,
        help="CV standard deviation for each model",
    )
    parser.add_argument(
        "--target", default=None,
        help="Optional target CSV for direct metric optimization",
    )
    parser.add_argument(
        "--metric", default="rmse",
        choices=["rmse", "mae", "logloss", "auc", "accuracy"],
        help="Scoring metric (default: rmse)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.95,
        help="CVaR confidence level, 0-1 (default: 0.95, higher = more conservative)",
    )
    parser.add_argument(
        "--output", default="ensemble_blend.csv",
        help="Output blended predictions CSV (default: ensemble_blend.csv)",
    )
    parser.add_argument(
        "--correlation-penalty", type=float, default=0.1,
        help="Penalty for correlated predictions, 0-1 (default: 0.1)",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    n_models = len(args.predictions)
    if len(args.cv_scores) != n_models:
        print(f"ERROR: {n_models} prediction files but {len(args.cv_scores)} CV scores",
              file=sys.stderr)
        sys.exit(1)
    if len(args.cv_stds) != n_models:
        print(f"ERROR: {n_models} prediction files but {len(args.cv_stds)} CV stds",
              file=sys.stderr)
        sys.exit(1)

    for f in args.predictions:
        if not os.path.exists(f):
            print(f"ERROR: Prediction file not found: {f}", file=sys.stderr)
            sys.exit(1)

    # --- Load predictions ---
    print(f"Loading {n_models} prediction files...")
    predictions = load_predictions(args.predictions)

    # Verify all predictions have the same length
    lengths = [len(p) for p in predictions]
    if len(set(lengths)) != 1:
        print(f"ERROR: Prediction files have different lengths: {lengths}",
              file=sys.stderr)
        sys.exit(1)

    cv_scores = np.array(args.cv_scores)
    cv_stds = np.array(args.cv_stds)

    # --- Compute prediction correlations ---
    print("Computing prediction correlations...")
    corr_matrix = compute_correlation_matrix(predictions)

    print("\nPrediction Correlation Matrix:")
    model_names = [Path(f).stem for f in args.predictions]
    max_name_len = max(len(n) for n in model_names)
    header = " " * (max_name_len + 2) + "  ".join(
        f"{n[:8]:>8}" for n in model_names
    )
    print(header)
    for i, name in enumerate(model_names):
        row = f"{name:<{max_name_len}}  " + "  ".join(
            f"{corr_matrix[i, j]:8.4f}" for j in range(n_models)
        )
        print(row)

    # --- Determine metric ---
    metric_fn, maximize = get_metric_fn(args.metric)

    # --- Optimize weights ---
    print(f"\nOptimizing ensemble weights (alpha={args.alpha}, "
          f"metric={args.metric})...")

    weights = None
    backend = None

    # Try cufolio first (GPU-accelerated)
    try:
        weights = optimize_cufolio(cv_scores, cv_stds, corr_matrix, args.alpha)
        backend = "cufolio (GPU-accelerated Mean-CVaR)"
        print(f"  Backend: {backend}")
    except (ImportError, Exception) as e:
        print(f"  cufolio not available ({e}), trying scipy...")

    # If target is provided, use direct hill-climbing for final refinement
    if args.target:
        print(f"  Target provided -- refining with hill-climbing on {args.metric}...")
        target_df = pd.read_csv(args.target)
        target_col = target_df.columns[-1]
        target_values = target_df[target_col].values.astype(np.float64)

        weights = optimize_hillclimb(
            predictions, target_values, metric_fn, maximize=maximize
        )
        backend = "hill-climbing (direct metric optimization)"

    # Fallback to scipy
    if weights is None:
        weights = optimize_scipy(
            cv_scores, cv_stds, corr_matrix,
            args.alpha, args.correlation_penalty, maximize,
        )
        backend = "scipy SLSQP (CPU, Mean-CVaR)"
        print(f"  Backend: {backend}")

    # --- Report results ---
    print(f"\n{'=' * 60}")
    print("ENSEMBLE OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Backend: {backend}")
    print(f"Alpha (CVaR confidence): {args.alpha}")
    print(f"Metric: {args.metric}")
    print(f"Correlation penalty: {args.correlation_penalty}")
    print()

    header_fmt = (f"{'Model':<{max_name_len}}  {'Weight':>8}  {'CV Score':>10}  "
                  f"{'CV Std':>8}  {'Contribution':>14}")
    print(header_fmt)
    sep = (f"{'-' * max_name_len}  {'--------':>8}  {'----------':>10}  "
           f"{'--------':>8}  {'-' * 14:>14}")
    print(sep)
    for i, name in enumerate(model_names):
        contrib = weights[i] * cv_scores[i]
        print(f"{name:<{max_name_len}}  {weights[i]:8.4f}  {cv_scores[i]:10.6f}  "
              f"{cv_stds[i]:8.6f}  {contrib:14.6f}")

    cov_matrix = compute_covariance_matrix(cv_stds, corr_matrix)
    expected_score = np.dot(weights, cv_scores)
    port_var = weights @ cov_matrix @ weights
    port_std = np.sqrt(max(port_var, 0))

    print()
    print(f"Expected ensemble CV score:   {expected_score:.6f}")
    print(f"Ensemble std (shake-up risk): {port_std:.6f}")
    if port_std > 0:
        baseline = cv_scores.mean()
        sharpe = (expected_score - baseline) / port_std
        print(f"Sharpe-like ratio:            {sharpe:.4f}")

    # Models excluded (weight < 0.01)
    excluded = [model_names[i] for i in range(n_models) if weights[i] < 0.01]
    if excluded:
        print(f"\nModels excluded (weight < 1%): {', '.join(excluded)}")
        print("  These models either don't improve the ensemble or add too "
              "much risk/correlation.")

    # --- Generate blended predictions ---
    blended = sum(w * p for w, p in zip(weights, predictions))

    template_df = pd.read_csv(args.predictions[0])
    pred_col = template_df.columns[-1]
    template_df[pred_col] = blended
    template_df.to_csv(args.output, index=False)
    print(f"\nBlended predictions saved to: {args.output}")

    # Save weights as JSON for reproducibility
    weights_file = str(Path(args.output).stem) + "_weights.json"
    weights_dict = {
        "weights": {name: float(w) for name, w in zip(model_names, weights)},
        "cv_scores": {name: float(s) for name, s in zip(model_names, cv_scores)},
        "cv_stds": {name: float(s) for name, s in zip(model_names, cv_stds)},
        "expected_score": float(expected_score),
        "ensemble_std": float(port_std),
        "alpha": args.alpha,
        "metric": args.metric,
        "backend": backend,
        "correlation_matrix": corr_matrix.tolist(),
    }
    with open(weights_file, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Weight details saved to: {weights_file}")

    # --- Recommendations ---
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 60}")

    high_corr_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if corr_matrix[i, j] > 0.95:
                high_corr_pairs.append(
                    (model_names[i], model_names[j], corr_matrix[i, j])
                )

    if high_corr_pairs:
        print("\nWARNING: Highly correlated model pairs (>0.95):")
        for m1, m2, r in high_corr_pairs:
            print(f"  {m1} <-> {m2}: r={r:.4f}")
        print("  Consider dropping one from each pair -- they add risk "
              "without diversification.")

    high_var = [
        (model_names[i], cv_stds[i])
        for i in range(n_models)
        if cv_stds[i] > 2 * np.median(cv_stds)
    ]
    if high_var:
        print("\nWARNING: High-variance models (>2x median std):")
        for name, std in high_var:
            print(f"  {name}: std={std:.6f}")
        print("  These models are shake-up risks. The optimizer has "
              "down-weighted them accordingly.")

    if not high_corr_pairs and not high_var:
        print("\nNo warnings. Ensemble looks well-diversified and stable.")


if __name__ == "__main__":
    main()
