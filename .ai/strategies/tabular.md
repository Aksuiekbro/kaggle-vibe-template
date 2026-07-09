# Strategy Reference: Tabular ML Competitions

Classification and regression on structured/tabular data. This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 0. Similar Winner Upsolve
- Before heavy feature engineering, find similar past regression/classification competitions
- Match by metric, split type, data shape, target distribution, and leakage risk
- Prefer top 3 winner writeups when available and record whether they converge on the same load-bearing pattern
- Extract winner patterns as hypotheses: CV scheme, feature families, model families, ensembling, calibration, target transforms, postprocessing
- Convert each pattern into a small validation experiment before adding it to STRATEGY.md
- Use prospective prediction/postmortem loops from `.ai/checklists/kaggle-upsolve.md` and `.ai/checklists/postmortem.md`

### 1. EDA + Baseline (first hour)
- **Use cuDF as the default data processing library.** cuDF is a drop-in replacement for pandas with 10-100x speedup on GPU. It mirrors the pandas API, so existing pandas code works with minimal changes.
  - Import pattern: `import cudf as pd` (same API as pandas)
  - Fallback for environments without a GPU:
    ```python
    try:
        import cudf as pd
    except ImportError:
        import pandas as pd
    ```
- Understand the data: dtypes, missing values, distributions, correlations
- Identify target variable and evaluation metric
- Train a single LightGBM/XGBoost with default parameters
- Submit this baseline — establish your floor

### 2. Feature Engineering (core work)
- Domain-specific features based on problem understanding
- Aggregation features (group-by statistics)
- Interaction features (ratios, products of important features)
- Target encoding (with proper CV to avoid leakage)
- Date/time features if applicable
- Text features (TF-IDF, embeddings) if text columns exist
- **Synthetic data augmentation with Data Designer (NVIDIA NeMo):** Data Designer can generate realistic synthetic tabular data to augment small training sets. This is especially useful for rare classes or highly imbalanced datasets where minority-class signal is weak. It learns the joint distribution of your real data and produces new rows that respect feature correlations and conditional distributions, reducing overfitting and improving generalization on underrepresented segments. Install: `npx skills add nvidia/skills/data-designer`.

### 3. Model Selection
**Gradient Boosting (primary):**
- LightGBM: fastest, good default choice
- XGBoost: slightly different bias, good for ensembling with LightGBM
- CatBoost: handles categorical features natively

**Other models (for diversity in ensembles):**
- Neural networks (TabNet, simple MLP)
- Linear models (Logistic/Ridge with good features)
- k-NN (sometimes adds ensemble diversity)

### 4. Hyperparameter Tuning
- Use Optuna for Bayesian optimization
- **GPU-accelerated alternative: TAO AutoML.** TAO AutoML runs hyperparameter search on GPU with built-in support for WandB (Weights & Biases) tracking for experiment management. It parallelizes trial evaluation across GPU cores, dramatically reducing wall-clock time for large search spaces. Use TAO AutoML when you have GPU access and want integrated experiment tracking with visual dashboards, artifact versioning, and team collaboration via WandB. Install: `npx skills add nvidia/skills/tao-run-automl`.
- Tune on CV, not on single split
- Key LightGBM params: learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
- Don't over-tune — diminishing returns after ~100 trials

### 5. Ensembling
- Blend diverse models (GBM + NN + linear)
- Stacking: use out-of-fold predictions as features for a meta-learner
- Weight optimization: find optimal blend weights via CV (not public LB)
- Diversity matters more than individual model quality

**Portfolio-optimized ensemble weights (cufolio):**

Treat models as portfolio "assets", their CV scores as "returns", and CV variance across folds as "risk". Use Mean-CVaR (Conditional Value-at-Risk) optimization to find ensemble weights that maximize expected score while minimizing shake-up risk — the chance that your ensemble underperforms on private LB relative to public LB.

How it works:
1. Collect out-of-fold predictions from each model across all CV folds.
2. Compute per-fold scores for each model — this gives you a distribution of "returns" per model.
3. Build a covariance matrix of fold-level scores to capture how models co-move (correlated models add less diversification).
4. Run Mean-CVaR optimization: maximize `E[score] - lambda * CVaR_alpha(score)` where `lambda` controls risk aversion and `alpha` sets the tail threshold (e.g., worst 5% of folds).
5. The optimizer outputs non-negative weights that sum to 1, forming your final blend.

This approach penalizes models that are volatile across folds or highly correlated with others, producing ensembles that are more robust to distribution shift between public and private test sets.

Run: `python tools/ensemble_optimizer.py`
Install cufolio skill: `npx skills add nvidia/skills/cufolio`.

## Implementation Guidelines

- **Language**: Python with scikit-learn, LightGBM, XGBoost, CatBoost, cuDF
- **GPU Acceleration**: Use RAPIDS cuDF for data processing and cuML for GPU-accelerated ML algorithms (e.g., PCA, k-NN, UMAP, linear models). These provide orders-of-magnitude speedups over CPU equivalents and integrate seamlessly with the rest of the RAPIDS ecosystem. Prefer GPU-backed libraries whenever a GPU is available.
- **CV Strategy**: Stratified K-Fold (k=5 or k=10) for classification. Group K-Fold if data has groups. Time-series split if temporal.
- **Reproducibility**: Set random seeds everywhere
- **Memory**: Use appropriate dtypes (int8/16/32 instead of int64, float32 instead of float64)

## Common Pitfalls

- Target leakage through features derived from the target
- Not using proper CV (especially with time-series or grouped data)
- Over-tuning to public LB (classic shake-up victim)
- Ignoring feature importance — adding features blindly
- Training on all data without holdout for final validation
- Ensemble of similar models (doesn't add diversity)

## What Worked in Past Competitions

- Most Kaggle tabular winners: LightGBM + XGBoost + CatBoost ensemble
- Playground Series: heavy feature engineering + stacking
- Feature selection matters as much as model choice
- Treat these as starting hypotheses. Durable claims should be backed by memory cards with sources, scope, evidence, and counter-evidence.
