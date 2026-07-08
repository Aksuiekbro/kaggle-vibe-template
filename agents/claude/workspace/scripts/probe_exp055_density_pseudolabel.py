"""exp_055 (own, probe, scheduler id exp_054): density-regularized
pseudo-label acceptance -- replace exp_051/052's raw-confidence threshold
with a combined confidence+feature-likelihood score, per Revisiting
Self-Training with Regularized Pseudo-Labeling for Tabular Data
(arxiv 2302.14013): f(x) = (alpha*gamma + 1) * c / (alpha + 1), where c is
classifier confidence and gamma measures how typical x's features are for
its predicted class (keeps pseudo-labels in high-density regions instead of
accepting any high-confidence-but-possibly-atypical row).

Simplified, tractable implementation of gamma for this XGBoost/no-gradient
setting (the paper computes it via empirical per-feature binned
distributions, which is what this does): for each predicted class k, build
per-feature histograms (15 bins, Laplace-smoothed) from that class's rows
in the augmented train fold; for a candidate row predicted as k, sum
per-feature log-density across all 405 features, then convert to a
percentile RANK across all candidate test rows (0-1 scale) so it is
directly comparable in scale to classifier confidence c before combining.

Isolates the acceptance-CRITERION change from exp_052's threshold sweep:
fixes the number of accepted pseudo-labels at exp_052's own thresh=0.85
count (423/583, its best-performing config) and compares "top-423 by raw
confidence" (exp_052's actual mechanism) against "top-423 by combined
score" for alpha in {0.25, 0.5, 1.0} -- same retrain/eval harness, same
80/20 holdout split/seed as exp_048-052, so any delta is attributable to
the selection criterion alone, not to a different accept count.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

# reference numbers, identical split/seed/features/model as exp_048-052
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573
# exp_052's own best config (raw confidence threshold=0.85): the comparison target
THRESH085_N_PL = 423
THRESH085_PLAIN = 0.9485
THRESH085_WEIGHTED = 0.8383
THRESH085_TOPQ = 0.9658

ALPHAS = (0.25, 0.5, 1.0)
N_BINS = 15


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def class_log_likelihood(X_class, X_query, n_bins=N_BINS):
    """Sum of per-feature log-density of X_query rows under X_class's marginal histograms."""
    n_feat = X_class.shape[1]
    log_lik = np.zeros(len(X_query))
    for j in range(n_feat):
        col_class = X_class[:, j]
        lo, hi = col_class.min(), col_class.max()
        if hi <= lo:
            continue
        counts, edges = np.histogram(col_class, bins=n_bins, range=(lo, hi))
        counts = counts.astype(float) + 1.0  # Laplace smoothing
        bin_width = (hi - lo) / n_bins
        density = counts / (counts.sum() * bin_width)
        bin_idx = np.clip(np.digitize(X_query[:, j], edges[1:-1]), 0, n_bins - 1)
        log_lik += np.log(density[bin_idx])
    return log_lik


t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048-052) ---
X_adv = np.vstack([X, X_test_real])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(X_test_real))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tr_idx, va_idx in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_idx], y_adv[tr_idx])
    p_test_all[va_idx] = clf.predict_proba(X_adv[va_idx])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
w_full = p_test_all[: len(X)]
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- single 80/20 holdout (same split/seed as exp_048-052) ---
idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug)} rows ({len(aug_df)} augmented)", flush=True)

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (no pseudo-label): plain={BASELINE_PLAIN:.4f} weighted={BASELINE_WEIGHTED:.4f} "
      f"topq={BASELINE_TOPQ:.4f}", flush=True)
print(f"exp_052 thresh=0.85 (raw-confidence, {THRESH085_N_PL} pl rows): plain={THRESH085_PLAIN:.4f} "
      f"weighted={THRESH085_WEIGHTED:.4f} topq={THRESH085_TOPQ:.4f}  <- comparison target", flush=True)

# stage-1 model: tr+aug only, used to pseudo-label the REAL unlabeled test set
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(X_tr_aug, y_tr_aug)
test_proba = clf_stage1.predict_proba(X_test_real)
test_conf = test_proba.max(axis=1)
test_pred = test_proba.argmax(axis=1)
print(f"\nstage-1 model fit ({time.time()-t0:.0f}s)", flush=True)

# feature log-likelihood of each test row under its OWN predicted class's train+aug distribution
log_lik = np.zeros(len(X_test_real))
for k in range(n_classes):
    mask_pred_k = test_pred == k
    if not mask_pred_k.any():
        continue
    mask_class_k = y_tr_aug == k
    if mask_class_k.sum() < 2:
        # too few rows for this class to build a histogram -- neutral (rank-safe) fallback
        log_lik[mask_pred_k] = np.nan
        continue
    log_lik[mask_pred_k] = class_log_likelihood(X_tr_aug[mask_class_k], X_test_real[mask_pred_k])
# neutral fallback for classes with no train+aug rows: median log-likelihood of the rest
if np.isnan(log_lik).any():
    fallback = np.nanmedian(log_lik)
    log_lik[np.isnan(log_lik)] = fallback
gamma_rank = pd.Series(log_lik).rank(pct=True).values  # 0-1 scale, comparable to confidence
print(f"feature-likelihood computed ({time.time()-t0:.0f}s), gamma_rank range "
      f"[{gamma_rank.min():.3f}, {gamma_rank.max():.3f}]", flush=True)

print("\n=== exp_055: density-regularized (confidence+feature-likelihood) pseudo-label acceptance, "
      f"top-{THRESH085_N_PL} by combined score vs exp_052's top-{THRESH085_N_PL} by raw confidence ===", flush=True)
for alpha in ALPHAS:
    f_score = (alpha * gamma_rank + 1) * test_conf / (alpha + 1)
    top_n_idx = np.argsort(-f_score)[:THRESH085_N_PL]
    pl_mask = np.zeros(len(test_conf), dtype=bool)
    pl_mask[top_n_idx] = True
    X_pl = X_test_real[pl_mask]
    y_pl = test_pred[pl_mask]
    X_tr_pl = np.vstack([X_tr_aug, X_pl])
    y_tr_pl = np.concatenate([y_tr_aug, y_pl])
    clf = make_xgb(n_classes)
    clf.fit(X_tr_pl, y_tr_pl)
    pred = clf.predict(X_va)
    overlap = pl_mask & (test_conf >= 0.85)
    print(f"\nalpha={alpha}: {pl_mask.sum()} pl rows ({overlap.sum()} overlap with exp_052's thresh=0.85 set), "
          f"fit done ({time.time()-t0:.0f}s)", flush=True)
    plain, weighted, topq = report(f"density_alpha_{alpha}", pred)
    print(f"delta plain    vs no-pl:        {plain - BASELINE_PLAIN:+.4f}", flush=True)
    print(f"delta topq     vs no-pl:        {topq - BASELINE_TOPQ:+.4f}", flush=True)
    print(f"delta plain    vs thresh=0.85:  {plain - THRESH085_PLAIN:+.4f}", flush=True)
    print(f"delta weighted vs thresh=0.85:  {weighted - THRESH085_WEIGHTED:+.4f}", flush=True)
    print(f"delta topq     vs thresh=0.85:  {topq - THRESH085_TOPQ:+.4f}  <- coordinator's gating metric, vs the mechanism this is meant to improve on", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
