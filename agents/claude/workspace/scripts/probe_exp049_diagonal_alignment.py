"""exp_049 (own, new): diagonal-only feature alignment -- per-feature mean/std
rescale of train grid features to match test's marginal mean/std, on top of
exp_042's XGBoost+more_oversample banked-best config.

exp_048 (full CORAL, closed) showed a mixed cross-metric result: weighted
+0.0585 (positive) but topq -0.0256 (negative, the coordinator's gating
metric) and plain -0.0343. Full CORAL's off-diagonal rotation can mix
feature correlations, potentially destroying class-relevant structure while
matching 2nd-order shift statistics. This is the diagonal special case: only
rescale+recenter each feature column independently (zero off-diagonal terms),
motivated by test-time batch-norm-calibration literature (arxiv/2110.04065)
which found full statistic substitution can hurt discriminative structure
while gentler marginal-only calibration is safer. Tests whether some of
CORAL's shift-closing benefit survives a lower-distortion transform without
the topq penalty.

Same probe harness as exp_048 (single stratified 80/20 full-data holdout,
same split/seed, same augmentation, same adversarial topq weights) for a
direct, comparable delta.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
DIAG_EPS = 1e-8  # guard against zero-variance columns


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def diagonal_transform(X_source, X_target):
    """Per-feature z-score X_source, then rescale/recenter to X_target's
    per-feature mean/std. No off-diagonal terms -- each column independent."""
    mean_s = X_source.mean(axis=0, keepdims=True)
    std_s = X_source.std(axis=0, keepdims=True) + DIAG_EPS
    mean_t = X_target.mean(axis=0, keepdims=True)
    std_t = X_target.std(axis=0, keepdims=True) + DIAG_EPS

    def apply(X):
        return (X - mean_s) / std_s * std_t + mean_t

    return apply


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

# --- adversarial P(is_test), same as exp_042/046/047/048, used only for topq scoring ---
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

# --- single 80/20 holdout (same split/seed as exp_048) ---
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


# --- baseline: exp_042 equivalent, no feature transform ---
clf_base = make_xgb(n_classes)
clf_base.fit(X_tr_aug, y_tr_aug)
pred_base = clf_base.predict(X_va)
print(f"baseline (no transform) fit done ({time.time()-t0:.0f}s)", flush=True)

# --- sanity check: diagonal_transform(X, X) must be ~identity ---
identity_fn = diagonal_transform(X_tr_aug, X_tr_aug)
X_tr_identity = identity_fn(X_tr_aug)
max_abs_diff = np.max(np.abs(X_tr_identity - X_tr_aug))
print(f"sanity check (transform(X,X) should be ~identity): max abs diff = {max_abs_diff:.6f}", flush=True)

# --- diagonal alignment: rescale (train+augmented) features to REAL test's marginal stats ---
diag_fn = diagonal_transform(X_tr_aug, X_test_real)
X_tr_aug_diag = diag_fn(X_tr_aug)
X_va_diag = diag_fn(X_va)  # apply the SAME fixed transform to the held-out fold
clf_diag = make_xgb(n_classes)
clf_diag.fit(X_tr_aug_diag, y_tr_aug)
pred_diag = clf_diag.predict(X_va_diag)
print(f"diagonal-aligned fit done ({time.time()-t0:.0f}s)", flush=True)

print("\n=== exp_049: XGBoost+aug, no-transform vs diagonal-aligned-to-real-test ===", flush=True)
plain_b, weighted_b, topq_b = report("no transform (exp_042 equivalent)", pred_base)
plain_d, weighted_d, topq_d = report("diagonal-aligned", pred_diag)
print(f"\ndelta plain:    {plain_d - plain_b:+.4f}", flush=True)
print(f"delta weighted: {weighted_d - weighted_b:+.4f}", flush=True)
print(f"delta topq:     {topq_d - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
