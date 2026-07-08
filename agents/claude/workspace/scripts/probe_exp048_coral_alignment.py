"""exp_048 (own, new): CORAL-style feature-space alignment (Sun & Saenko 2016,
arxiv/1607.01719) on top of exp_042's XGBoost+more_oversample banked-best
config.

Every prior shift-mitigation attempt this competition operated on the LOSS
(class_weight exp_009, focal-loss exp_045, importance-weighting exp_046 --
all killed) or on per-clip SCALE (RMS-normalization, exp_004) or added
DATA (augmentation, exp_018/033/037/038). None of them directly transform
the feature distribution's second-order statistics to match test. CORAL:
whiten train features by train covariance, then recolor with test's
covariance (a single linear transform, computed unsupervised from the
FEATURE MATRICES only -- test labels are never touched, so this is a valid
train-time transform, not leakage).

Single stratified 80/20 full-data holdout (probe fidelity, per exp_012's
lesson: full rows, not a row subsample). Compares vs exp_042's untransformed
XGBoost+more_oversample baseline on the SAME split, scored on
plain/weighted/topq (coordinator's gating metric). CORAL fit uses the TRAIN
holdout fold's covariance (not full train) vs the real held-out TEST set's
covariance, mirroring how it would be deployed (fit on train, applied to
match the actual test set) while still evaluating on labeled train rows.
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
CORAL_EPS = 1e-4  # ridge term for covariance invertibility


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def coral_transform(X_source, X_target):
    """Whiten X_source by its own covariance, recolor with X_target's
    covariance. Returns a function that applies the same fixed transform
    to any source-domain matrix (fit once, reused for train/holdout)."""
    def cov(M):
        Mc = M - M.mean(axis=0, keepdims=True)
        C = (Mc.T @ Mc) / (len(M) - 1)
        C += CORAL_EPS * np.eye(C.shape[0])
        return C

    Cs = cov(X_source)
    Ct = cov(X_target)

    # matrix square root / inverse square root via eigendecomposition (symmetric PSD)
    def mat_power(C, p):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        return (V * (w ** p)) @ V.T

    Cs_inv_sqrt = mat_power(Cs, -0.5)
    Ct_sqrt = mat_power(Ct, 0.5)
    A = Cs_inv_sqrt @ Ct_sqrt  # whiten then recolor, applied on the right: X_new = X @ A
    mean_s = X_source.mean(axis=0, keepdims=True)
    mean_t = X_target.mean(axis=0, keepdims=True)

    def apply(X):
        return (X - mean_s) @ A + mean_t

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

# --- adversarial P(is_test), same as exp_042/046/047, used only for topq scoring ---
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

# --- single 80/20 holdout ---
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
print(f"baseline (no CORAL) fit done ({time.time()-t0:.0f}s)", flush=True)

# --- sanity check: CORAL(X, X) must be ~identity (no-op when source==target) ---
identity_fn = coral_transform(X_tr_aug, X_tr_aug)
X_tr_identity = identity_fn(X_tr_aug)
max_abs_diff = np.max(np.abs(X_tr_identity - X_tr_aug))
print(f"sanity check (CORAL(X,X) should be ~identity): max abs diff = {max_abs_diff:.6f}", flush=True)

# --- CORAL: align (train+augmented) features to the REAL test set's covariance ---
coral_fn = coral_transform(X_tr_aug, X_test_real)
X_tr_aug_coral = coral_fn(X_tr_aug)
X_va_coral = coral_fn(X_va)  # apply the SAME fixed transform to the held-out fold
clf_coral = make_xgb(n_classes)
clf_coral.fit(X_tr_aug_coral, y_tr_aug)
pred_coral = clf_coral.predict(X_va_coral)
print(f"CORAL-aligned fit done ({time.time()-t0:.0f}s)", flush=True)

print("\n=== exp_048: XGBoost+aug, no-transform vs CORAL-aligned-to-real-test ===", flush=True)
plain_b, weighted_b, topq_b = report("no CORAL (exp_042 equivalent)", pred_base)
plain_c, weighted_c, topq_c = report("CORAL-aligned", pred_coral)
print(f"\ndelta plain:    {plain_c - plain_b:+.4f}", flush=True)
print(f"delta weighted: {weighted_c - weighted_b:+.4f}", flush=True)
print(f"delta topq:     {topq_c - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
