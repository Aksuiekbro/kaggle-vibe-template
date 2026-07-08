"""exp_050 (own, new): adversarial-importance-guided feature elimination --
drop the grid features that most strongly separate train from test (by the
adversarial P(is_test) classifier's feature_importances_, averaged over its
5 folds), refit XGBoost+more_oversample on the reduced set, compare to
exp_042-equivalent baseline.

Distinct mechanism from exp_047/048 (CORAL, full covariance) and exp_048/049
(diagonal alignment, marginal mean/std): those TRANSFORM every feature to
match test's distribution. This ELIMINATES only the specific shift-carrying
columns and leaves every retained feature completely untouched -- no risk
of distorting retained class-relevant correlations, at the cost of losing
whatever weak signal the dropped columns carried.

Reuses the exact same 80/20 holdout split/seed/augmentation as exp_048/049,
so the untransformed baseline numbers are identical by construction
(same features, same model, same split) -- referenced as constants below
rather than refit, to spend the freed 2nd core on the new variant only.
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

# baseline reference: identical split/seed/features/model as this script,
# fit and reported by exp_048 (probe_exp048_coral_alignment.py, "no CORAL" row)
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

DROP_FRACS = (0.05, 0.15, 0.30, 0.40)  # exp_051: characterize curve around exp_050's 10%/20% points


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


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

# --- adversarial P(is_test), same as exp_042/046/047/048/049, collect feature importances too ---
X_adv = np.vstack([X, X_test_real])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(X_test_real))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
importances = np.zeros(len(feat_cols))
for tr_idx, va_idx in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_idx], y_adv[tr_idx])
    p_test_all[va_idx] = clf.predict_proba(X_adv[va_idx])[:, 1]
    importances += clf.feature_importances_
adv_auc = roc_auc_score(y_adv, p_test_all)
w_full = p_test_all[: len(X)]
importances /= 5
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

order = np.argsort(-importances)  # descending: most shift-carrying first
top_shift_feats = [feat_cols[i] for i in order[:10]]
print(f"top-10 most shift-carrying features: {top_shift_feats}", flush=True)

# --- single 80/20 holdout (same split/seed as exp_048/049) ---
idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug_full = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug_full)} rows ({len(aug_df)} augmented)", flush=True)

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

print("\n=== exp_050: XGBoost+aug, drop top adversarial-importance features ===", flush=True)
for frac in DROP_FRACS:
    n_drop = int(round(len(feat_cols) * frac))
    drop_idx = set(order[:n_drop].tolist())
    keep_idx = [i for i in range(len(feat_cols)) if i not in drop_idx]
    X_tr_reduced = X_tr_aug_full[:, keep_idx]
    X_va_reduced = X_va[:, keep_idx]
    clf = make_xgb(n_classes)
    clf.fit(X_tr_reduced, y_tr_aug)
    pred = clf.predict(X_va_reduced)
    print(f"\ndrop top {frac:.0%} ({n_drop}/{len(feat_cols)} features) fit done ({time.time()-t0:.0f}s)", flush=True)
    plain, weighted, topq = report(f"drop_top_{frac:.0%}", pred)
    print(f"delta plain:    {plain - BASELINE_PLAIN:+.4f}", flush=True)
    print(f"delta weighted: {weighted - BASELINE_WEIGHTED:+.4f}", flush=True)
    print(f"delta topq:     {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
