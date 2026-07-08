"""exp_068 (scheduler id) full 3-fold CV: LightGBM with true stock defaults
(min_data_in_leaf=20) + more_oversample augmentation vs the currently-banked
XGBoost + more_oversample baseline (exp_042/sub_025), scored under the
coordinator's shift-aware topq metric.

Motivation: exp_068's probe (single 80/20 holdout sweep of min_data_in_leaf in
{2,5,10,20,30,50}) found ONLY mdl=20 (the true LightGBM default) positive
(+0.0342 topq), with both neighbors (10, 30) flat/negative -- a sharp isolated
peak, not a smooth trend. That shape is the signature of single-holdout noise
on a thin 117-row top-quartile slice (n_rare=16), the same failure mode that
reversed exp_012/exp_019 at full CV. This script settles it with a real
fold-safe 3-fold CV before any submission decision, same harness as
exp_040/042/051.

Fold-safe: augmentation is generated only from each fold's TRAIN split.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def make_lgbm_mdl20():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )


t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

# --- 1. adversarial classifier: train vs test, OOF P(is_test) per train row ---
X_adv = np.vstack([X, test[feat_cols].values])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(test))])
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
w = p_test_all[: len(X)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- 2. fold-safe 3-fold CV: XGBoost+aug (banked) vs LGBM mdl=20 + aug ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
pred_xgb = np.zeros(len(y), dtype=int)
pred_lgbm = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf_xgb = make_xgb(n_classes)
    clf_xgb.fit(X_tr_aug, y_tr_aug)
    pred_xgb[va_idx] = clf_xgb.predict(X_va)

    clf_lgbm = make_lgbm_mdl20()
    clf_lgbm.fit(X_tr_aug, y_tr_aug)
    pred_lgbm[va_idx] = clf_lgbm.predict(X_va)

    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"xgb_acc={accuracy_score(y[va_idx], pred_xgb[va_idx]):.4f} "
          f"lgbm_mdl20_acc={accuracy_score(y[va_idx], pred_lgbm[va_idx]):.4f}", flush=True)

correct_xgb = (pred_xgb == y).astype(float)
correct_lgbm = (pred_lgbm == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== exp_068 full CV: XGBoost+aug (banked) vs LGBM min_data_in_leaf=20 +aug ===", flush=True)
plain_x, weighted_x, topq_x = report("XGBoost + more_oversample (banked baseline)", correct_xgb)
plain_l, weighted_l, topq_l = report("LGBM min_data_in_leaf=20 + more_oversample", correct_lgbm)
print(f"\ndelta plain:    {plain_l - plain_x:+.4f}", flush=True)
print(f"delta weighted: {weighted_l - weighted_x:+.4f}", flush=True)
print(f"delta topq:     {topq_l - topq_x:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp068_pred_xgb.npy", pred_xgb)
np.save(DIR + "exp068_pred_lgbm_mdl20.npy", pred_lgbm)
np.save(DIR + "exp068_labels.npy", y)
