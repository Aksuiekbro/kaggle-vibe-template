"""exp_050 (own, scheduler id exp_049) full 3-fold CV: does dropping the
top-20% most shift-carrying grid features (by adversarial P(is_test)
feature_importances_) improve on exp_042 (XGBoost+more_oversample, all
features) under the coordinator's shift-aware topq metric?

exp_050's probe (single 80/20 holdout, exp050_probe.log) found:
  drop_top_10%: topq +0.0000 (flat)
  drop_top_20%: plain -0.0129, weighted +0.0506, topq +0.0085 (first
    positive topq delta since exp_042 landed)
Promoting drop_top_20% only (best probe config) to fold-safe 3-fold CV
before trusting it -- prior probe-scale deltas have reversed sign at full
CV (exp_010, exp_016).

Fold-safe: augmentation is generated only from each fold's TRAIN split
(identical to exp_042's full CV script). Feature importances for the
drop ranking come from the adversarial train-vs-test classifier, which
uses no labels and the same full train/test features exp_042 already
uses for weighting -- ranking itself cannot leak fold labels.
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
DROP_FRAC = 0.20

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)


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

# --- adversarial P(is_test), OOF weights + averaged feature importances ---
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
w = p_test_all[: len(X)]
importances /= 5
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

order = np.argsort(-importances)  # descending: most shift-carrying first
n_drop = int(round(len(feat_cols) * DROP_FRAC))
drop_idx = set(order[:n_drop].tolist())
keep_idx = [i for i in range(len(feat_cols)) if i not in drop_idx]
print(f"dropping top {DROP_FRAC:.0%} ({n_drop}/{len(feat_cols)} features), "
      f"{len(keep_idx)} retained", flush=True)

# --- fold-safe 3-fold CV: exp_042 reference (all feats + aug) vs drop_top_20% + aug ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
pred_ref = np.zeros(len(y), dtype=int)
pred_drop = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf_ref = make_xgb(n_classes)
    clf_ref.fit(X_tr_aug, y_tr_aug)
    pred_ref[va_idx] = clf_ref.predict(X_va)

    X_tr_aug_drop = X_tr_aug[:, keep_idx]
    X_va_drop = X_va[:, keep_idx]
    clf_drop = make_xgb(n_classes)
    clf_drop.fit(X_tr_aug_drop, y_tr_aug)
    pred_drop[va_idx] = clf_drop.predict(X_va_drop)

    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"ref_acc={accuracy_score(y[va_idx], pred_ref[va_idx]):.4f} "
          f"drop_acc={accuracy_score(y[va_idx], pred_drop[va_idx]):.4f}", flush=True)

correct_ref = (pred_ref == y).astype(float)
correct_drop = (pred_drop == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== exp_050 full CV: exp_042 reference (all feats+aug) vs drop_top_20%+aug ===", flush=True)
plain_r, weighted_r, topq_r = report("exp_042 reference (all features)", correct_ref)
plain_d, weighted_d, topq_d = report("drop_top_20% features", correct_drop)
print(f"\ndelta plain:    {plain_d - plain_r:+.4f}", flush=True)
print(f"delta weighted: {weighted_d - weighted_r:+.4f}", flush=True)
print(f"delta topq:     {topq_d - topq_r:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp050_pred_ref.npy", pred_ref)
np.save(DIR + "exp050_pred_drop.npy", pred_drop)
np.save(DIR + "exp050_labels.npy", y)
np.save(DIR + "exp050_keep_idx.npy", np.array(keep_idx))
