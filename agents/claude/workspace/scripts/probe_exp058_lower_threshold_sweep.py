"""exp_059 (scheduler id exp_058, own, new): even-lower confidence-threshold
sweep (0.60/0.65/0.70) for fold-honest pseudo-labeling.

exp_057 (scheduler id exp_056) found topq improving monotonically as
threshold drops from 0.90->0.75 (+0.0000/+0.0085/+0.0085/+0.0171), with no
reversal yet -- 0.75 may not be the peak. This fills in the next lower band
to find where (if anywhere) the trend actually turns over, using the
identical harness/split/seed/augmentation config as exp_051/052/055/056/057
for direct comparability.
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

# baseline reference: identical split/seed/features/model as exp_048/049/050/057
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573
THRESH_075_TOPQ = 0.9744  # exp_057's current best (scheduler id exp_056)

CONF_THRESHOLDS = (0.70, 0.65, 0.60)


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

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048/049/050/057) ---
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

# --- single 80/20 holdout (same split/seed as exp_048/049/050/057) ---
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


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

# stage-1 model: tr+aug only, used to pseudo-label the REAL unlabeled test set
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(X_tr_aug, y_tr_aug)
test_proba = clf_stage1.predict_proba(X_test_real)
test_conf = test_proba.max(axis=1)
test_pred = test_proba.argmax(axis=1)
print(f"\nstage-1 model fit ({time.time()-t0:.0f}s). Real test confidence "
      f"percentiles: p50={np.percentile(test_conf, 50):.3f} p75={np.percentile(test_conf, 75):.3f} "
      f"p90={np.percentile(test_conf, 90):.3f} max={test_conf.max():.3f}", flush=True)

print("\n=== exp_059: even-lower confidence-threshold sweep (0.60/0.65/0.70), "
      "XGBoost+aug+pseudo-label(real test) ===", flush=True)
for thresh in CONF_THRESHOLDS:
    pl_mask = test_conf >= thresh
    n_pl = pl_mask.sum()
    if n_pl == 0:
        print(f"\nthreshold {thresh}: 0 pseudo-labeled rows, skipping", flush=True)
        continue
    X_pl = X_test_real[pl_mask]
    y_pl = test_pred[pl_mask]
    X_tr_pl = np.vstack([X_tr_aug, X_pl])
    y_tr_pl = np.concatenate([y_tr_aug, y_pl])
    clf = make_xgb(n_classes)
    clf.fit(X_tr_pl, y_tr_pl)
    pred = clf.predict(X_va)
    print(f"\nthreshold {thresh}: {n_pl}/{len(test_conf)} real test rows pseudo-labeled, "
          f"fit done ({time.time()-t0:.0f}s)", flush=True)
    plain, weighted, topq = report(f"pseudolabel_thresh_{thresh}", pred)
    print(f"delta plain    vs no-pl:      {plain - BASELINE_PLAIN:+.4f}", flush=True)
    print(f"delta weighted vs no-pl:      {weighted - BASELINE_WEIGHTED:+.4f}", flush=True)
    print(f"delta topq     vs no-pl:      {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)
    print(f"delta topq     vs thresh=0.75: {topq - THRESH_075_TOPQ:+.4f}", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
