"""exp_047 (own, new): TRAIN-TIME importance weighting by adversarial
test-likeness, for XGBoost-alone + exp_037/038's more_oversample augmentation
(exp_042's current banked-best config).

Every prior use of exp_006's adversarial P(is_test) classifier this
competition was EVAL-ONLY (exp_006/020/023/033/037/038/040/042/043/044 all
used it to WEIGHT OR FILTER THE METRIC, never to weight the loss during
fitting). This is a genuinely new mechanism: pass w = P(is_test) as
sample_weight to XGBoost's .fit(), so train rows that look more like the
real test distribution get more influence on the learned model itself, not
just on how we score it afterward. This directly targets the actual
unexplained mechanism (a model trained to fit train-like rows equally may
not be the model that best fits test-like rows) rather than diagnosing it
post-hoc as every previous exp_006-derived experiment did.

Single stratified 80/20 full-data holdout (probe fidelity, per exp_012's
lesson: full rows, not a row subsample). Compares vs exp_042's un-weighted
XGBoost+more_oversample baseline on the SAME split, scored on
plain/weighted/topq (coordinator's gating metric).
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

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

# --- adversarial classifier: OOF P(is_test) per train row (same as exp_042) ---
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
w_full = p_test_all[: len(X)]
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- single 80/20 holdout ---
idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_tr = w_full[tr_idx]
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


# --- baseline: exp_042 equivalent, uniform sample weight ---
clf_base = make_xgb(n_classes)
clf_base.fit(X_tr_aug, y_tr_aug)
pred_base = clf_base.predict(X_va)
print(f"baseline (uniform weight) fit done ({time.time()-t0:.0f}s)", flush=True)

# --- importance-weighted: sample_weight = adversarial P(is_test) ---
# augmented rows: give them the mean train-side weight (they have no direct
# adversarial-classifier score of their own, and are synthetic anyway)
w_tr_aug_full = np.concatenate([w_tr, np.full(len(aug_df), w_tr.mean())])
clf_wt = make_xgb(n_classes)
clf_wt.fit(X_tr_aug, y_tr_aug, sample_weight=w_tr_aug_full)
pred_wt = clf_wt.predict(X_va)
print(f"importance-weighted fit done ({time.time()-t0:.0f}s)", flush=True)

print("\n=== exp_047: XGBoost+aug, uniform vs adversarial-importance-weighted training ===", flush=True)
plain_b, weighted_b, topq_b = report("uniform weight (exp_042 equivalent)", pred_base)
plain_w, weighted_w, topq_w = report("importance-weighted (w=P(is_test))", pred_wt)
print(f"\ndelta plain:    {plain_w - plain_b:+.4f}", flush=True)
print(f"delta weighted: {weighted_w - weighted_b:+.4f}", flush=True)
print(f"delta topq:     {topq_w - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
