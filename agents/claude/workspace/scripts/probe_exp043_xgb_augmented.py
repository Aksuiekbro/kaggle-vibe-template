"""exp_043 (own, probe, scheduler id exp_042): combine exp_040's XGBoost-alone
(best topq delta of the competition, +0.0257 full CV, wins all 3 metrics) with
exp_037/038's more_oversample rare-class augmentation (best augmentation
config for LGBM-alone, +0.0206 full CV). XGBoost has never been
augmentation-tuned; the augmentation config has never been tried on a
non-LGBM model family. Two independently full-CV-validated wins on
orthogonal axes -- testing whether they compose additively.

Single 80/20 full-data stratified holdout probe (per exp_012/C13 fidelity
lesson), same split/adversarial-weighting structure as probe_exp040_xgboost.py.
Reuses exp_038's augment_rows() (4-arg SNR/rate version) and RARE_THRESHOLD.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

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

# --- adversarial P(is_test) per train row, for the topq shift-aware metric ---
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
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s), n_topq={top_mask.sum()}", flush=True)

tr_idx, va_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w[va_idx]
top_mask_va = top_mask[va_idx]


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq_va={top_mask_va.sum()})", flush=True)
    return plain, weighted, topq


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


t1 = time.time()
clf_base = make_xgb()
clf_base.fit(X_tr, y_tr)
pred_base = clf_base.predict(X_va)
print(f"XGBoost baseline (no augment) fit+predict done ({time.time()-t1:.0f}s)", flush=True)

t2 = time.time()
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
rng_master = np.random.default_rng(SEED)
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"augmented {len(rare_tr_rows)} rare rows -> {len(aug_df)} synthetic rows ({time.time()-t2:.0f}s)", flush=True)

t3 = time.time()
clf_aug = make_xgb()
clf_aug.fit(X_tr_aug, y_tr_aug)
pred_aug = clf_aug.predict(X_va)
print(f"XGBoost augmented fit+predict done ({time.time()-t3:.0f}s)", flush=True)

print("\n=== exp_043: XGBoost-alone, no-augment vs more_oversample-augmented, full-data 80/20 holdout ===", flush=True)
plain_b, weighted_b, topq_b = report("XGBoost baseline (no augment)", pred_base)
plain_a, weighted_a, topq_a = report("XGBoost + more_oversample augment", pred_aug)
print(f"\ndelta plain:    {plain_a - plain_b:+.4f}", flush=True)
print(f"delta weighted: {weighted_a - weighted_b:+.4f}", flush=True)
print(f"delta topq:     {topq_a - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
