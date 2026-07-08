"""exp_037 full 3-fold CV confirmation: LGBM-alone + rare-class augmentation
using whichever hyperparameter config exp_037's probe picked as best, scored
under exp_006/033's shift-aware topq metric (the coordinator's stated gate).

Same fold-safe design as exp_033 (augmentation happens strictly inside each
fold's TRAIN split). Set BEST_CFG below to the winning probe config before
launching.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from probe_exp037_augment_tuning import augment_rows, CONFIGS

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3
SEED = 42

# Fill in from probe_exp037_augment_tuning.py's winning config name, e.g. "more_oversample"
BEST_CFG_NAME = "REPLACE_ME"

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
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
cfg = CONFIGS[BEST_CFG_NAME]
rare_classes = set(full_counts[full_counts < cfg["threshold"]].index.tolist())
print(f"config={BEST_CFG_NAME} {cfg} -- rare classes (<{cfg['threshold']} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

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

# --- 2. fold-safe 3-fold CV: baseline (no augment) vs winning config ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
pred_base = np.zeros(len(y), dtype=int)
pred_aug = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    clf_base = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_base.fit(X_tr, y_tr)
    pred_base[va_idx] = clf_base.predict(X_va)

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, cfg["snrs"], cfg["rates"])
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf_aug = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_aug.fit(X_tr_aug, y_tr_aug)
    pred_aug[va_idx] = clf_aug.predict(X_va)
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"base_acc={accuracy_score(y[va_idx], pred_base[va_idx]):.4f} "
          f"aug_acc={accuracy_score(y[va_idx], pred_aug[va_idx]):.4f}", flush=True)

correct_base = (pred_base == y).astype(float)
correct_aug = (pred_aug == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\n=== shift-aware comparison: LGBM-alone, no-augment vs {BEST_CFG_NAME} ===", flush=True)
plain_b, weighted_b, topq_b = report("baseline (no augment)", correct_base)
plain_a, weighted_a, topq_a = report(f"augmented ({BEST_CFG_NAME})", correct_aug)
print(f"\ndelta plain:   {plain_a - plain_b:+.4f}", flush=True)
print(f"delta weighted:{weighted_a - weighted_b:+.4f}", flush=True)
print(f"delta topq:    {topq_a - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"vs exp_033 (baseline_exp018 config) topq delta was +0.0189 -- this run tests whether tuning beats that", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
