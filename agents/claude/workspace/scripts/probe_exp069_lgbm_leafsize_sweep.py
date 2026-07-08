"""exp_068 (scheduler id) / exp_069 (idea label): isolate min_data_in_leaf as
the single variable distinguishing exp_044's GBDT-blend probe (LGBM+aug,
min_child_samples=2, implied weaker than XGBoost+aug) from exp_066's
few-shot-hyperparameter ablation control (LGBM+aug, min_data_in_leaf=20 true
LightGBM default, topq=0.9915 -- +0.0342 vs XGBoost+aug baseline).

min_data_in_leaf and min_child_samples are the SAME LightGBM parameter under
different aliases -- exp_044 (2) and exp_066's default (20) never isolated
this single knob head-to-head on identical data. This probe sweeps
min_data_in_leaf across {2, 5, 10, 20, 30, 50} with every other hyperparameter
and the exact same 80/20 split/seed/augmentation held fixed, to find out
whether exp_066's result was a genuine leaf-size effect or noise from the
single 117-row top-quartile holdout.

Cheap: single 80/20 holdout, 6 LGBM fits, no XGBoost refit needed (baseline
topq=0.9573 is the fixed, already-established XGBoost+aug reference used by
every probe since exp_048).
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes_ids = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes_ids)}/{len(full_counts)}", flush=True)

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

idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes_ids)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug)} rows ({len(aug_df)} augmented)", flush=True)

is_rare_va = np.array([le.classes_[c] in rare_classes_ids for c in y_va])
q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    rare_acc = correct[is_rare_va].mean() if is_rare_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} "
          f"rare_acc={rare_acc:.4f} (n_topq={top_mask.sum()}, n_rare={is_rare_va.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048/042 XGBoost+aug, same split): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

for mdl in (2, 5, 10, 20, 30, 50):
    tstart = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_data_in_leaf=mdl,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_tr_aug, y_tr_aug)
    pred = clf.predict(X_va)
    plain, weighted, topq = report(f"lgbm_min_data_in_leaf={mdl}", pred)
    print(f"  delta topq vs XGBoost+aug baseline: {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric "
          f"(fit {time.time()-tstart:.0f}s)", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
