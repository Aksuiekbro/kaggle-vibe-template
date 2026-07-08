"""exp_042 probe (own): test-time augmentation (TTA) for LGBM-alone + exp_037's
more_oversample rare-class train-side augmentation -- average predicted
class probabilities over noise + time-stretch variants of each HELD-OUT
clip at INFERENCE time, instead of only ever augmenting the TRAIN side.

Every prior augmentation experiment this competition (exp_018/025/033/037/
038/039) augmented train only. This tests whether averaging over label-
preserving perturbations of the clip being predicted (inference-time
variance reduction) helps -- a genuinely new axis, not a variant of the
closed augmentation-hyperparameter or model-family axes.

Single stratified 80/20 holdout (probe fidelity, full rows per exp_012's
lesson). TTA variants are generated only from the held-out rows' own audio
(no leakage -- held-out labels are never used to fit anything).
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import extract_from_signal
from probe_exp037_augment_tuning import augment_rows, augment_variants, CONFIGS

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
SEED = 42
BEST_CFG_NAME = "more_oversample"

# Deliberately gentler than train-side augmentation -- TTA variants must stay
# close to the original clip's real class-relevant content, not enlarge a
# training set.
TTA_SNRS = (20.0, 12.0)
TTA_RATES = (0.95, 1.05)

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
cfg = CONFIGS[BEST_CFG_NAME]
rare_classes = set(full_counts[full_counts < cfg["threshold"]].index.tolist())

idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
aug_df = augment_rows(rare_tr_rows, rng_master, cfg["snrs"], cfg["rates"])
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train fold: {len(tr_idx)} rows + {len(aug_df)} augmented rare rows = {len(X_tr_aug)}", flush=True)

clf = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
clf.fit(X_tr_aug, y_tr_aug)
print(f"fit done ({time.time()-t0:.0f}s)", flush=True)

proba_orig = clf.predict_proba(X_va)
fold_classes = clf.classes_
full_proba_orig = np.zeros((len(va_idx), n_classes))
full_proba_orig[:, fold_classes] = proba_orig
pred_no_tta = fold_classes[np.argmax(proba_orig, axis=1)]
acc_no_tta = accuracy_score(y_va, pred_no_tta)
print(f"no-TTA held-out acc: {acc_no_tta:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- TTA: generate variants of each held-out clip, average probabilities ---
va_rows = train.iloc[va_idx]
t_tta = time.time()
tta_proba_sum = full_proba_orig.copy()
tta_proba_count = np.ones(len(va_idx))
rng_tta = np.random.default_rng(SEED + 1)

for i, path in enumerate(va_rows["Path"].values):
    variants, sr = augment_variants(path, rng_tta, TTA_SNRS, TTA_RATES)
    feats_list = [extract_from_signal(v, sr) for v in variants]
    Xv = pd.DataFrame(feats_list)[feat_cols].values
    pv = clf.predict_proba(Xv)
    full_pv = np.zeros((len(variants), n_classes))
    full_pv[:, fold_classes] = pv
    tta_proba_sum[i] += full_pv.sum(axis=0)
    tta_proba_count[i] += len(variants)
    if (i + 1) % 100 == 0:
        print(f"  TTA progress: {i+1}/{len(va_idx)} ({time.time()-t_tta:.0f}s)", flush=True)

tta_proba_avg = tta_proba_sum / tta_proba_count[:, None]
pred_tta = np.argmax(tta_proba_avg, axis=1)
acc_tta = accuracy_score(y_va, pred_tta)

print("\n=== exp_042 TTA probe summary ===", flush=True)
print(f"no-TTA acc:  {acc_no_tta:.4f}", flush=True)
print(f"TTA acc:     {acc_tta:.4f}", flush=True)
print(f"delta:       {acc_tta - acc_no_tta:+.4f}", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
