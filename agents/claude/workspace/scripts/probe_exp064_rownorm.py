"""exp_064 (scheduler id exp_063): per-clip (row-wise) z-normalization of the
extracted grid-harmonic feature vector.

Every feature-alignment mechanism tried so far (exp_047/048 CORAL, exp_048/049
diagonal alignment) transforms each FEATURE COLUMN independently using
train/test population statistics. exp_049's own notes proved this is
mathematically neutral for tree models: a monotonic per-column affine rescale
cannot change where XGBoost/LightGBM place histogram splits, so that whole
axis was structurally incapable of showing an effect for our model family.

This experiment normalizes the other way: for each CLIP (row), subtract that
row's own mean across its ~405 feature dimensions and divide by that row's
own std. This is NOT decomposable into independent per-column monotonic
transforms -- column j's output for a given row depends on every other
column's value in that same row (via the row mean/std) -- so it is not
covered by exp_049's invariance argument and can actually change split
behavior.

Motivated by CMVN (cepstral mean/variance normalization), the standard
channel-mismatch fix in speech recognition, and exp_003/004's diagnosis that
the train/test shift looks like a per-recording loudness/noise-floor
difference (test ~1.5x higher spectral contrast, ~0.5x lower ZCR and
chroma_cqt means). No cross-sample statistics are used anywhere (each row
normalizes only against itself), so this cannot leak train/test population
info in either direction.

Known risk (flagged before running, not after): the 405 columns mix
heterogeneous feature families (mfcc/chroma/contrast/zcr/tonnetz) with
different natural scales and units. Z-scoring a row across all of them
together is statistically unusual -- it could destroy real inter-family
signal along with whatever shared per-clip level artifact it removes. Cheap
single-holdout probe first (C4) rather than reasoning this out from priors.

Same split/seed/augmentation/model as exp_048-063 for direct comparability.
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

# baseline reference: identical split/seed/features/model as exp_048-063
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def row_normalize(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.clip(sd, 1e-8, None)
    return (X - mu) / sd


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


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

print("\n=== exp_064: per-clip (row-wise) z-normalization ===", flush=True)
X_tr_aug_rn = row_normalize(X_tr_aug)
X_va_rn = row_normalize(X_va)

clf = make_xgb(n_classes)
clf.fit(X_tr_aug_rn, y_tr_aug)
pred_va = clf.predict(X_va_rn)
print(f"\nrow-normalized fit ({time.time()-t0:.0f}s)", flush=True)
plain_rn, weighted_rn, topq_rn = report("row_normalized", pred_va)
print(f"delta plain    vs exp_048-ref: {plain_rn - BASELINE_PLAIN:+.4f}", flush=True)
print(f"delta weighted vs exp_048-ref: {weighted_rn - BASELINE_WEIGHTED:+.4f}", flush=True)
print(f"delta topq     vs exp_048-ref: {topq_rn - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
