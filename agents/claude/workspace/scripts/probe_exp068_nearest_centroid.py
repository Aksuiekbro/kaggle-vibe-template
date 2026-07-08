"""exp_068: nearest-class-mean / prototypical classifier (sklearn
NearestCentroid) as a genuinely new base-model inductive bias, vs the
established XGBoost baseline on the same grid-harmonic features/split/
augmentation as exp_048-067.

Source: Snell et al. 2017, Prototypical Networks for Few-Shot Learning
(arxiv 1703.05175) -- nearest-class-mean in embedding space is the standard
few-shot-learning baseline; a class centroid is well-defined and low-variance
even from a single training sample, unlike a GBDT split-based decision
boundary which needs enough same-class rows to find a good split. Every
model tried on this competition so far (LGBM, XGBoost, CatBoost, MLP) is
split- or margin-based; no distance/metric-based classifier has been tried.

Mechanism hypothesis: this should specifically help rare_acc (14/82 classes
<10 samples) even if it's flat or negative on plain/topq for the majority
classes, since GBDTs structurally struggle to isolate a good split region for
classes with very few rows.

Risk (flagged before running): grid-harmonic features (405 raw dims) were
engineered for tree-based split-finding, not necessarily a clean Euclidean
metric space -- sweep raw vs PCA-reduced (20/50/100 components) to check
whether high-dimensional noise swamps the centroid signal before concluding
the mechanism itself is negative.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

scaler = StandardScaler().fit(X_tr_aug)
X_tr_s = scaler.transform(X_tr_aug)
X_va_s = scaler.transform(X_va)


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

print("\n=== exp_068: NearestCentroid, raw standardized features (405 dims) ===", flush=True)
nc = NearestCentroid()
nc.fit(X_tr_s, y_tr_aug)
pred_raw = nc.predict(X_va_s)
plain_r, weighted_r, topq_r = report("nearest_centroid_raw", pred_raw)
print(f"delta topq vs exp_048-ref: {topq_r - BASELINE_TOPQ:+.4f}", flush=True)

for n_comp in (20, 50, 100):
    print(f"\n=== exp_068: NearestCentroid, PCA(n={n_comp}) ===", flush=True)
    pca = PCA(n_components=n_comp, random_state=42).fit(X_tr_s)
    X_tr_p = pca.transform(X_tr_s)
    X_va_p = pca.transform(X_va_s)
    nc_p = NearestCentroid()
    nc_p.fit(X_tr_p, y_tr_aug)
    pred_p = nc_p.predict(X_va_p)
    plain_p, weighted_p, topq_p = report(f"nearest_centroid_pca{n_comp}", pred_p)
    print(f"delta topq vs exp_048-ref: {topq_p - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
