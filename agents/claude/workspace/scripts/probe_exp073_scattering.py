"""exp_073 (own, research-sourced): does the wavelet scattering transform
(kymatio Scattering1D, J=8 Q=8, 8kHz, mean+std pooled -> 468 feats,
extract_scattering_features.py --full) carry signal on its own, and does it
add anything on top of the exp_005 grid-harmonic features?

Mechanistically distinct from every feature family tried so far (grid ACF/
cepstrum harmonic-evidence scoring, CREPE pitch-CNN embedding exp_028,
PANNs Cnn14 AudioSet embedding exp_029 -- both of which lost to grid-alone).
Motivated by kymatio's own docs (SOTA musical-instrument classification
under limited annotated training data) and arxiv 1601.00287 (scattering
linearizes pitch dynamics separately from timbre). See RESEARCH.md
2026-07-08 for full writeup.

Same 80/20 holdout harness/split/seed/augmentation/topq-weighting as
exp_035-072 for direct comparability. No pseudo-labeling here (isolate the
feature-family question first; pseudo-label threshold=0.60 is a separate,
already-closed lever per exp_058/059/072).

Three variants, run through identical XGBoost+aug pipeline:
  1. grid-only        (reference, BASELINE_TOPQ = 0.9573)
  2. scattering-only
  3. grid + scattering concatenated (additive test; per exp_007's rejected
     generic+grid combination, concatenation has previously diluted rather
     than added -- test anyway, C7 no monoculture)
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

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

# reference: identical split/seed/features/model as exp_048/049/050/057/059/072
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


t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

variants = {
    "scattering-only": (
        scat_train,
        scat_test,
        scat_feat_cols,
    ),
    "grid+scattering": (
        grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner"),
        grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner"),
        grid_feat_cols + scat_feat_cols,
    ),
}

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

results = {}
for name, (train, test, feat_cols) in variants.items():
    le = LabelEncoder()
    y = le.fit_transform(train["Pitch_ID"].values)
    n_classes = len(le.classes_)
    X = train[feat_cols].values
    X_test_real = test[feat_cols].values

    # adversarial P(is_test), reused only for topq weighting (same as exp_048/.../072)
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

    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_va = w_full[va_idx]

    rng_master = np.random.default_rng(SEED)
    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    # augment_rows -> extract_from_signal only computes grid features; it cannot
    # produce scattering columns (no kymatio call in that path). Augmentation is
    # only applied when every feat_col survives that extraction (grid-only).
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    if set(feat_cols).issubset(aug_df.columns):
        X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
        y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
    else:
        print(f"  [{name}] augmentation skipped: scattering features not producible by "
              f"augment_rows -> extract_from_signal; training on non-augmented rows only", flush=True)
        X_tr_aug, y_tr_aug = X_tr, y_tr

    clf = make_xgb(n_classes)
    clf.fit(X_tr_aug, y_tr_aug)
    pred = clf.predict(X_va)

    q75 = np.quantile(w_va, 0.75)
    top_mask = w_va >= q75
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    results[name] = (plain, weighted, topq)
    print(f"\n{name}: n_train={len(X_tr_aug)} ({len(aug_df)} aug) n_feats={len(feat_cols)} "
          f"adv_auc={adv_auc:.4f} plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} "
          f"(n_topq={top_mask.sum()}) ({time.time()-t0:.0f}s)", flush=True)
    print(f"  delta topq vs grid-baseline (0.9573): {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print("\n=== summary ===", flush=True)
for name, (plain, weighted, topq) in results.items():
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f}", flush=True)
print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
