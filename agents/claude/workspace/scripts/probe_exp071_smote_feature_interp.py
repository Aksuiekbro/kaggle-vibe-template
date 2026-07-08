"""exp_070 (scheduler id) / exp_071 (idea label): SMOTE-style feature-space
interpolation augmentation for rare classes (<10 samples), on top of
exp_042's banked XGBoost+more_oversample config -- same 80/20 holdout
harness as exp_048-070.

Mechanistically distinct from exp_018/033/037's more_oversample (waveform
noise+time-stretch, re-extracted through the full grid-feature pipeline):
this interpolates between pairs of same-class REAL grid-harmonic feature
vectors directly (convex combination, t~U(0.2,0.8)), no audio touched.
Targets the same persistent rare-class problem via convex-hull densification
of the feature manifold instead of noise perturbation.

Fold-safe: interpolation pairs are drawn only from the TRAIN split; the
held-out 20% is never touched. Classes with <2 train samples after the
split can't form a pair and are skipped (not fabricated).

Tested standalone (SMOTE replacing more_oversample) and stacked (SMOTE
added on top of more_oversample).
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
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
SMOTE_PER_CLASS_TARGET = 12  # bring each rare class up to roughly this many synthetic+real rows

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
tr_rows = train.iloc[tr_idx].reset_index(drop=True)

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


def fit_eval(name, X_tr_use, y_tr_use):
    tstart = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )
    clf.fit(X_tr_use, y_tr_use)
    pred = clf.predict(X_va)
    plain, weighted, topq = report(name, pred)
    print(f"  delta topq vs XGBoost+aug baseline: {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric "
          f"(fit {time.time()-tstart:.0f}s, n_train={len(X_tr_use)})", flush=True)


def smote_interp(rows_df, X_rows, rng, per_class_target):
    """Interpolate between same-class pairs (train-split rows only) to
    synthesize new feature vectors. Skips classes with <2 available rows."""
    synth_X, synth_y = [], []
    pitch_ids = rows_df["Pitch_ID"].values
    for pid in rare_classes_ids:
        cls_mask = pitch_ids == pid
        cls_idx = np.where(cls_mask)[0]
        n_have = len(cls_idx)
        if n_have < 2:
            continue
        n_needed = max(0, per_class_target - n_have)
        for _ in range(n_needed):
            i, j = rng.choice(cls_idx, size=2, replace=(n_have < 2))
            t = rng.uniform(0.2, 0.8)
            synth_X.append(X_rows[i] + t * (X_rows[j] - X_rows[i]))
            synth_y.append(pid)
    if not synth_X:
        return np.empty((0, X_rows.shape[1])), np.array([])
    return np.vstack(synth_X), np.array(synth_y)


print(f"\nbaseline (reference, exp_048/042 XGBoost+aug, same split): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

rng_master = np.random.default_rng(SEED)

# variant A: SMOTE-only (replaces more_oversample entirely)
rng_smote = np.random.default_rng(rng_master.integers(0, 2**31))
smote_X, smote_y_raw = smote_interp(tr_rows, X_tr, rng_smote, SMOTE_PER_CLASS_TARGET)
smote_y = le.transform(smote_y_raw) if len(smote_y_raw) else np.array([], dtype=int)
X_tr_smote = np.vstack([X_tr, smote_X]) if len(smote_X) else X_tr
y_tr_smote = np.concatenate([y_tr, smote_y]) if len(smote_y) else y_tr
print(f"SMOTE synthesized {len(smote_X)} rows across {len(rare_classes_ids)} rare classes", flush=True)
fit_eval("smote_only", X_tr_smote, y_tr_smote)

# variant B: more_oversample-only (reproduces exp_042 baseline mechanism for reference)
rng_wave = np.random.default_rng(SEED)
aug_df = augment_rows(tr_rows[tr_rows["Pitch_ID"].isin(rare_classes_ids)], rng_wave,
                       MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_wave = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_wave = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
fit_eval("more_oversample_only (reference reproduction)", X_tr_wave, y_tr_wave)

# variant C: stacked (more_oversample + SMOTE on top)
rng_smote2 = np.random.default_rng(rng_master.integers(0, 2**31))
smote_X2, smote_y_raw2 = smote_interp(tr_rows, X_tr, rng_smote2, SMOTE_PER_CLASS_TARGET)
smote_y2 = le.transform(smote_y_raw2) if len(smote_y_raw2) else np.array([], dtype=int)
X_tr_stack = np.vstack([X_tr_wave, smote_X2]) if len(smote_X2) else X_tr_wave
y_tr_stack = np.concatenate([y_tr_wave, smote_y2]) if len(smote_y2) else y_tr_wave
fit_eval("more_oversample_plus_smote (stacked)", X_tr_stack, y_tr_stack)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
