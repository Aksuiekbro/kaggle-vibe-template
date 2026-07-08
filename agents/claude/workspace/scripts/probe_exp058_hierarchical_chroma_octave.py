"""exp_058 (own, new): hierarchical chroma+octave decomposition.

exp_034 measured that 58.0% of this model's own OOF misclassifications land
at exactly the semitone distances (7/12/19/24) consistent with octave/fifth
confusion -- the classic missing-fundamental failure mode. Two prior fixes
for this exact error mode both failed: exp_035 (added one new *feature* to
the same flat 82-way objective, full CV topq -0.0051) and exp_036 (a
*post-hoc decision rule* on the already-trained flat classifier's outputs,
probe -0.088). Neither touched the training objective itself.

This probe restructures the objective instead, per the deep-layered-learning
pattern from pitch/key-estimation literature (arxiv 1804.07297, 1906.07145):
decompose Pitch_ID into an octave-invariant chroma component (Pitch_ID % 12)
and a register/octave component (Pitch_ID // 12), train two heads on the
same feature set, and fuse via joint probability scoring restricted to the
(chroma, octave) combinations that actually occur among this dataset's 82
labels (not all 12*7 combinations exist).

Reuses the exact same 80/20 holdout split/seed, augmentation config, and
XGBoost hyperparameters as exp_048/049/050/051/052 for direct comparability
to the banked exp_042/BASELINE_* reference numbers.
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

# baseline reference: identical split/seed/features/model as exp_048/049/050/051/052
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


def make_xgb_proba(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)  # flat 82-way encoding, for the baseline model + valid-class enumeration
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048-052) ---
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

# --- single 80/20 holdout (same split/seed as exp_048-052) ---
idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]  # flat-encoded labels
w_va = w_full[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug_flat = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug)} rows ({len(aug_df)} augmented)", flush=True)

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred_flat):
    correct = (pred_flat == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

# --- flat 82-way reproduction check (should land near BASELINE_*) ---
clf_flat = make_xgb(n_classes)
clf_flat.fit(X_tr_aug, y_tr_aug_flat)
pred_flat_check = clf_flat.predict(X_va)
print(f"\nflat-82way reproduction check (fit done, {time.time()-t0:.0f}s):", flush=True)
report("flat_82way", pred_flat_check)

# --- hierarchical chroma+octave decomposition ---
# original (un-encoded) Pitch_ID values for chroma/octave arithmetic
pitch_id_tr_aug = np.concatenate([train["Pitch_ID"].values[tr_idx], aug_df["Pitch_ID"].values])
pitch_id_va = train["Pitch_ID"].values[va_idx]

chroma_tr = pitch_id_tr_aug % 12
octave_tr = pitch_id_tr_aug // 12
n_chroma = int(chroma_tr.max()) + 1  # 12
n_octave = int(octave_tr.max()) + 1

print(f"\nchroma classes: {n_chroma}, octave classes: {n_octave}", flush=True)

# enumerate the valid (chroma, octave) -> flat-encoded-class map from classes actually seen in training
valid_classes_flat = np.unique(y_tr_aug_flat)
valid_pitch_ids = le.inverse_transform(valid_classes_flat)
valid_chroma = valid_pitch_ids % 12
valid_octave = valid_pitch_ids // 12

clf_chroma = make_xgb_proba(n_chroma)
clf_chroma.fit(X_tr_aug, chroma_tr)
print(f"chroma head fit ({time.time()-t0:.0f}s)", flush=True)

clf_octave = make_xgb_proba(n_octave)
clf_octave.fit(X_tr_aug, octave_tr)
print(f"octave head fit ({time.time()-t0:.0f}s)", flush=True)

proba_chroma_va = clf_chroma.predict_proba(X_va)  # (n_va, n_chroma)
proba_octave_va = clf_octave.predict_proba(X_va)  # (n_va, n_octave)

# joint score for each valid (chroma, octave) combo = P(chroma) * P(octave), argmax over valid combos only
joint_scores = proba_chroma_va[:, valid_chroma] * proba_octave_va[:, valid_octave]  # (n_va, n_valid_classes)
best_valid_idx = joint_scores.argmax(axis=1)
pred_hier_flat = valid_classes_flat[best_valid_idx]

print(f"\n=== exp_058: hierarchical chroma+octave joint-score fusion ===", flush=True)
plain, weighted, topq = report("hierarchical_chroma_octave", pred_hier_flat)
print(f"delta plain:    {plain - BASELINE_PLAIN:+.4f}", flush=True)
print(f"delta weighted: {weighted - BASELINE_WEIGHTED:+.4f}", flush=True)
print(f"delta topq:     {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

# sanity: how often does each head get its own sub-target right (independent of fusion)
chroma_va_true = pitch_id_va % 12
octave_va_true = pitch_id_va // 12
chroma_acc = (proba_chroma_va.argmax(axis=1) == chroma_va_true).mean()
octave_acc = (proba_octave_va.argmax(axis=1) == octave_va_true).mean()
print(f"\nchroma-head-alone accuracy: {chroma_acc:.4f}", flush=True)
print(f"octave-head-alone accuracy: {octave_acc:.4f}", flush=True)
print(f"(product of independent head accuracies as a naive ceiling estimate: {chroma_acc*octave_acc:.4f})", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
