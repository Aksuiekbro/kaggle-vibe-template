"""exp_079 (own): probability-level ensemble blend of two SAME-ALGORITHM
(XGBoost) but DIFFERENT-FEATURE-SET models:
  - grid+aug: grid-harmonic features + more_oversample rare-class augmentation
    (the pre-pseudo-label half of the banked sub_027 pipeline)
  - grid+scattering: grid-harmonic + wavelet scattering features, no
    augmentation (scattering isn't augment-compatible; exp_073/074 full CV
    topq 0.9658 standalone)

Neither feature-concatenation (exp_073, capped) nor pseudo-labeling
(exp_075, composes NEGATIVELY with scattering) captured any interaction
between these two levers. This tests a THIRD composition mechanism --
prediction-level weighted blending of two independently-trained models --
which is mechanistically different from both: no shared feature space, no
label-injection loop, just a convex combination of probabilities. Per
STRATEGY.md, this differs from the already-closed "ensembling" axis (LGBM+
MLP / GBDT model-family blends, shift-fragile per coordinator) because both
arms here are the SAME model family (XGBoost), only the feature set differs
-- less likely to reproduce the model-family-driven shift-fragility pattern
the coordinator flagged.

Fold-safe: identical StratifiedKFold(N_SPLITS=3, shuffle=True,
random_state=SEED) on the identical 2330-row/label array confirmed shared
between exp071 (grid+aug) and exp073 (grid+scattering) -- verified same y
order before writing this script, so per-row OOF probabilities from both
arms align by index without any merge needed.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
BLEND_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)


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

train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")
assert len(train) == len(grid_train), "row-count mismatch after scattering merge -- fold alignment assumption broken"

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
assert np.array_equal(y, le.transform(grid_train["Pitch_ID"].values)), "row order shifted by merge"

X_grid = train[grid_feat_cols].values
X_scat = train[grid_feat_cols + scat_feat_cols].values
X_grid_test = test[grid_feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())

# adversarial weights on grid-only feature space (matches exp042/071's own gating weight)
X_adv = np.vstack([X_grid, X_grid_test])
y_adv = np.concatenate([np.zeros(len(X_grid)), np.ones(len(X_grid_test))])
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
w = p_test_all[: len(X_grid)]
print(f"adversarial AUC (grid features): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
proba_aug = np.zeros((len(y), n_classes))
proba_scat = np.zeros((len(y), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_grid, y)):
    t_fold = time.time()

    # arm 1: grid + more_oversample augmentation (no scattering)
    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_grid[tr_idx], aug_df[grid_feat_cols].values])
    y_tr_aug = np.concatenate([y[tr_idx], le.transform(aug_df["Pitch_ID"].values)])
    clf_aug = make_xgb(n_classes)
    clf_aug.fit(X_tr_aug, y_tr_aug)
    proba_aug[va_idx] = clf_aug.predict_proba(X_grid[va_idx])

    # arm 2: grid + scattering, no augmentation
    clf_scat = make_xgb(n_classes)
    clf_scat.fit(X_scat[tr_idx], y[tr_idx])
    proba_scat[va_idx] = clf_scat.predict_proba(X_scat[va_idx])

    acc_aug = accuracy_score(y[va_idx], proba_aug[va_idx].argmax(axis=1))
    acc_scat = accuracy_score(y[va_idx], proba_scat[va_idx].argmax(axis=1))
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): aug_acc={acc_aug:.4f} scat_acc={acc_scat:.4f}", flush=True)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return topq


print("\n=== exp_079: proba-blend of grid+aug (XGB) and grid+scattering (XGB) ===", flush=True)
topq_aug = report("grid+aug alone", proba_aug)
topq_scat = report("grid+scattering alone", proba_scat)

best_topq, best_w = -1, None
for bw in BLEND_WEIGHTS:
    blend = bw * proba_aug + (1 - bw) * proba_scat
    topq_b = report(f"blend(w_aug={bw})", blend)
    if topq_b > best_topq:
        best_topq, best_w = topq_b, bw

print(f"\nbest blend weight (w_aug): {best_w}, topq={best_topq:.4f}", flush=True)
print(f"delta vs grid+aug alone:        {best_topq - topq_aug:+.4f}", flush=True)
print(f"delta vs grid+scattering alone: {best_topq - topq_scat:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(GRID_DIR + "exp079_proba_aug.npy", proba_aug)
np.save(GRID_DIR + "exp079_proba_scat.npy", proba_scat)
np.save(GRID_DIR + "exp079_labels.npy", y)
