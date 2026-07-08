"""exp_081 follow-through: fit the banked XGBoost+aug+PL(thresh=0.50) pipeline
(same as fit_full_exp071_pseudolabel_thresh050.py) AND ExtraTrees+aug (no PL,
matching full_cv_exp081's arm 2) on the FULL train set, proba-blend at
w_xgb=0.3 (full 3-fold CV winner, topq 0.9708 vs XGBoost-alone's 0.9674,
delta +0.0034 -- confirmed same direction as the probe, recorded via
scheduler.py record --id exp_081 --stage full --delta 0.0034). Writes a
submission CSV for the score gate.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
CONF_THRESHOLD = 0.50
W_XGB = 0.3


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def make_et():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=1, random_state=42,
    )


t0 = time.time()
train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
ytr = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
Xtr = train[feat_cols].values
Xte = test[feat_cols].values
print(f"train rows: {len(Xtr)}, classes: {n_classes}", flush=True)

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
rare_rows = train[train["Pitch_ID"].isin(rare_classes)]
rng = np.random.default_rng(SEED)
aug_df = augment_rows(rare_rows, rng, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
Xtr_aug = np.vstack([Xtr, aug_df[feat_cols].values])
ytr_aug = np.concatenate([ytr, le.transform(aug_df["Pitch_ID"].values)])
print(f"rare classes (<{RARE_THRESHOLD}): {len(rare_classes)}/{len(full_counts)}, "
      f"augmented rows added: {len(aug_df)} ({time.time()-t0:.0f}s)", flush=True)

# arm 1: banked pipeline -- XGBoost + aug + pseudo-label(thresh=0.50)
t1 = time.time()
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(Xtr_aug, ytr_aug)
print(f"stage-1 XGBoost fit done ({time.time()-t1:.0f}s)", flush=True)

test_proba_stage1 = clf_stage1.predict_proba(Xte)
test_conf = test_proba_stage1.max(axis=1)
test_pred = test_proba_stage1.argmax(axis=1)
pl_mask = test_conf >= CONF_THRESHOLD
n_pl = int(pl_mask.sum())
Xpl = Xte[pl_mask]
ypl = test_pred[pl_mask]
print(f"pseudo-labeled {n_pl}/{len(test_conf)} real test rows at conf>={CONF_THRESHOLD}", flush=True)

Xtr_pl = np.vstack([Xtr_aug, Xpl])
ytr_pl = np.concatenate([ytr_aug, ypl])

t2 = time.time()
clf_xgb_pl = make_xgb(n_classes)
clf_xgb_pl.fit(Xtr_pl, ytr_pl)
proba_xgb = clf_xgb_pl.predict_proba(Xte)
print(f"stage-2 XGBoost+PL fit+predict done ({time.time()-t2:.0f}s)", flush=True)

# arm 2: ExtraTrees + aug (no PL), same augmented data as stage-1
t3 = time.time()
clf_et = make_et()
clf_et.fit(Xtr_aug, ytr_aug)
proba_et = clf_et.predict_proba(Xte)
print(f"ExtraTrees+aug fit+predict done ({time.time()-t3:.0f}s)", flush=True)

proba_blend = W_XGB * proba_xgb + (1 - W_XGB) * proba_et
pred_idx = proba_blend.argmax(axis=1)
pred = le.inverse_transform(pred_idx)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp081_xgbpl_extratrees_blend.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
