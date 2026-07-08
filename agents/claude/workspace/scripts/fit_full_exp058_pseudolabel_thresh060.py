"""exp_058 follow-through: fit XGBoost + exp_037/038's more_oversample
rare-class augmentation + fold-honest pseudo-labeling (CONF_THRESHOLD=0.60,
exp_059's lower-threshold-sweep winner) on the FULL train set and predict
test. Writes a submission CSV for the score gate.

Full 3-fold CV (full_cv_exp058_pseudolabel_thresh060.py) found this beats the
exp_042/sub_025-equivalent augmented-only baseline on the coordinator's
shift-aware topq metric: plain +0.0009, weighted +0.0180, topq +0.0137 --
roughly 2x exp_056's thresh=0.75 full-CV result (+0.0069). Recorded via
scheduler.py record --id exp_058 --stage full --delta 0.0137. This is the
current best full-CV-confirmed lever since sub_025 -- generating the actual
submission (stage-1 model trained on real full train, pseudo-labels real test
at conf>=0.60, stage-2 model retrained on train+pseudo-labels, predicts final
labels for all test rows).
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
CONF_THRESHOLD = 0.60


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
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

t1 = time.time()
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(Xtr_aug, ytr_aug)
print(f"stage-1 fit done ({time.time()-t1:.0f}s)", flush=True)

test_proba = clf_stage1.predict_proba(Xte)
test_conf = test_proba.max(axis=1)
test_pred = test_proba.argmax(axis=1)
pl_mask = test_conf >= CONF_THRESHOLD
n_pl = int(pl_mask.sum())
Xpl = Xte[pl_mask]
ypl = test_pred[pl_mask]
print(f"pseudo-labeled {n_pl}/{len(test_conf)} real test rows at conf>={CONF_THRESHOLD}", flush=True)

Xtr_pl = np.vstack([Xtr_aug, Xpl])
ytr_pl = np.concatenate([ytr_aug, ypl])

t2 = time.time()
clf_stage2 = make_xgb(n_classes)
clf_stage2.fit(Xtr_pl, ytr_pl)
pred_idx = clf_stage2.predict(Xte)
pred = le.inverse_transform(pred_idx)
print(f"stage-2 fit+predict done ({time.time()-t2:.0f}s)", flush=True)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp058_xgb_augmented_pseudolabel060.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
