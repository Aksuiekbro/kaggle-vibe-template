"""exp_033 follow-through: fit LGBM-ALONE (no MLP, no blend -- same model
family as sub_020, the only submission that has ever beaten its own CV
expectation on LB) on the FULL train set, augmented with exp_018's fold-safe
rare-class transform (applied once to the whole train set since there's no
held-out fold to leak into at final-fit time), and predict test. Writes a
submission.csv for the score gate.

This is the first candidate since PLAN.md's coordinator submission freeze
(note #2) to clear the shift-aware gate: exp_033's 3-fold CV found plain
+0.0094, weighted +0.0127, and -- the coordinator's actual gating metric --
topq (most-test-like-quartile accuracy) +0.0189 over the un-augmented
LGBM-alone baseline (the sub_020 equivalent).
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import augment_rows, RARE_THRESHOLD, SEED

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

t0 = time.time()
train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

rare_rows = train[train["Pitch_ID"].isin(rare_classes)]
rng_master = np.random.default_rng(SEED)
aug_df = augment_rows(rare_rows, rng_master)
print(f"augmented {len(rare_rows)} rare rows -> {len(aug_df)} extra rows ({time.time()-t0:.0f}s)", flush=True)

y_raw = np.concatenate([train["Pitch_ID"].values, aug_df["Pitch_ID"].values])
le = LabelEncoder()
ytr = le.fit_transform(y_raw)
n_classes = len(np.unique(ytr))

Xtr = np.vstack([train[feat_cols].values, aug_df[feat_cols].values])
Xte = test[feat_cols].values
print(f"train rows after augmentation: {len(Xtr)} (was {len(train)})", flush=True)

t1 = time.time()
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
)
clf_lgb.fit(Xtr, ytr)
pred_idx = clf_lgb.predict(Xte)
pred = le.inverse_transform(pred_idx)
print(f"LGBM fit+predict done ({time.time()-t1:.0f}s)", flush=True)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp033_lgbm_aug_only.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
