"""exp_042 follow-through: fit XGBoost + exp_037/038's more_oversample
rare-class augmentation on the FULL train set and predict test. Writes a
submission CSV for the score gate.

exp_042's fold-safe 3-fold CV found this beats XGBoost-alone (exp_040) on
all three shift-aware metrics (plain +0.0056, weighted +0.0315, topq
+0.0086), and combined with exp_040's model-family win the total vs the
sub_020-equivalent LGBM-no-augment baseline (topq 0.9177) is +0.0343 --
the best topq delta of the competition, beating exp_040 alone (+0.0257) and
exp_038 (+0.0206). Supersedes both as the top-priority banked submission
candidate for the next available slot.
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
clf_xgb = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
    num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
    verbosity=0,
)
clf_xgb.fit(Xtr_aug, ytr_aug)
pred_idx = clf_xgb.predict(Xte)
pred = le.inverse_transform(pred_idx)
print(f"XGBoost fit+predict done ({time.time()-t1:.0f}s)", flush=True)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp042_xgboost_augmented.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
