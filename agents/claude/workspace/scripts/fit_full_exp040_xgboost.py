"""exp_040 follow-through: fit XGBoost-ALONE (no LGBM, no MLP, no blend, no
augmentation) on the FULL train set and predict test. Writes a submission
CSV for the score gate.

exp_040's fold-safe 3-fold CV found XGBoost-alone beats LGBM-alone on grid
features on ALL THREE shift-aware metrics (plain +0.0318, weighted +0.0165,
topq +0.0257 -- the coordinator's own gating metric), resolving the probe's
weighted-metric disagreement (-0.1005 single-holdout) in XGBoost's favor on
every fold. This is the best topq delta of the competition so far, beating
exp_038's augmented-LGBM (+0.0206). Supersedes
submission_exp038_lgbm_more_oversample.csv as the top-priority banked
candidate for the next available submission slot (today's daily cap
already spent).
"""
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

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

t1 = time.time()
clf_xgb = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
    num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
    verbosity=0,
)
clf_xgb.fit(Xtr, ytr)
pred_idx = clf_xgb.predict(Xte)
pred = le.inverse_transform(pred_idx)
print(f"XGBoost fit+predict done ({time.time()-t1:.0f}s)", flush=True)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp040_xgboost_alone.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
