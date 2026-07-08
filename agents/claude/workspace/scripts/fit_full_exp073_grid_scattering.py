"""exp_073/074 follow-through: fit XGBoost on grid+scattering (8kHz, no aug,
no pseudo-label -- both compose flat/negative per exp_076/exp_075) on the
FULL train set and predict test. Writes a submission CSV.

Prepared as the mechanistically-diverse hedge candidate (final-selection-by-
public-lb memory card): grid+scattering's full-CV topq (~0.9658, exp_073/074)
is below the banked exp_071 pseudo-label pipeline's topq (~0.9674, sub_027),
but it is a genuinely different feature family (adds a wavelet-scattering
axis on top of the harmonic grid) rather than a variant of the same pipeline,
so it carries different private-LB shake-up risk.
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

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
feat_cols = grid_feat_cols + scat_feat_cols

train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")
print(f"train rows: {len(train)}, test rows: {len(test)}, feats: {len(feat_cols)} ({time.time()-t0:.0f}s)", flush=True)

le = LabelEncoder()
ytr = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
Xtr = train[feat_cols].values
Xte = test[feat_cols].values

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
out_path = GRID_DIR + "submission_exp073_grid_scattering.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
