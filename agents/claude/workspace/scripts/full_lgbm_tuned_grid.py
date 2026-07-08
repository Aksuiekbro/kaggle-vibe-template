"""exp_010 full CV: LGBM hyperparameter tuning on grid-harmonic features
(exp_005 full-scale kernel output), StratifiedKFold(3), vs exp_005 baseline
LGBM config full OOF 0.9086.

Probe (300/100 subsample, N_SPLITS=2): baseline (n_est=40, leaves=15,
min_child=2) OOF=0.5876 vs tuned (n_est=200, leaves=7, min_child=5)
OOF=0.6667, delta +0.0790 -- promoting to full per C13.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))

CONFIGS = {
    "baseline (exp_005 full config)": dict(n_estimators=300, num_leaves=31, min_child_samples=2),
    "tuned (exp_010)": dict(n_estimators=500, num_leaves=7, min_child_samples=5),
}

N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(X, y))

t0 = time.time()
results = {}
for label, params in CONFIGS.items():
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in splits:
        clf = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"], learning_rate=0.05,
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])
    acc = accuracy_score(y, oof_pred)
    results[label] = acc
    print(f"{label}: {params} -> OOF acc={acc:.4f}  ({time.time()-t0:.0f}s elapsed)", flush=True)

base = results["baseline (exp_005 full config)"]
tuned = results["tuned (exp_010)"]
print(f"\nn={len(train)} classes={n_classes} n_feats={len(feat_cols)}")
print(f"delta (tuned - baseline): {tuned - base:+.4f}")
print(f"delta (tuned - exp_005 ref 0.9086): {tuned - 0.9086:+.4f}")
print(f"total time: {time.time()-t0:.0f}s")
