"""exp_012 probe: does more LGBM capacity (num_leaves) help on grid features
at FULL data scale, using a single stratified 80/20 split (not 3-fold) to cut
cost ~3x while keeping full row/class coverage?

Motivated by exp_010's full-scale reversal: a tiny 300/100-row subsample probe
predicted shallower trees (leaves=7) would help (+0.079), but the full 2330-row
run showed leaves=7 hurts (-0.0043 vs baseline leaves=31). The scheduler's own
note after exp_010: "if it recurs, raise probe fidelity." Rather than shrink
the row count, this probe keeps ALL 2330 rows/82 classes and only cuts fold
count (1 split instead of 3) to stay cheap, since exp_010 showed capacity
conclusions do not transfer from small-row subsamples.

Baseline (exp_005/exp_010 full-CV best): num_leaves=31, n_estimators=300,
min_child_samples=2 -> full 3-fold OOF 0.9090. This probe checks whether MORE
capacity (num_leaves=63, 127) helps, since baseline (31) beat LESS capacity
(7) at full scale.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

class_counts = train["Pitch_ID"].value_counts()
keep_classes = class_counts[class_counts >= 2].index
train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

CONFIGS = {
    "baseline (leaves=31)": dict(n_estimators=300, num_leaves=31, min_child_samples=2),
    "leaves=63": dict(n_estimators=300, num_leaves=63, min_child_samples=2),
    "leaves=127": dict(n_estimators=300, num_leaves=127, min_child_samples=2),
}

t0 = time.time()
results = {}
for label, params in CONFIGS.items():
    clf = lgb.LGBMClassifier(
        n_estimators=params["n_estimators"], learning_rate=0.05,
        num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, clf.predict(X_va))
    results[label] = acc
    print(f"{label}: {params} -> holdout acc={acc:.4f}  ({time.time()-t0:.0f}s elapsed)", flush=True)

base = results["baseline (leaves=31)"]
for label, acc in results.items():
    if label != "baseline (leaves=31)":
        print(f"delta ({label} - baseline): {acc - base:+.4f}")
print(f"n={len(train)} classes={n_classes} n_feats={len(feat_cols)} total_time={time.time()-t0:.0f}s")
