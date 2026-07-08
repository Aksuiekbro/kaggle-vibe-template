"""exp_010 probe: LGBM hyperparameter tuning on grid-harmonic features (exp_005
feature set) at probe scale -- more estimators, shallower num_leaves, higher
min_child_samples, to check overfit-reduction on rare classes before spending
a full-scale run.

Baseline (exp_005 probe config): n_estimators=40, num_leaves=15,
min_child_samples=2 -> OOF 0.5876 (reference from probe_lgbm_grid.py).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

train = pd.read_parquet(OUT_DIR + "train_grid_features_probe.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

class_counts_all = train["Pitch_ID"].value_counts()
N_SPLITS = 2
keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))

CONFIGS = {
    "baseline (exp_005 probe config)": dict(n_estimators=40, num_leaves=15, min_child_samples=2),
    "more estimators, shallower, higher min_child": dict(n_estimators=200, num_leaves=7, min_child_samples=5),
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(X, y))

results = {}
for label, params in CONFIGS.items():
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in splits:
        clf = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"], learning_rate=0.1,
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])
    acc = accuracy_score(y, oof_pred)
    results[label] = acc
    print(f"{label}: {params} -> OOF acc={acc:.4f}")

base = results["baseline (exp_005 probe config)"]
tuned = results["more estimators, shallower, higher min_child"]
print(f"\ndelta (tuned - baseline): {tuned - base:+.4f}")
