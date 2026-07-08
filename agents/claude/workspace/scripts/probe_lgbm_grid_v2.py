"""exp_014 probe: LGBM on grid-harmonic features v2 (widened ACF/cepstrum lag
window) vs exp_005's original grid features (exact-rounded lag), same probe
subsample and LGBM config as exp_005.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"


def run_probe(parquet_path, label):
    train = pd.read_parquet(parquet_path)
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

    class_counts_all = train["Pitch_ID"].value_counts()
    N_SPLITS = 2
    keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
    t = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

    X = t[feat_cols].values
    y_raw = t["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
    acc = accuracy_score(y, oof)
    print(f"{label}: n={len(t)} classes={n_classes} n_feats={len(feat_cols)} OOF acc={acc:.4f}")
    return acc


acc_v1 = run_probe(OUT_DIR + "train_grid_features_probe.parquet", "grid v1 (exp_005, exact lag)")
acc_v2 = run_probe(OUT_DIR + "train_grid_v2_features_probe.parquet", "grid v2 (exp_014, windowed lag)")
print(f"delta (v2 - v1): {acc_v2 - acc_v1:+.4f}")
