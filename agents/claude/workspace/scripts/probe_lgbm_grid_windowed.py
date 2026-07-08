"""exp_015 probe: LGBM on exp_005 grid features alone vs grid + temporal-
windowed harmonic features (exp_015), same probe subsample/LGBM config.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

grid = pd.read_parquet(OUT_DIR + "train_grid_features_probe.parquet")
win = pd.read_parquet(OUT_DIR + "train_grid_windowed_features_probe.parquet")
win_feat_cols = [c for c in win.columns if c not in ("Path", "Pitch_ID")]
combined = grid.merge(win[["Path"] + win_feat_cols], on="Path", how="inner")


def run_probe(df, feat_cols, label):
    class_counts_all = df["Pitch_ID"].value_counts()
    N_SPLITS = 2
    keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
    t = df[df["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

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


grid_feat_cols = [c for c in grid.columns if c not in ("Path", "Pitch_ID")]
acc_grid = run_probe(combined, grid_feat_cols, "grid-only (exp_005 ref)")
acc_combined = run_probe(combined, grid_feat_cols + win_feat_cols, "grid + windowed (exp_015)")
print(f"delta (combined - grid-only): {acc_combined - acc_grid:+.4f}")
