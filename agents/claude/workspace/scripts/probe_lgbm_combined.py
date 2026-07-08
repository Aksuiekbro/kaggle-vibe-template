"""exp_007 probe: LGBM on grid-harmonic (exp_005) + normalized-librosa (exp_004) features combined,
vs each feature set alone -- does generic spectral/chroma add signal on top of the grid features?"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
N_SPLITS = 2


def run(df, label):
    feat_cols = [c for c in df.columns if c not in ("Path", "Pitch_ID")]
    class_counts_all = df["Pitch_ID"].value_counts()
    keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
    df = df[df["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

    X = df[feat_cols].values
    y_raw = df["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=len(np.unique(y)), random_state=42, verbosity=-1, n_jobs=1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])

    acc = accuracy_score(y, oof_pred)
    print(f"{label}: n={len(df)} classes={len(np.unique(y))} n_feats={len(feat_cols)} OOF acc={acc:.4f}")
    return acc


norm = pd.read_parquet(OUT_DIR + "train_features_norm_probe.parquet")
grid = pd.read_parquet(OUT_DIR + "train_grid_features_probe.parquet")

grid_only_cols = [c for c in grid.columns if c not in ("Path", "Pitch_ID")]
combined = norm.merge(grid[["Path"] + grid_only_cols], on="Path", how="inner")
assert len(combined) == len(norm) == len(grid)

acc_norm = run(norm, "normalized-librosa only (exp_004 probe feats)")
acc_grid = run(grid, "grid-harmonic only (exp_005 probe feats)")
acc_combined = run(combined, "combined (norm-librosa + grid-harmonic)")

print(f"\ndelta (combined - grid only): {acc_combined - acc_grid:+.4f}")
print(f"delta (combined - norm only): {acc_combined - acc_norm:+.4f}")
