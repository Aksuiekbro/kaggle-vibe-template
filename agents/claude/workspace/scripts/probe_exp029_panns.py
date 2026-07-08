"""exp_029 probe: does concatenating a frozen pretrained PANNs (Cnn14,
AudioSet-trained general audio-event CNN) global embedding onto the exp_005
grid-harmonic features add signal? Same 308-row probe subsample and
LGBM config as probe_lgbm_grid.py / probe_exp028_crepe.py for direct
comparability.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"


def run(df, label):
    feat_cols = [c for c in df.columns if c not in ("Path", "Pitch_ID")]

    class_counts_all = df["Pitch_ID"].value_counts()
    N_SPLITS = 2
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


grid = pd.read_parquet(OUT_DIR + "train_grid_features_probe.parquet")
panns = pd.read_parquet(OUT_DIR + "train_panns_features_probe.parquet")
panns_feat_cols = [c for c in panns.columns if c.startswith("panns_")]

grid_plus_panns = grid.merge(panns[["Path"] + panns_feat_cols], on="Path", how="left")

acc_grid = run(grid.copy(), "grid-only (exp_005 reference)")
acc_grid_panns = run(grid_plus_panns.copy(), "grid + PANNs Cnn14 embedding")
acc_panns_only = run(panns.copy(), "PANNs Cnn14 embedding only")

print(f"\ndelta (grid+panns - grid-only): {acc_grid_panns - acc_grid:+.4f}")
print(f"panns-only vs grid-only: {acc_panns_only - acc_grid:+.4f}")
