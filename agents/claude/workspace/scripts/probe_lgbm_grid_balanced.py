"""exp_009 probe: class-balanced LGBM on grid-harmonic features (exp_005 feature set).

Compares plain vs class_weight='balanced' on the same probe subsample/fold setup
used by exp_005/exp_007, to see whether balancing helps minority-class recall
without hurting overall OOF accuracy (classes range 3-56 samples in full data).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"


def run(fname, label, class_weight=None):
    train = pd.read_parquet(OUT_DIR + fname)
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

    class_counts_all = train["Pitch_ID"].value_counts()
    N_SPLITS = 2
    keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
    train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

    X = train[feat_cols].values
    y_raw = train["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=len(np.unique(y)), random_state=42, verbosity=-1, n_jobs=1,
            class_weight=class_weight,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])

    acc = accuracy_score(y, oof_pred)
    bal_acc = balanced_accuracy_score(y, oof_pred)
    print(f"{label}: n={len(train)} classes={len(np.unique(y))} n_feats={len(feat_cols)} "
          f"OOF acc={acc:.4f} balanced_acc={bal_acc:.4f}")
    return acc, bal_acc


acc_plain, bal_plain = run("train_grid_features_probe.parquet", "grid-harmonic plain (exp_005 baseline)")
acc_bal, bal_bal = run("train_grid_features_probe.parquet", "grid-harmonic class_weight=balanced", class_weight="balanced")

print(f"\noverall acc delta (balanced - plain): {acc_bal - acc_plain:+.4f}")
print(f"balanced_accuracy delta (balanced - plain): {bal_bal - bal_plain:+.4f}")
