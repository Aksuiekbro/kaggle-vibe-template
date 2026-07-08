"""exp_004 probe: same LGBM probe recipe as probe_lgbm.py but on RMS-normalized features,
compared directly against the non-normalized probe on the identical subsample."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"


def run(suffix, label):
    train = pd.read_parquet(OUT_DIR + f"train_features{suffix}_probe.parquet")
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
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])

    acc = accuracy_score(y, oof_pred)
    print(f"{label}: n={len(train)} classes={len(np.unique(y))} OOF acc={acc:.4f}")
    return acc


acc_plain = run("", "plain (exp_002 probe features)")
acc_norm = run("_norm", "RMS-normalized (exp_004 probe features)")
print(f"\ndelta (norm - plain): {acc_norm - acc_plain:+.4f}")
