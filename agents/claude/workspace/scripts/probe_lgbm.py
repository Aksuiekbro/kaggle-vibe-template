"""exp_002 probe: MFCC+spectral features -> LightGBM multiclass, StratifiedKFold.

Usage: ~/ml/bin/python probe_lgbm.py [--full]
On a subsampled probe, many of the 82 classes have <2 rows and can't be
stratified into folds at all -- they're dropped from the CV-scored subset
(kept in mind for the full run, where per-class counts are 3-56).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
suffix = "" if args.full else "_probe"

train = pd.read_parquet(OUT_DIR + f"train_features{suffix}.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

class_counts_all = train["Pitch_ID"].value_counts()
N_SPLITS = 3 if args.full else 2
keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
dropped = len(class_counts_all) - len(keep_classes)
if dropped:
    print(f"dropping {dropped}/{len(class_counts_all)} classes with <{N_SPLITS} rows (can't stratify)")
train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)

class_counts = pd.Series(y_raw).value_counts()
print(f"n={len(train)} classes={len(class_counts)} min_per_class={class_counts.min()} max_per_class={class_counts.max()}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_pred = np.zeros(len(y), dtype=int)
fold_accs = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = lgb.LGBMClassifier(
        n_estimators=100 if args.full else 40,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        num_class=len(np.unique(y)),
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    oof_pred[va_idx] = pred
    acc = accuracy_score(y_va, pred)
    fold_accs.append(acc)
    print(f"fold {fold}: acc={acc:.4f} (n_va={len(va_idx)})")

overall_acc = accuracy_score(y, oof_pred)
print(f"\nOOF accuracy: {overall_acc:.4f}")
print(f"fold mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}")

baseline = 0.024  # majority-class baseline (exp_001)
print(f"delta vs majority baseline: {overall_acc - baseline:.4f}")
