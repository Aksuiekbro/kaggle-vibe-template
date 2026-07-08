"""exp_003 probe: adversarial validation, train-vs-test on audio features.

Usage: ~/ml/bin/python probe_adversarial.py [--full]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
suffix = "" if args.full else "_probe"

train = pd.read_parquet(OUT_DIR + f"train_features{suffix}.parquet")
test = pd.read_parquet(OUT_DIR + f"test_features{suffix}.parquet")

feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
assert feat_cols == [c for c in test.columns if c not in ("Path", "Pitch_ID")]

X_train = train[feat_cols].values
X_test = test[feat_cols].values
X = np.vstack([X_train, X_test])
y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    clf.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

auc = roc_auc_score(y, oof)
print(f"adversarial AUC (train vs test): {auc:.4f}")
print("AUC > 0.6 => distribution shift, reconsider CV scheme")

# feature importance for the shift (top movers), fit on all data
clf = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1, n_jobs=1)
clf.fit(X, y)
imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nTop 10 features driving train/test separability:")
print(imp.head(10))
