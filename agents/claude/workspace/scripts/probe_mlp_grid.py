"""exp_008 probe: MLP (StandardScaler, no PCA at this tiny scale) on grid-harmonic
features vs exp_005's LGBM on the same probe subsample.

Motivation: gemini's independent convergence on a similar grid-harmonic feature
family (cross-reviewed sub_018) found MLP >> LGBM on those features (single MLP
CV ~0.94 vs their LGBM baseline far lower). C7 (no approach monoculture): our
grid-harmonic features have only been tried through one model family (LGBM,
exp_005/exp_007). Probe whether an MLP on the *same* features gets a similar
lift before committing a full-scale run.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y), dtype=int)
oof_mlp = np.zeros(len(y), dtype=int)
for tr_idx, va_idx in skf.split(X, y):
    sc = StandardScaler().fit(X[tr_idx])
    Xtr, Xva = sc.transform(X[tr_idx]), sc.transform(X[va_idx])

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=len(np.unique(y)), random_state=42, verbosity=-1, n_jobs=1,
    )
    clf_lgb.fit(X[tr_idx], y[tr_idx])
    oof_lgb[va_idx] = clf_lgb.predict(X[va_idx])

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
        early_stopping=True, random_state=42,
    )
    clf_mlp.fit(Xtr, y[tr_idx])
    oof_mlp[va_idx] = clf_mlp.predict(Xva)

acc_lgb = accuracy_score(y, oof_lgb)
acc_mlp = accuracy_score(y, oof_mlp)
print(f"n={len(train)} classes={len(np.unique(y))} n_feats={len(feat_cols)}")
print(f"LGBM  OOF acc: {acc_lgb:.4f}  (exp_005 reference)")
print(f"MLP   OOF acc: {acc_mlp:.4f}")
print(f"delta (mlp - lgbm): {acc_mlp - acc_lgb:+.4f}")
