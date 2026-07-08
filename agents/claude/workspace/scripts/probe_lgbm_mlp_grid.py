"""exp_013 probe: LGBM + MLP soft-vote (SVM dropped per exp_008's cost blowup)
on grid-harmonic features vs exp_005's LGBM-only baseline, same probe subsample.

C7 (no monoculture): only LGBM has been tried within-family on grid features.
exp_008 showed MLP alone is far worse (0.2096) but soft-vote with SVM edged
LGBM-only out slightly (+0.0034). This checks whether LGBM+MLP alone (no SVM,
no ~3321-pairwise-classifier full-run cost blowup) preserves that gain.
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
n_classes = len(np.unique(y))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y), dtype=int)
oof_ens = np.zeros(len(y), dtype=int)

for tr_idx, va_idx in skf.split(X, y):
    sc = StandardScaler().fit(X[tr_idx])
    Xtr, Xva = sc.transform(X[tr_idx]), sc.transform(X[va_idx])

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf_lgb.fit(X[tr_idx], y[tr_idx])
    proba_lgb = clf_lgb.predict_proba(X[va_idx])
    oof_lgb[va_idx] = proba_lgb.argmax(axis=1)

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
        early_stopping=True, random_state=42,
    )
    clf_mlp.fit(Xtr, y[tr_idx])
    proba_mlp = clf_mlp.predict_proba(Xva)

    proba_ens = proba_lgb + proba_mlp
    oof_ens[va_idx] = proba_ens.argmax(axis=1)

acc_lgb = accuracy_score(y, oof_lgb)
acc_ens = accuracy_score(y, oof_ens)
print(f"n={len(train)} classes={n_classes} n_feats={len(feat_cols)}")
print(f"LGBM-only OOF acc:     {acc_lgb:.4f}  (exp_005 reference 0.5876)")
print(f"LGBM+MLP soft-vote:    {acc_ens:.4f}")
print(f"delta (ensemble - lgbm-only): {acc_ens - acc_lgb:+.4f}")
