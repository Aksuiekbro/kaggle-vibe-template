"""exp_008 full CV: LGBM + MLP + linear SVM soft-vote ensemble on the full-scale
grid-harmonic features (exp_005's kernel output), StratifiedKFold(3), vs
exp_005's LGBM-only full OOF (0.9086) reference.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))

N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y), dtype=int)
oof_ens = np.zeros(len(y), dtype=int)

fold = 0
t0 = time.time()
for tr_idx, va_idx in skf.split(X, y):
    fold += 1
    sc = StandardScaler().fit(X[tr_idx])
    Xtr, Xva = sc.transform(X[tr_idx]), sc.transform(X[va_idx])

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
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

    clf_svm = SVC(kernel="linear", probability=True, random_state=42)
    clf_svm.fit(Xtr, y[tr_idx])
    proba_svm = clf_svm.predict_proba(Xva)

    proba_ens = proba_lgb + proba_mlp + proba_svm
    oof_ens[va_idx] = proba_ens.argmax(axis=1)

    print(f"fold {fold} done at {time.time()-t0:.0f}s -- "
          f"lgb_fold_acc={accuracy_score(y[va_idx], oof_lgb[va_idx]):.4f} "
          f"ens_fold_acc={accuracy_score(y[va_idx], oof_ens[va_idx]):.4f}")

acc_lgb = accuracy_score(y, oof_lgb)
acc_ens = accuracy_score(y, oof_ens)
print(f"\nn={len(train)} classes={n_classes} n_feats={len(feat_cols)}")
print(f"LGBM-only full OOF acc:      {acc_lgb:.4f}  (exp_005 reference 0.9086)")
print(f"LGBM+MLP+SVM soft-vote OOF:  {acc_ens:.4f}")
print(f"delta (ensemble - lgbm-only): {acc_ens - acc_lgb:+.4f}")
print(f"delta (ensemble - exp_005 ref 0.9086): {acc_ens - 0.9086:+.4f}")
print(f"total time: {time.time()-t0:.0f}s")
