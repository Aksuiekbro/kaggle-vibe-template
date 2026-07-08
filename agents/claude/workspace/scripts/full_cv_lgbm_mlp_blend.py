"""exp_017: proper full 3-fold StratifiedKFold CV comparing LGBM-alone,
MLP-alone, and LGBM+MLP soft-vote blend on grid features (2330 rows/82
classes). Confirms whether exp_016's single-holdout result (MLP 0.9227 vs
LGBM 0.9013, delta +0.0215) survives proper CV, and gives real full-scale
evidence for a within-family blend.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))
classes = np.arange(n_classes)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(y), n_classes))
oof_mlp = np.zeros((len(y), n_classes))
pred_lgb = np.zeros(len(y), dtype=int)
pred_mlp = np.zeros(len(y), dtype=int)

t_start = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    t0 = time.time()
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf_lgb.fit(Xtr, ytr)
    proba_lgb = clf_lgb.predict_proba(Xva)
    fold_classes_lgb = clf_lgb.classes_
    full_proba_lgb = np.zeros((len(va_idx), n_classes))
    full_proba_lgb[:, fold_classes_lgb] = proba_lgb
    oof_lgb[va_idx] = full_proba_lgb
    pred_lgb[va_idx] = fold_classes_lgb[np.argmax(proba_lgb, axis=1)]
    acc_lgb_fold = accuracy_score(yva, pred_lgb[va_idx])
    t_lgb = time.time() - t0

    t1 = time.time()
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
        early_stopping=True, random_state=42,
    )
    clf_mlp.fit(Xtr_s, ytr)
    proba_mlp = clf_mlp.predict_proba(Xva_s)
    fold_classes_mlp = clf_mlp.classes_
    full_proba_mlp = np.zeros((len(va_idx), n_classes))
    full_proba_mlp[:, fold_classes_mlp] = proba_mlp
    oof_mlp[va_idx] = full_proba_mlp
    pred_mlp[va_idx] = fold_classes_mlp[np.argmax(proba_mlp, axis=1)]
    acc_mlp_fold = accuracy_score(yva, pred_mlp[va_idx])
    t_mlp = time.time() - t1

    print(f"fold {fold}: LGBM acc={acc_lgb_fold:.4f} ({t_lgb:.0f}s), "
          f"MLP acc={acc_mlp_fold:.4f} ({t_mlp:.0f}s)", flush=True)

acc_lgb_oof = accuracy_score(y, pred_lgb)
acc_mlp_oof = accuracy_score(y, pred_mlp)

# soft-vote blend weight sweep
best_w, best_acc = None, -1
for w in np.arange(0.0, 1.01, 0.1):
    blend_proba = w * oof_lgb + (1 - w) * oof_mlp
    blend_pred = classes[np.argmax(blend_proba, axis=1)]
    acc = accuracy_score(y, blend_pred)
    print(f"blend w_lgb={w:.1f}: acc={acc:.4f}", flush=True)
    if acc > best_acc:
        best_acc, best_w = acc, w

print(f"\nLGBM-only OOF acc: {acc_lgb_oof:.4f}")
print(f"MLP-only OOF acc:  {acc_mlp_oof:.4f}")
print(f"Best blend: w_lgb={best_w:.1f}, acc={best_acc:.4f}")
print(f"delta (best_blend - max(lgb,mlp)): {best_acc - max(acc_lgb_oof, acc_mlp_oof):+.4f}")
print(f"total time: {time.time()-t_start:.0f}s")

np.save(DATA_DIR + "oof_lgb_exp017.npy", oof_lgb)
np.save(DATA_DIR + "oof_mlp_exp017.npy", oof_mlp)
np.save(DATA_DIR + "y_exp017.npy", y)
