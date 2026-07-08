"""exp_075 (own, probe before full CV per C13): does pseudo-labeling
(thresh=0.60, exp_058's confirmed-best threshold) compose with exp_073/074's
confirmed grid+scattering feature win (full CV topq 0.9657 vs grid-only
0.9434, delta +0.0223)?

Single 80/20 holdout (same split/seed as exp_035-074), no augmentation
(scattering isn't augment-compatible, same constraint as the full CV).
Compares:
  1. grid+scattering, no pseudo-label (reference, full-CV topq 0.9657)
  2. grid+scattering + pseudo-label(thresh=0.60)
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import SEED

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
CONF_THRESHOLD = 0.60

REFERENCE_TOPQ_NO_PL = 0.9657  # exp_073/074 full CV, grid+scattering, no aug, no pl


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
feat_cols = grid_feat_cols + scat_feat_cols

train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

X_adv = np.vstack([X, X_test_real])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(X_test_real))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tr_idx, va_idx in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_idx], y_adv[tr_idx])
    p_test_all[va_idx] = clf.predict_proba(X_adv[va_idx])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
w_full = p_test_all[: len(X)]
print(f"adversarial AUC (grid+scattering): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return topq


clf_base = make_xgb(n_classes)
clf_base.fit(X_tr, y_tr)
pred_base = clf_base.predict(X_va)
topq_base = report("grid+scattering, no pseudo-label", pred_base)

test_proba = clf_base.predict_proba(X_test_real)
test_conf = test_proba.max(axis=1)
test_pred = test_proba.argmax(axis=1)
pl_mask = test_conf >= CONF_THRESHOLD
n_pl = int(pl_mask.sum())
X_pl = X_test_real[pl_mask]
y_pl = test_pred[pl_mask]
print(f"{n_pl}/{len(test_conf)} real test rows pseudo-labeled at thresh={CONF_THRESHOLD} ({time.time()-t0:.0f}s)", flush=True)

X_tr_pl = np.vstack([X_tr, X_pl])
y_tr_pl = np.concatenate([y_tr, y_pl])
clf_pl = make_xgb(n_classes)
clf_pl.fit(X_tr_pl, y_tr_pl)
pred_pl = clf_pl.predict(X_va)
topq_pl = report(f"grid+scattering + pseudo-label({CONF_THRESHOLD})", pred_pl)

print(f"\ndelta topq (this probe, pl vs no-pl):     {topq_pl - topq_base:+.4f}  <- coordinator's gating metric", flush=True)
print(f"delta topq vs exp_073/074 full-CV reference ({REFERENCE_TOPQ_NO_PL}): {topq_pl - REFERENCE_TOPQ_NO_PL:+.4f}", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
