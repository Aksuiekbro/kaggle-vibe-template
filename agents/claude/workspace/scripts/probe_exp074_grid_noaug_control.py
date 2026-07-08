"""exp_074 (own): clean control for exp_073's grid+scattering result.

exp_073's "grid+scattering" variant (topq=0.9658, delta +0.0085 vs
BASELINE_TOPQ=0.9573) is confounded: BASELINE_TOPQ came from a run WITH
rare-class augmentation (n_train=2263), but grid+scattering had augmentation
skipped (scattering features aren't producible by augment_rows ->
extract_from_signal), so it trained on n_train=1864 unaugmented rows. The
+0.0085 delta conflates "added scattering features" with "removed
augmentation" -- two changes, not one.

This script isolates the scattering effect: same split/seed/row-filter
(inner-joined to the scattering test set, same as exp_073) and NO
augmentation, but grid features only. If this control's topq is close to
0.9573 (the augmented reference), scattering added no real signal (the noise
floor already explains grid+scattering's +0.0085). If this control's topq
is well below 0.9658, scattering is doing real work.
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

BASELINE_TOPQ_AUGMENTED = 0.9573
GRID_SCATTERING_TOPQ = 0.9658


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

grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

# same row-filter as exp_073's grid+scattering variant (inner join on Path),
# but only keep grid columns -- isolates the scattering feature contribution.
train = grid_train.merge(scat_train[["Path"]], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"]], on="Path", how="inner")

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[grid_feat_cols].values
X_test_real = test[grid_feat_cols].values

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

idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

clf = make_xgb(n_classes)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_va)

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75
correct = (pred == y_va).astype(float)
plain = correct.mean()
weighted = np.average(correct, weights=w_va)
topq = correct[top_mask].mean()

print(f"grid-only-no-aug (control): n_train={len(X_tr)} n_feats={len(grid_feat_cols)} "
      f"adv_auc={adv_auc:.4f} plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} "
      f"(n_topq={top_mask.sum()}) ({time.time()-t0:.0f}s)", flush=True)
print(f"  delta topq vs augmented grid-baseline ({BASELINE_TOPQ_AUGMENTED}): "
      f"{topq - BASELINE_TOPQ_AUGMENTED:+.4f}  (expected negative -- this isolates the aug-removal cost)", flush=True)
print(f"  delta topq vs grid+scattering ({GRID_SCATTERING_TOPQ}): "
      f"{GRID_SCATTERING_TOPQ - topq:+.4f}  <- scattering's isolated marginal contribution", flush=True)
