"""exp_079 (scheduler id) full 3-fold CV promotion: grid-only vs grid+scattering
Q=8 (banked reference) vs grid+scattering Q=4 (probe winner, +0.0294 topq at
n=34 subsample scale, monotonic across Q in {4, 8, 16}).

Same fold-safe pattern and topq gating metric as exp_073/074's original
grid+scattering full CV. No augmentation in any arm (scattering features are
not producible by augment_rows -> extract_from_signal; orthogonal question).
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import SEED

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
N_SPLITS = 3


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
scat8_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat8_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")
scatq4_train = pd.read_parquet(SCAT_DIR + "train_scattering_q4_features.parquet")
scatq4_test = pd.read_parquet(SCAT_DIR + "test_scattering_q4_features.parquet")

scat8_feat_cols = [c for c in scat8_train.columns if c.startswith("sc_")]
scatq4_feat_cols = [c for c in scatq4_train.columns if c.startswith("scq4_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

# inner-join across all three feature sets so every arm shares the same rows
train = (grid_train
         .merge(scat8_train[["Path"] + scat8_feat_cols], on="Path", how="inner")
         .merge(scatq4_train[["Path"] + scatq4_feat_cols], on="Path", how="inner"))
test = (grid_test
        .merge(scat8_test[["Path"] + scat8_feat_cols], on="Path", how="inner")
        .merge(scatq4_test[["Path"] + scatq4_feat_cols], on="Path", how="inner"))

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)

variants = {
    "grid-only": (train[grid_feat_cols].values, test[grid_feat_cols].values, grid_feat_cols),
    "grid+scattering-Q8": (
        train[grid_feat_cols + scat8_feat_cols].values,
        test[grid_feat_cols + scat8_feat_cols].values,
        grid_feat_cols + scat8_feat_cols,
    ),
    "grid+scattering-Q4": (
        train[grid_feat_cols + scatq4_feat_cols].values,
        test[grid_feat_cols + scatq4_feat_cols].values,
        grid_feat_cols + scatq4_feat_cols,
    ),
}

adv_w = {}
for name, (X, X_test_real, feat_cols) in variants.items():
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
    adv_w[name] = p_test_all[: len(X)]
    print(f"[{name}] adversarial AUC: {adv_auc:.4f} n_feats={len(feat_cols)} ({time.time()-t0:.0f}s)", flush=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
preds = {name: np.zeros(len(y), dtype=int) for name in variants}

for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
    t_fold = time.time()
    for name, (X, _, feat_cols) in variants.items():
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        clf = make_xgb(n_classes)
        clf.fit(X_tr, y_tr)
        preds[name][va_idx] = clf.predict(X_va)
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): " +
          " ".join(f"{name}_acc={accuracy_score(y[va_idx], preds[name][va_idx]):.4f}" for name in variants),
          flush=True)

print("\n=== exp_079 full CV: grid-only vs grid+scattering-Q8 vs grid+scattering-Q4 ===", flush=True)
results = {}
for name in variants:
    correct = (preds[name] == y).astype(float)
    w = adv_w[name]
    q75 = np.quantile(w, 0.75)
    top_mask = w >= q75
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    results[name] = (plain, weighted, topq)
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)

plain_b, weighted_b, topq_b = results["grid-only"]
plain_8, weighted_8, topq_8 = results["grid+scattering-Q8"]
plain_4, weighted_4, topq_4 = results["grid+scattering-Q4"]
print(f"\ndelta topq (Q8 vs grid-only): {topq_8 - topq_b:+.4f}", flush=True)
print(f"delta topq (Q4 vs grid-only): {topq_4 - topq_b:+.4f}", flush=True)
print(f"delta topq (Q4 vs Q8, the axis this experiment tests): {topq_4 - topq_8:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(GRID_DIR + "exp079_pred_grid_only.npy", preds["grid-only"])
np.save(GRID_DIR + "exp079_pred_grid_scattering_q8.npy", preds["grid+scattering-Q8"])
np.save(GRID_DIR + "exp079_pred_grid_scattering_q4.npy", preds["grid+scattering-Q4"])
np.save(GRID_DIR + "exp079_labels.npy", y)
