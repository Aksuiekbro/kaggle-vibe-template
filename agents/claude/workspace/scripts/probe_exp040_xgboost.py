"""exp_040 (own, probe): XGBoost-alone vs LGBM-alone on the existing grid-
harmonic features, full-data single 80/20 stratified holdout (per exp_012's
fidelity lesson for this dataset: full rows, fewer folds, not a row
subsample -- row-subsampled probes have previously flipped sign on this
82-class/2330-row data).

A genuinely new model-family axis: XGBoost's histogram-based, level-wise
tree growth and different default L1/L2 regularization is a different
mechanism from LightGBM's leaf-wise growth, not a hyperparameter variant of
an already-tried family. Motivation: LGBM-alone (sub_020) is this
competition's only consistently shift-transferring submission so far, while
every LGBM+MLP blend variant has been shift-fragile -- testing whether a
different base learner is MORE or LESS shift-robust than LGBM by the same
adversarial-quartile (topq) metric used to gate exp_033/037's submissions.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SEED = 42

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values

# --- adversarial P(is_test) per train row, for the topq shift-aware metric ---
X_adv = np.vstack([X, test[feat_cols].values])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(test))])
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
w = p_test_all[: len(X)]
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s), n_topq={top_mask.sum()}", flush=True)

tr_idx, va_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w[va_idx]
top_mask_va = top_mask[va_idx]


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq_va={top_mask_va.sum()})", flush=True)
    return plain, weighted, topq


t1 = time.time()
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
)
clf_lgb.fit(X_tr, y_tr)
pred_lgb = clf_lgb.predict(X_va)
print(f"LGBM fit+predict done ({time.time()-t1:.0f}s)", flush=True)

t2 = time.time()
clf_xgb = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
    num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
    verbosity=0,
)
clf_xgb.fit(X_tr, y_tr)
pred_xgb = clf_xgb.predict(X_va)
print(f"XGBoost fit+predict done ({time.time()-t2:.0f}s)", flush=True)

print(f"\n=== exp_040: XGBoost-alone vs LGBM-alone, grid features, full-data 80/20 holdout ===", flush=True)
plain_l, weighted_l, topq_l = report("LGBM-alone", pred_lgb)
plain_x, weighted_x, topq_x = report("XGBoost-alone", pred_xgb)
print(f"\ndelta plain:    {plain_x - plain_l:+.4f}", flush=True)
print(f"delta weighted: {weighted_x - weighted_l:+.4f}", flush=True)
print(f"delta topq:     {topq_x - topq_l:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
