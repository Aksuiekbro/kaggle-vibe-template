"""exp_043 (own, probe, idea-label exp_044): CatBoost-alone vs XGBoost-alone
(current best single model, exp_040) on the existing grid-harmonic features,
full-data single 80/20 stratified holdout (per exp_012/040's fidelity lesson:
full rows, fewer folds, not a row subsample).

Tests whether shift-robustness is a general GBDT-family property (ordered
boosting + symmetric trees is architecturally distinct from both LGBM's
leaf-wise and XGBoost's level-wise/histogram growth) or specific to
XGBoost's regularization/split-finding (exp_040 would be a one-off).
"""
import time
import warnings

import catboost as cb
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
clf_xgb = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
    num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
    verbosity=0,
)
clf_xgb.fit(X_tr, y_tr)
pred_xgb = clf_xgb.predict(X_va)
print(f"XGBoost fit+predict done ({time.time()-t1:.0f}s)", flush=True)

t2 = time.time()
clf_cb = cb.CatBoostClassifier(
    iterations=300, learning_rate=0.05, depth=6,
    loss_function="MultiClass", random_seed=42, thread_count=1,
    verbose=False,
)
clf_cb.fit(X_tr, y_tr)
pred_cb = clf_cb.predict(X_va).astype(int).ravel()
print(f"CatBoost fit+predict done ({time.time()-t2:.0f}s)", flush=True)

print(f"\n=== exp_043: CatBoost-alone vs XGBoost-alone, grid features, full-data 80/20 holdout ===", flush=True)
plain_x, weighted_x, topq_x = report("XGBoost-alone", pred_xgb)
plain_c, weighted_c, topq_c = report("CatBoost-alone", pred_cb)
print(f"\ndelta plain:    {plain_c - plain_x:+.4f}", flush=True)
print(f"delta weighted: {weighted_c - weighted_x:+.4f}", flush=True)
print(f"delta topq:     {topq_c - topq_x:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
