"""exp_040 (own): full 3-fold CV of XGBoost-alone vs LGBM-alone on grid
features (no augmentation, no blend), scored under exp_006/033/038's
shift-aware topq metric (the coordinator's stated gating metric).

exp_040's probe (single 80/20 holdout) found XGBoost-alone beats LGBM-alone
on topq (0.9444 vs 0.9167, d+0.0278) and plain (d+0.0429), but LOSES badly
on the weighted metric (d-0.1005) -- a genuinely new, not-yet-seen
disagreement pattern between the two shift-aware metrics that hasn't shown
up for any LGBM-family variant this competition. Promoting to fold-safe
3-fold CV before trusting either the gain or the weighted-metric warning
sign, per exp_012/019's single-holdout-can-reverse lesson.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SEED = 42
N_SPLITS = 3

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values

# --- 1. adversarial classifier: train vs test, OOF P(is_test) per train row ---
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
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- 2. fold-safe 3-fold CV: LGBM-alone vs XGBoost-alone, no augmentation ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
pred_lgb = np.zeros(len(y), dtype=int)
pred_xgb = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    clf_lgb = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_lgb.fit(X_tr, y_tr)
    pred_lgb[va_idx] = clf_lgb.predict(X_va)

    clf_xgb = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )
    clf_xgb.fit(X_tr, y_tr)
    pred_xgb[va_idx] = clf_xgb.predict(X_va)

    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"lgb_acc={accuracy_score(y[va_idx], pred_lgb[va_idx]):.4f} "
          f"xgb_acc={accuracy_score(y[va_idx], pred_xgb[va_idx]):.4f}", flush=True)

correct_lgb = (pred_lgb == y).astype(float)
correct_xgb = (pred_xgb == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== exp_040 full CV: LGBM-alone vs XGBoost-alone, grid features, no augment ===", flush=True)
plain_l, weighted_l, topq_l = report("LGBM-alone", correct_lgb)
plain_x, weighted_x, topq_x = report("XGBoost-alone", correct_xgb)
print(f"\ndelta plain:    {plain_x - plain_l:+.4f}", flush=True)
print(f"delta weighted: {weighted_x - weighted_l:+.4f}", flush=True)
print(f"delta topq:     {topq_x - topq_l:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp040_pred_lgb.npy", pred_lgb)
np.save(DIR + "exp040_pred_xgb.npy", pred_xgb)
np.save(DIR + "exp040_labels.npy", y)
