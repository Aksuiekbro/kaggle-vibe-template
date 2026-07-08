"""exp_080 full 3-fold CV promotion: ExtraTreesClassifier vs XGBoost-alone on
grid-harmonic features. Probe (full-data single 80/20 holdout) showed
topq +0.0093 (plain +0.0086, weighted +0.1087) for ExtraTrees over the banked
XGBoost-alone family. Same fold-safe pattern and topq gating metric as
exp_040/041/079.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SEED = 42
N_SPLITS = 3

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

# --- adversarial P(is_test) per train row, for the topq shift-aware metric ---
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
w = p_test_all[: len(X)]
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
print(f"adversarial AUC: {adv_auc:.4f} n_feats={len(feat_cols)} ({time.time()-t0:.0f}s)", flush=True)


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def make_et():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=1, random_state=42,
    )


models = {"XGBoost-alone": make_xgb, "ExtraTrees-alone": make_et}
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
preds = {name: np.zeros(len(y), dtype=int) for name in models}

for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
    t_fold = time.time()
    for name, factory in models.items():
        clf = factory()
        clf.fit(X[tr_idx], y[tr_idx])
        preds[name][va_idx] = clf.predict(X[va_idx])
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): " +
          " ".join(f"{name}_acc={accuracy_score(y[va_idx], preds[name][va_idx]):.4f}" for name in models),
          flush=True)

print("\n=== exp_080 full CV: ExtraTrees vs XGBoost-alone, grid features ===", flush=True)
results = {}
for name in models:
    correct = (preds[name] == y).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    results[name] = (plain, weighted, topq)
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)

plain_x, weighted_x, topq_x = results["XGBoost-alone"]
plain_e, weighted_e, topq_e = results["ExtraTrees-alone"]
print(f"\ndelta plain:    {plain_e - plain_x:+.4f}", flush=True)
print(f"delta weighted: {weighted_e - weighted_x:+.4f}", flush=True)
print(f"delta topq:     {topq_e - topq_x:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp080_pred_xgboost.npy", preds["XGBoost-alone"])
np.save(DIR + "exp080_pred_extratrees.npy", preds["ExtraTrees-alone"])
np.save(DIR + "exp080_labels.npy", y)
