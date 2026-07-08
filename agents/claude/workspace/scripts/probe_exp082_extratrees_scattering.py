"""exp_082 (own, probe): does the scattering feature-family gain (confirmed
+0.0223 topq for XGBoost, exp_073/074/079 full CV) transfer to the ExtraTrees
model family (exp_080's confirmed +0.0017 topq over XGBoost-alone on grid
features), or is it XGBoost-specific? Full-data single 80/20 stratified
holdout per exp_012/040's fidelity lesson (row-subsampled probes have
repeatedly flipped sign on this 82-class/2330-row data) -- reuses the
already-extracted grid and scattering-8k parquet files, no new extraction.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
SEED = 42

t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)

variants = {
    "grid-only": (train[grid_feat_cols].values, test[grid_feat_cols].values),
    "grid+scattering": (
        train[grid_feat_cols + scat_feat_cols].values,
        test[grid_feat_cols + scat_feat_cols].values,
    ),
}

# adversarial P(is_test) per variant, for the topq shift-aware metric
adv_w = {}
for name, (X, X_test_real) in variants.items():
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
    adv_w[name] = w
    print(f"[{name}] adversarial AUC: {adv_auc:.4f} n_feats={X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

tr_idx, va_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
)
y_tr, y_va = y[tr_idx], y[va_idx]

results = {}
for name, (X, _) in variants.items():
    w = adv_w[name]
    q75 = np.quantile(w, 0.75)
    top_mask_va = (w >= q75)[va_idx]
    X_tr, X_va = X[tr_idx], X[va_idx]

    t1 = time.time()
    clf_et = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=1, random_state=42,
    )
    clf_et.fit(X_tr, y_tr)
    pred = clf_et.predict(X_va)
    print(f"ExtraTrees fit+predict on {name} done ({time.time()-t1:.0f}s)", flush=True)

    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w[va_idx])
    topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
    results[name] = (plain, weighted, topq)
    print(f"ExtraTrees on {name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask_va.sum()})", flush=True)

plain_g, weighted_g, topq_g = results["grid-only"]
plain_s, weighted_s, topq_s = results["grid+scattering"]
print(f"\n=== exp_082 probe: ExtraTrees, grid-only vs grid+scattering ===", flush=True)
print(f"delta plain:    {plain_s - plain_g:+.4f}", flush=True)
print(f"delta weighted: {weighted_s - weighted_g:+.4f}", flush=True)
print(f"delta topq:     {topq_s - topq_g:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
