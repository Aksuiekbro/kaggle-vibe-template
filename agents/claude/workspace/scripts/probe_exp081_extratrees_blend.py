"""exp_081 (own, probe): does blending ExtraTrees (confirmed +0.0017 topq
over XGBoost-alone on grid features, exp_080 full CV) into the banked best
pipeline (XGBoost + more_oversample augmentation + pseudo-label@thresh=0.50,
sub_027 lineage, full-CV topq 0.9674) add ensemble diversity, or does it
plateau flat like exp_079's grid+aug/grid+scattering proba-blend did (w=0.4-
0.6 flat, +0.0000 vs best single)? ExtraTrees is a genuinely different model
family (bagged full-random-split trees vs boosted trees), so its error
correlation with XGBoost may be lower than the intra-boosting/intra-feature
blends already tested.

Full-data single 80/20 stratified holdout (not a row-subsample) per
exp_012/040's fidelity lesson. Fold-safe: augmentation and pseudo-labeling
are both generated only from the train split.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
PL_THRESH = 0.50
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
BLEND_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)


def make_xgb(n_classes):
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


t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())

# adversarial P(is_test), for the topq shift-aware gating metric
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
print(f"adversarial AUC (grid features): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

tr_idx, va_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
)
X_va, y_va = X[va_idx], y[va_idx]
w_va = w[va_idx]
q75 = np.quantile(w_va, 0.75)
top_mask_va = w_va >= q75

tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
rng = np.random.default_rng(SEED)
aug_df = augment_rows(rare_tr_rows, rng, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X[tr_idx], aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y[tr_idx], le.transform(aug_df["Pitch_ID"].values)])
print(f"train {len(tr_idx)} rows -> {len(X_tr_aug)} after rare-class augmentation ({time.time()-t0:.0f}s)", flush=True)

# --- banked pipeline: XGBoost + aug + pseudo-label(thresh=0.50) ---
t1 = time.time()
clf_base = make_xgb(n_classes)
clf_base.fit(X_tr_aug, y_tr_aug)
test_proba = clf_base.predict_proba(X_test_real)
test_conf = test_proba.max(axis=1)
test_pred = test_proba.argmax(axis=1)
pl_mask = test_conf >= PL_THRESH
X_pl, y_pl = X_test_real[pl_mask], test_pred[pl_mask]
X_tr_pl = np.vstack([X_tr_aug, X_pl])
y_tr_pl = np.concatenate([y_tr_aug, y_pl])
clf_xgb_pl = make_xgb(n_classes)
clf_xgb_pl.fit(X_tr_pl, y_tr_pl)
proba_xgb = clf_xgb_pl.predict_proba(X_va)
print(f"XGBoost+aug+PL(thr={PL_THRESH}) fit+predict done, {pl_mask.sum()}/{len(test_conf)} pseudo-labeled ({time.time()-t1:.0f}s)", flush=True)

# --- ExtraTrees + aug (no PL, keep it simple for this diversity probe) ---
t2 = time.time()
clf_et = make_et()
clf_et.fit(X_tr_aug, y_tr_aug)
proba_et = clf_et.predict_proba(X_va)
print(f"ExtraTrees+aug fit+predict done ({time.time()-t2:.0f}s)", flush=True)


def score(proba, name):
    pred = proba.argmax(axis=1)
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask_va.sum()})", flush=True)
    return topq


print("\n=== exp_081 probe: XGBoost+aug+PL(0.50) vs ExtraTrees+aug vs proba-blend ===", flush=True)
topq_xgb = score(proba_xgb, "XGBoost+aug+PL(thr=0.50) alone (banked pipeline)")
topq_et = score(proba_et, "ExtraTrees+aug alone")

best_w, best_topq = None, -1.0
for wgt in BLEND_WEIGHTS:
    proba_blend = wgt * proba_xgb + (1 - wgt) * proba_et
    topq_b = score(proba_blend, f"blend(w_xgb={wgt})")
    if topq_b > best_topq:
        best_w, best_topq = wgt, topq_b

print(f"\nbest blend weight (w_xgb): {best_w}, topq={best_topq:.4f}", flush=True)
print(f"delta vs XGBoost+aug+PL alone: {best_topq - topq_xgb:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
