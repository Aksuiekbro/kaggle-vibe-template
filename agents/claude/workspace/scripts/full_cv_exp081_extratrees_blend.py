"""exp_081 full 3-fold CV promotion: probe (probe_exp081_extratrees_blend.py,
full-data single 80/20 holdout) found blending ExtraTrees+aug into the banked
XGBoost+aug+PL(thresh=0.50) pipeline (sub_027 lineage) at w_xgb=0.6-0.7 gives
topq +0.0085 over XGBoost+aug+PL alone. This competition has repeated probe-
vs-full sign flips (exp_079 Q-sweep is the most recent, 6th+ occurrence per
STRATEGY.md), so the probe signal must be confirmed on fold-honest 3-fold CV
before being submission-worthy.

Fold-safe: augmentation and pseudo-labeling are both generated only from each
fold's TRAIN split (pseudo-label source model predicts on the real test set,
labels only training data -- never touches the held-out validation fold,
matching exp071's full-CV pattern). ExtraTrees is fit on the same augmented
fold-train data as the XGBoost+PL arm's pre-PL stage, for a fair comparison.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3
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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
proba_xgb = np.zeros((len(y), n_classes))
proba_et = np.zeros((len(y), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_va = X[va_idx]

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X[tr_idx], aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y[tr_idx], le.transform(aug_df["Pitch_ID"].values)])

    # arm 1: banked pipeline -- XGBoost + aug + pseudo-label(thresh=0.50)
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
    proba_xgb[va_idx] = clf_xgb_pl.predict_proba(X_va)

    # arm 2: ExtraTrees + aug (no PL), same fold-train augmented data
    clf_et = make_et()
    clf_et.fit(X_tr_aug, y_tr_aug)
    proba_et[va_idx] = clf_et.predict_proba(X_va)

    acc_xgb = accuracy_score(y[va_idx], proba_xgb[va_idx].argmax(axis=1))
    acc_et = accuracy_score(y[va_idx], proba_et[va_idx].argmax(axis=1))
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"xgb_pl_acc={acc_xgb:.4f} et_acc={acc_et:.4f} pl'd={pl_mask.sum()}/{len(test_conf)}", flush=True)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return topq


print("\n=== exp_081 full CV: XGBoost+aug+PL(0.50) vs ExtraTrees+aug vs proba-blend ===", flush=True)
topq_xgb = report("XGBoost+aug+PL(thr=0.50) alone (banked pipeline)", proba_xgb)
topq_et = report("ExtraTrees+aug alone", proba_et)

best_w, best_topq = None, -1.0
for wgt in BLEND_WEIGHTS:
    proba_blend = wgt * proba_xgb + (1 - wgt) * proba_et
    topq_b = report(f"blend(w_xgb={wgt})", proba_blend)
    if topq_b > best_topq:
        best_w, best_topq = wgt, topq_b

print(f"\nbest blend weight (w_xgb): {best_w}, topq={best_topq:.4f}", flush=True)
print(f"delta vs XGBoost+aug+PL alone: {best_topq - topq_xgb:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp081_proba_xgb.npy", proba_xgb)
np.save(DIR + "exp081_proba_et.npy", proba_et)
np.save(DIR + "exp081_labels.npy", y)
