"""exp_046 (own, scheduler id exp_045): focal-loss custom multiclass objective
for XGBoost on top of exp_042's current-best config (grid features +
more_oversample rare-class augmentation).

Distinct mechanism from exp_009's already-killed class_weight='balanced'
(frequency-based reweighting, probe delta -0.0309): this reweights each
sample's gradient/Hessian by (1 - p_true)^gamma (a hardness/confidence-based
modulating factor), so easy majority-class rows stop dominating the gradient
even without touching class weights directly.

Gradient/Hessian derivation: standard multiclass softmax cross-entropy has
grad_k = p_k - y_k, hess_k = p_k*(1-p_k) (the diagonal approximation XGBoost's
own multi:softprob uses internally). Focal loss multiplies the true-class loss
term by (1-p_true)^gamma; this probe applies that same modulating factor to
every class's gradient/Hessian row (the standard approximate extension used in
public multiclass-focal-loss-for-GBDT implementations, not an exact per-class
Newton derivation, which is intractable in closed form for softmax focal loss).

CAUTION (per RESEARCH.md block-24 risk note): custom multiclass Hessians are
easy to get subtly wrong. Verified empirically first (see shape-probe commit)
that xgboost 3.3.0's native train() API passes/expects (n_samples, num_class)
2D arrays to a custom obj, not a flattened 1D array -- confirmed via a toy
4-class/20-row unit test before writing this. Sanity-gated below: gamma=0
must reduce to (within probe noise) the native multi:softmax objective's
accuracy, since it is then mathematically the same loss (grad_k = p_k - y_k
exactly, modulating factor = 1 everywhere) -- if gamma=0 disagrees sharply
with the native objective, that indicates a plumbing bug and the gamma>0
deltas below should not be trusted.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

XGB_PARAMS = dict(
    eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
    seed=42, nthread=1,
)
N_ROUNDS = 300


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_focal_obj(gamma, num_class):
    def obj(preds, dtrain):
        y = dtrain.get_label().astype(int)
        n = len(y)
        p = softmax(preds)
        y_onehot = np.zeros((n, num_class))
        y_onehot[np.arange(n), y] = 1.0
        p_true = p[np.arange(n), y]
        modulating = (1.0 - p_true) ** gamma
        grad = (p - y_onehot) * modulating[:, None]
        hess = np.clip(p * (1.0 - p), 1e-6, None) * modulating[:, None]
        return grad, hess
    return obj


def fit_predict(X_tr, y_tr, X_va, num_class, gamma):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va)
    params = dict(XGB_PARAMS, num_class=num_class, disable_default_eval_metric=1)
    bst = xgb.train(params, dtrain, num_boost_round=N_ROUNDS, obj=make_focal_obj(gamma, num_class))
    margins = bst.predict(dva, output_margin=True)
    return margins.argmax(axis=1)


def fit_predict_native(X_tr, y_tr, X_va, num_class):
    clf = xgb.XGBClassifier(
        n_estimators=N_ROUNDS, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=num_class, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict(X_va)


if __name__ == "__main__":
    t0 = time.time()
    train = pd.read_parquet(DIR + "train_grid_features.parquet")
    test = pd.read_parquet(DIR + "test_grid_features.parquet")
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

    le = LabelEncoder()
    y = le.fit_transform(train["Pitch_ID"].values)
    n_classes = len(le.classes_)
    X = train[feat_cols].values

    full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
    rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
    print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

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
    print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_va = w[va_idx]
    q75 = np.quantile(w_va, 0.75)
    top_mask_va = w_va >= q75

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    rng_master = np.random.default_rng(SEED)
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    def report(name, pred):
        correct = (pred == y_va).astype(float)
        plain = correct.mean()
        weighted = np.average(correct, weights=w_va)
        topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
        print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask_va.sum()})", flush=True)
        return plain, weighted, topq

    print("\n=== exp_046: focal-loss custom objective vs native multi:softmax, XGBoost+more_oversample-aug ===", flush=True)
    t1 = time.time()
    pred_native = fit_predict_native(X_tr_aug, y_tr_aug, X_va, n_classes)
    print(f"native multi:softmax done ({time.time()-t1:.0f}s)", flush=True)
    plain_ref, weighted_ref, topq_ref = report("native (exp_042 reference)", pred_native)

    t2 = time.time()
    pred_g0 = fit_predict(X_tr_aug, y_tr_aug, X_va, n_classes, gamma=0.0)
    print(f"custom obj gamma=0 done ({time.time()-t2:.0f}s)", flush=True)
    plain_g0, weighted_g0, topq_g0 = report("custom gamma=0 (sanity check)", pred_g0)
    print(f"  SANITY delta vs native (should be ~0, confirms plumbing correct): plain={plain_g0-plain_ref:+.4f} topq={topq_g0-topq_ref:+.4f}", flush=True)

    for gamma in (1.0, 2.0):
        t3 = time.time()
        pred_g = fit_predict(X_tr_aug, y_tr_aug, X_va, n_classes, gamma=gamma)
        print(f"custom obj gamma={gamma} done ({time.time()-t3:.0f}s)", flush=True)
        plain_g, weighted_g, topq_g = report(f"custom gamma={gamma}", pred_g)
        print(f"  delta topq vs native reference: {topq_g - topq_ref:+.4f}", flush=True)

    print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
