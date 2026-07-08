"""exp_044 (own, scheduler id exp_044): soft-vote probability blend of
LGBM-alone+more_oversample-aug (exp_037/038) and XGBoost-alone+more_oversample-
aug (exp_042), the first GBDT-only ensemble tried this competition.

Every prior blend (sub_022/024, exp_025/030/032) mixed LGBM with an MLP and
lost 0.02-0.04 on LB despite inflating plain CV; exp_031's diagnostic pinned
this on MLP's own error pattern dominating the blend at w_lgb=0.4, not
blending per se. Two independently full-CV-validated tree models with
different inductive biases (leaf-wise vs level-wise/histogram growth) may
combine additively where tree+NN did not.

Single 80/20 full-data holdout probe (per exp_012 fidelity lesson), reusing
exp_037/038/042's augment_rows + rare-class threshold. Sweeps blend weight
w_lgb in (0.3, 0.5, 0.7) against both LGBM-alone and XGBoost-alone references,
scored on plain/weighted/topq (coordinator's shift-aware gating metric).
"""
import sys
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
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


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

    t1 = time.time()
    clf_lgb = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_lgb.fit(X_tr_aug, y_tr_aug)
    proba_lgb = clf_lgb.predict_proba(X_va)
    print(f"LGBM+aug fit+predict done ({time.time()-t1:.0f}s)", flush=True)

    t2 = time.time()
    clf_xgb = make_xgb(n_classes)
    clf_xgb.fit(X_tr_aug, y_tr_aug)
    proba_xgb = clf_xgb.predict_proba(X_va)
    print(f"XGBoost+aug fit+predict done ({time.time()-t2:.0f}s)", flush=True)

    def report(name, correct):
        plain = correct.mean()
        weighted = np.average(correct, weights=w_va)
        topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
        print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask_va.sum()})", flush=True)
        return plain, weighted, topq

    print("\n=== exp_044: LGBM+aug-alone / XGBoost+aug-alone / soft-vote blend ===", flush=True)
    pred_lgb = proba_lgb.argmax(axis=1)
    pred_xgb = proba_xgb.argmax(axis=1)
    plain_l, weighted_l, topq_l = report("LGBM+aug-alone", (pred_lgb == y_va).astype(float))
    plain_x, weighted_x, topq_x = report("XGBoost+aug-alone", (pred_xgb == y_va).astype(float))

    best_topq_ref = max(topq_l, topq_x)
    for w_lgb in (0.3, 0.5, 0.7):
        proba_blend = w_lgb * proba_lgb + (1 - w_lgb) * proba_xgb
        pred_blend = proba_blend.argmax(axis=1)
        plain_b, weighted_b, topq_b = report(f"blend w_lgb={w_lgb}", (pred_blend == y_va).astype(float))
        print(f"  delta topq vs best single model: {topq_b - best_topq_ref:+.4f}", flush=True)

    print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
