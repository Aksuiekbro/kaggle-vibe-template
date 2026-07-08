"""exp_071 (scheduler id exp_071, filename exp_072): full 3-fold CV of the
even-lower confidence-threshold sweep (0.30/0.40/0.50) for fold-honest
pseudo-labeling. Probe (exp072_probe.log) found all three thresholds tied
at topq=0.9658 (delta +0.0085 vs no-pl), identical to thresh=0.60's probe
number -- no probe-level improvement over the already-banked thresh=0.60
(sub_026/exp_058, full CV topq +0.0137). Running full CV anyway because this
exact axis already reversed once between probe and full (thresh=0.60 probe
ranked below thresh=0.75, but full CV reversed that), so a tied probe does
not guarantee a tied full result.

Fold-safe: augmentation and the stage-1 pseudo-labeling model are both
generated only from each fold's TRAIN split. Base (aug-only) model is fit
once per fold and reused across all three thresholds to avoid redundant fits.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
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
THRESHOLDS = (0.50, 0.40, 0.30)

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
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
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

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
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
pred_base = np.zeros(len(y), dtype=int)
pred_pl = {thr: np.zeros(len(y), dtype=int) for thr in THRESHOLDS}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf_base = make_xgb(n_classes)
    clf_base.fit(X_tr_aug, y_tr_aug)
    pred_base[va_idx] = clf_base.predict(X_va)

    test_proba = clf_base.predict_proba(X_test_real)
    test_conf = test_proba.max(axis=1)
    test_pred = test_proba.argmax(axis=1)

    fold_msg = [f"fold {fold} base done ({time.time()-t_fold:.0f}s): base_acc={accuracy_score(y[va_idx], pred_base[va_idx]):.4f}"]
    for thr in THRESHOLDS:
        t_thr = time.time()
        pl_mask = test_conf >= thr
        n_pl = int(pl_mask.sum())
        X_pl = X_test_real[pl_mask]
        y_pl = test_pred[pl_mask]

        X_tr_pl = np.vstack([X_tr_aug, X_pl])
        y_tr_pl = np.concatenate([y_tr_aug, y_pl])
        clf_pl = make_xgb(n_classes)
        clf_pl.fit(X_tr_pl, y_tr_pl)
        pred_pl[thr][va_idx] = clf_pl.predict(X_va)
        fold_msg.append(
            f"thr={thr}: {n_pl}/{len(test_conf)} pl'd, acc={accuracy_score(y[va_idx], pred_pl[thr][va_idx]):.4f} ({time.time()-t_thr:.0f}s)"
        )
    print(" | ".join(fold_msg), flush=True)

correct_base = (pred_base == y).astype(float)
q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== exp_071 full CV: lower-threshold pseudo-label sweep (0.30/0.40/0.50) ===", flush=True)
plain_b, weighted_b, topq_b = report("XGBoost + more_oversample augment (baseline, no PL)", correct_base)
for thr in THRESHOLDS:
    correct_pl = (pred_pl[thr] == y).astype(float)
    plain_p, weighted_p, topq_p = report(f"XGBoost + aug + pseudo-label(thresh={thr})", correct_pl)
    print(f"  delta topq vs no-pl baseline: {topq_p - topq_b:+.4f}", flush=True)
    print(f"  delta topq vs thresh=0.60 full CV (+0.0137): {(topq_p - topq_b) - 0.0137:+.4f}", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp071_pred_base.npy", pred_base)
for thr in THRESHOLDS:
    np.save(DIR + f"exp071_pred_pl_{thr}.npy", pred_pl[thr])
np.save(DIR + "exp071_labels.npy", y)
