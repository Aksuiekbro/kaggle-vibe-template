"""exp_041 (own, scheduler id exp_041, filename exp_042): full 3-fold CV of
test-time augmentation (TTA) for LGBM-alone + exp_037's more_oversample
rare-class train-side augmentation, scored under the coordinator's
shift-aware topq metric.

exp_041's probe (single 80/20 holdout, exp042_probe.log) found a small
positive plain-accuracy signal (0.9099 -> 0.9142, +0.0043) but did not
measure weighted/topq. Promoting to fold-safe 3-fold CV with the full
plain/weighted/topq breakdown before treating TTA as submission-worthy,
same bar as every other lever this competition.

Fold-safe: train-side augmentation is generated only from each fold's TRAIN
split; TTA variants are generated only from each fold's HELD-OUT clips'
own audio (no label leakage -- held-out labels are never used to fit
anything).
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import extract_from_signal, RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows, augment_variants

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
# Same gentler TTA-only perturbation set as the probe -- TTA variants must
# stay close to the original clip's class-relevant content.
TTA_SNRS = (20.0, 12.0)
TTA_RATES = (0.95, 1.05)

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

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

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

# --- 2. fold-safe 3-fold CV: no-TTA vs TTA, both on more_oversample-augmented LGBM ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
rng_tta = np.random.default_rng(SEED + 1)
pred_no_tta = np.zeros(len(y), dtype=int)
proba_tta_avg = np.zeros((len(y), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t_fold = time.time()
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr = y[tr_idx]

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf.fit(X_tr_aug, y_tr_aug)
    fold_classes = clf.classes_

    proba_orig = clf.predict_proba(X_va)
    full_proba_orig = np.zeros((len(va_idx), n_classes))
    full_proba_orig[:, fold_classes] = proba_orig
    pred_no_tta[va_idx] = fold_classes[np.argmax(proba_orig, axis=1)]
    print(f"fold {fold} fit done ({time.time()-t_fold:.0f}s), "
          f"no_tta_acc={accuracy_score(y[va_idx], pred_no_tta[va_idx]):.4f}", flush=True)

    t_tta = time.time()
    va_rows = train.iloc[va_idx]
    tta_sum = full_proba_orig.copy()
    tta_count = np.ones(len(va_idx))
    for i, path in enumerate(va_rows["Path"].values):
        variants, sr = augment_variants(path, rng_tta, TTA_SNRS, TTA_RATES)
        feats_list = [extract_from_signal(v, sr) for v in variants]
        Xv = pd.DataFrame(feats_list)[feat_cols].values
        pv = clf.predict_proba(Xv)
        full_pv = np.zeros((len(variants), n_classes))
        full_pv[:, fold_classes] = pv
        tta_sum[i] += full_pv.sum(axis=0)
        tta_count[i] += len(variants)
        if (i + 1) % 200 == 0:
            print(f"  fold {fold} TTA progress: {i+1}/{len(va_idx)} ({time.time()-t_tta:.0f}s)", flush=True)
    proba_tta_avg[va_idx] = tta_sum / tta_count[:, None]
    pred_tta_fold = fold_classes[np.argmax(proba_tta_avg[va_idx][:, fold_classes], axis=1)]
    print(f"fold {fold} TTA done ({time.time()-t_tta:.0f}s), "
          f"tta_acc={accuracy_score(y[va_idx], pred_tta_fold):.4f}", flush=True)

pred_tta = np.argmax(proba_tta_avg, axis=1)
correct_no_tta = (pred_no_tta == y).astype(float)
correct_tta = (pred_tta == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== exp_041 full CV: LGBM+more_oversample, no-TTA vs TTA ===", flush=True)
plain_n, weighted_n, topq_n = report("no-TTA", correct_no_tta)
plain_t, weighted_t, topq_t = report("TTA", correct_tta)
print(f"\ndelta plain:    {plain_t - plain_n:+.4f}", flush=True)
print(f"delta weighted: {weighted_t - weighted_n:+.4f}", flush=True)
print(f"delta topq:     {topq_t - topq_n:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DIR + "exp041_pred_no_tta.npy", pred_no_tta)
np.save(DIR + "exp041_pred_tta.npy", pred_tta)
np.save(DIR + "exp041_labels.npy", y)
