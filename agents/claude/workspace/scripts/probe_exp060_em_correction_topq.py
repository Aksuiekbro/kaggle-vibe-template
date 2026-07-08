"""exp_060 (scheduler id exp_059), part 2: measure the actual topq delta from
Saerens/Latinne/Decaestecker EM prior-shift correction, on top of the same
XGBoost+aug baseline used throughout (exp_048/057/058/059).

probe_exp060_em_prior_shift.py already confirmed there IS a real prior shift
worth correcting (KL=0.30, one class at 41x train prior, 6.2% of real-test
predictions flip under correction) but only checked in-sample prediction
flips -- no topq metric. This script:

  1. Fits the stage-1 model on the same 80/20 train+aug split as exp_057/058/059.
  2. Runs EM on the model's posteriors over the REAL unlabeled test set only
     (no labels touched -- fold/label-safe).
  3. Applies the resulting prior-ratio correction to the VALIDATION fold's
     own posteriors (same feature distribution shift applies there, weighted
     by the adversarial w used for topq) and re-derives predictions.
  4. Reports plain/weighted/topq delta vs the uncorrected baseline, using the
     same q75-of-w topq definition as every other experiment in this family.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
MAX_EM_ITERS = 20
EM_TOL = 1e-6

# baseline reference: identical split/seed/features/model as exp_048/057/058/059
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def em_prior(proba, train_prior, max_iters=MAX_EM_ITERS, tol=EM_TOL):
    prior_hat = train_prior.copy()
    for it in range(max_iters):
        ratio = prior_hat / np.clip(train_prior, 1e-12, None)
        reweighted = proba * ratio[None, :]
        reweighted /= reweighted.sum(axis=1, keepdims=True)
        new_prior = reweighted.mean(axis=0)
        delta = np.abs(new_prior - prior_hat).max()
        prior_hat = new_prior
        if delta < tol:
            break
    return prior_hat, it + 1, delta


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

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048/057/058/059) ---
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
w_full = p_test_all[: len(X)]
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- single 80/20 holdout (same split/seed as exp_048/057/058/059) ---
idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_va = w_full[va_idx]

rng_master = np.random.default_rng(SEED)
tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug)} rows ({len(aug_df)} augmented)", flush=True)

q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

clf = make_xgb(n_classes)
clf.fit(X_tr_aug, y_tr_aug)
proba_test_real = clf.predict_proba(X_test_real)
proba_va = clf.predict_proba(X_va)
pred_va_uncorrected = proba_va.argmax(axis=1)
print(f"\nstage-1 model fit ({time.time()-t0:.0f}s)", flush=True)

report("uncorrected (this holdout's own fit)", pred_va_uncorrected)

# train prior from the (unaugmented) true train label counts, EM estimated
# ONLY from the real unlabeled test set's posteriors -- no labels touched.
train_prior = np.bincount(y_tr, minlength=n_classes) / len(y_tr)
prior_hat, n_iter, final_delta = em_prior(proba_test_real, train_prior)
print(f"EM converged in {n_iter} iterations (max |delta|={final_delta:.2e}) using real test set only", flush=True)

# apply the SAME learned ratio to the validation fold's own posteriors
ratio = prior_hat / np.clip(train_prior, 1e-12, None)
proba_va_corrected = proba_va * ratio[None, :]
proba_va_corrected /= proba_va_corrected.sum(axis=1, keepdims=True)
pred_va_corrected = proba_va_corrected.argmax(axis=1)

print("\n=== exp_060: EM prior-shift correction applied to validation posteriors ===", flush=True)
plain_c, weighted_c, topq_c = report("EM-corrected", pred_va_corrected)
plain_u, weighted_u, topq_u = report("uncorrected (repeat)", pred_va_uncorrected)
print(f"\ndelta plain    vs uncorrected: {plain_c - plain_u:+.4f}", flush=True)
print(f"delta weighted vs uncorrected: {weighted_c - weighted_u:+.4f}", flush=True)
print(f"delta topq     vs uncorrected: {topq_c - topq_u:+.4f}  <- coordinator's gating metric", flush=True)
print(f"delta topq     vs BASELINE_TOPQ ref: {topq_c - BASELINE_TOPQ:+.4f}", flush=True)
print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
