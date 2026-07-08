"""exp_062: shrinkage-regularized (Dirichlet/James-Stein-style) label-shift
prior correction -- fixes exp_059/exp_060's catastrophic failure mode.

exp_059's raw EM prior-shift correction trusted the EM-estimated per-class
test prior fully and made topq much WORSE (0.9573 -> 0.7949, delta -0.1624),
even though the underlying shift is real (KL=0.30 vs train prior). Diagnosis
recorded at the time: with ~7 test rows/class on average and many classes at
0-3, the EM estimate itself is too noisy to trust outright.

This script keeps the exact same harness/split/model as exp_057/058/059/060
(the reference numbers below) but replaces the "trust the EM estimate fully"
step with a shrinkage estimator: each class's corrected prior is a convex
combination of the noisy EM estimate and the (trustworthy, seen-in-training)
train prior, weighted by how many real-test rows the current model actually
assigns to that class (soft count = sum of predicted proba mass). Classes
with few assigned test rows shrink hard toward the train prior instead of
being corrected on a near-single-sample estimate; classes with many assigned
rows keep most of the EM correction.

    lambda_c = n_c / (n_c + alpha)          # n_c = soft assigned test count
    prior_c  = lambda_c * EM_c + (1 - lambda_c) * train_prior_c

alpha -> 0 recovers raw EM (exp_059/060's failed mechanism, sanity check);
alpha -> inf recovers no correction (baseline, sanity check). Sweep alpha in
{1, 5, 20, 50} to find where shrinkage first recovers a non-negative delta.

Motivated by FMAPLS (arxiv 2511.18615, Dirichlet-regularized dynamic
label-shift estimation) and RLLS (arxiv 1903.09734, regularized label-shift
learning) -- both establish that raw unregularized label-shift MLE/EM is
unstable under small per-class samples and needs exactly this kind of
regularization, i.e. this is the "mechanism that specifically handles
small-sample prior estimation" PLAN_DRAFT.md flagged as the reopening
condition for the domain-adaptation family.
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
ALPHAS = [1.0, 5.0, 20.0, 50.0]

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

plain_u, weighted_u, topq_u = report("uncorrected (this holdout's own fit)", pred_va_uncorrected)

train_prior = np.bincount(y_tr, minlength=n_classes) / len(y_tr)
prior_hat, n_iter, final_delta = em_prior(proba_test_real, train_prior)
print(f"EM converged in {n_iter} iterations (max |delta|={final_delta:.2e}) using real test set only", flush=True)

# soft per-class assigned test count under the CURRENT (uncorrected) model --
# this is the "how much evidence do we actually have for this class" signal
# that drives shrinkage strength.
soft_counts = proba_test_real.sum(axis=0)
print(f"soft test-assigned counts: min={soft_counts.min():.2f} max={soft_counts.max():.2f} "
      f"median={np.median(soft_counts):.2f}", flush=True)

print("\n=== exp_062: shrinkage-regularized EM prior-shift correction (alpha sweep) ===", flush=True)
for alpha in ALPHAS:
    lam = soft_counts / (soft_counts + alpha)
    prior_shrunk = lam * prior_hat + (1 - lam) * train_prior
    prior_shrunk /= prior_shrunk.sum()
    ratio = prior_shrunk / np.clip(train_prior, 1e-12, None)
    proba_va_corrected = proba_va * ratio[None, :]
    proba_va_corrected /= proba_va_corrected.sum(axis=1, keepdims=True)
    pred_va_corrected = proba_va_corrected.argmax(axis=1)
    plain_c, weighted_c, topq_c = report(f"alpha={alpha}", pred_va_corrected)
    print(f"  delta plain={plain_c - plain_u:+.4f} weighted={weighted_c - weighted_u:+.4f} "
          f"topq={topq_c - topq_u:+.4f}  <- coordinator's gating metric", flush=True)

# sanity check: alpha=0 must reproduce exp_059/060's raw-EM result (topq -0.1624)
lam0 = np.ones(n_classes)
ratio0 = prior_hat / np.clip(train_prior, 1e-12, None)
proba_va_raw = proba_va * ratio0[None, :]
proba_va_raw /= proba_va_raw.sum(axis=1, keepdims=True)
pred_va_raw = proba_va_raw.argmax(axis=1)
plain_r, weighted_r, topq_r = report("alpha=0 (raw EM, sanity check vs exp_059/060)", pred_va_raw)
print(f"  delta topq vs uncorrected: {topq_r - topq_u:+.4f} (expect ~ -0.1624 per exp_059/060)", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
