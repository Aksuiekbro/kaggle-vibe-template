"""exp_060 (scheduler id exp_059): cheap probe for Saerens/Latinne/Decaestecker
EM label-prior-shift correction. Fits ONE full XGBoost+aug model (exp_042's
recipe) on all train data, gets predict_proba on the real unlabeled test set,
then runs the EM procedure to estimate the test set's class prior -- no
labels touched, this only checks whether there IS room for prior-shift
correction to matter before spending a full fold-safe CV run.

If the EM-estimated test prior is close to the train prior (small KL /
max-ratio), the shift is not a label-prior effect and this closes fast at
near-zero further cost.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
MAX_EM_ITERS = 20
EM_TOL = 1e-6

t0 = time.time()
train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
ytr = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
Xtr = train[feat_cols].values
Xte = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
rare_rows = train[train["Pitch_ID"].isin(rare_classes)]
rng = np.random.default_rng(SEED)
aug_df = augment_rows(rare_rows, rng, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
Xtr_aug = np.vstack([Xtr, aug_df[feat_cols].values])
ytr_aug = np.concatenate([ytr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train rows: {len(Xtr)} (+{len(aug_df)} aug), classes: {n_classes} ({time.time()-t0:.0f}s)", flush=True)

clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
    num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
    verbosity=0,
)
t1 = time.time()
clf.fit(Xtr_aug, ytr_aug)
proba_test = clf.predict_proba(Xte)  # (583, 82), base-classifier posteriors P(y|x, train_prior)
print(f"fit+predict_proba done ({time.time()-t1:.0f}s)", flush=True)

# train prior: label encoder order, from the (unaugmented) true train label counts
train_prior = np.bincount(ytr, minlength=n_classes) / len(ytr)

# Saerens/Latinne/Decaestecker EM: alternately re-weight posteriors by
# (prior_hat / train_prior) and re-estimate prior_hat as the mean posterior.
prior_hat = train_prior.copy()
for it in range(MAX_EM_ITERS):
    ratio = prior_hat / np.clip(train_prior, 1e-12, None)
    reweighted = proba_test * ratio[None, :]
    reweighted /= reweighted.sum(axis=1, keepdims=True)
    new_prior = reweighted.mean(axis=0)
    delta = np.abs(new_prior - prior_hat).max()
    prior_hat = new_prior
    if delta < EM_TOL:
        break
print(f"EM converged in {it+1} iterations (max |delta|={delta:.2e})", flush=True)

kl = np.sum(prior_hat * np.log(np.clip(prior_hat, 1e-12, None) / np.clip(train_prior, 1e-12, None)))
max_ratio = (prior_hat / np.clip(train_prior, 1e-12, None)).max()
min_ratio = (prior_hat / np.clip(train_prior, 1e-12, None)).min()
l1 = np.abs(prior_hat - train_prior).sum()

print(f"\n=== exp_060 probe: EM-estimated test prior vs train prior ===", flush=True)
print(f"KL(test_hat || train): {kl:.4f}", flush=True)
print(f"L1 distance: {l1:.4f} (max possible 2.0)", flush=True)
print(f"prior ratio range: [{min_ratio:.3f}, {max_ratio:.3f}] (1.0 = no shift)", flush=True)

top_movers = np.argsort(-np.abs(prior_hat - train_prior))[:10]
print("\ntop 10 classes by |prior_hat - train_prior|:", flush=True)
for idx in top_movers:
    cls = le.inverse_transform([idx])[0]
    print(f"  class {cls}: train_prior={train_prior[idx]:.4f} test_prior_hat={prior_hat[idx]:.4f} "
          f"ratio={prior_hat[idx]/max(train_prior[idx],1e-12):.2f}", flush=True)

# quick plain-accuracy sanity check on TRAIN (in-sample, just to confirm reweighting
# doesn't flip predictions pathologically -- not a CV metric, just a sanity gate)
base_pred = proba_test.argmax(axis=1)
ratio_final = prior_hat / np.clip(train_prior, 1e-12, None)
reweighted_final = proba_test * ratio_final[None, :]
reweighted_final /= reweighted_final.sum(axis=1, keepdims=True)
corrected_pred = reweighted_final.argmax(axis=1)
frac_changed = (base_pred != corrected_pred).mean()
print(f"\nfraction of real-test predictions that flip under EM correction: {frac_changed:.4f}", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(DATA_DIR + "exp060_proba_test.npy", proba_test)
np.save(DATA_DIR + "exp060_train_prior.npy", train_prior)
np.save(DATA_DIR + "exp060_prior_hat.npy", prior_hat)
