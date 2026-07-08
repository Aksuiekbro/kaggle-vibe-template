"""exp_067: LightGBM with few-shot-tuned hyperparameters vs its own defaults,
on the same grid-harmonic features/split/augmentation as exp_048-066.

Source: arxiv 2411.04324 (Gradient Boosting Trees and LLMs for Tabular Data
Few-Shot Learning) -- identifies that LightGBM's default min_data_in_leaf=20
blocks node splitting whenever a candidate child node would hold fewer than
20 training rows, which is true for most splits touching any of this
competition's 82 classes (3-56 samples/class train, 14 classes <10). Their
recommended few-shot config: min_data_in_leaf=1, num_leaves=4,
extra_trees=True, feature_fraction=0.5, bagging_fraction=0.5.

This is NOT a re-test of the closed model-family axis (exp_040 compared
LightGBM's untouched defaults against XGBoost/CatBoost) or the closed
XGBoost-hyperparameter axis (exp_053/054 swept depth/n_est/lr only, and
XGBoost's own default min_child_weight=1 is already permissive -- the paper's
diagnosed blocker is specific to LightGBM's much stricter default).

Mechanism hypothesis: LightGBM may have lost to XGBoost here partly because
its default leaf-size floor mechanically prevents it from ever isolating the
thin classes, not because of a genuine inductive-bias gap. If so, the
few-shot config should move rare_acc specifically, even if plain/topq are
flat or mixed for the majority classes.

Risk (flagged before running): num_leaves=4 is a large capacity cut; the
closed LGBM capacity-tuning axis (exp_010/012) found num_leaves tuning to be
a dead lever in isolation, but never combined with min_data_in_leaf=1 --
this run tests the combination the paper reports, not num_leaves alone.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

# baseline reference: identical split/seed/features/model as exp_048-066 (XGBoost)
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes_ids = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes_ids)}/{len(full_counts)}", flush=True)

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
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes_ids)]
aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])
print(f"train: {len(X_tr_aug)} rows ({len(aug_df)} augmented)", flush=True)

is_rare_va = np.array([le.classes_[c] in rare_classes_ids for c in y_va])
q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    rare_acc = correct[is_rare_va].mean() if is_rare_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} "
          f"rare_acc={rare_acc:.4f} (n_topq={top_mask.sum()}, n_rare={is_rare_va.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

print("\n=== exp_067a: LightGBM, stock defaults (num_leaves=31, min_data_in_leaf=20) ===", flush=True)
clf_default = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
)
clf_default.fit(X_tr_aug, y_tr_aug)
pred_default = clf_default.predict(X_va)
plain_d, weighted_d, topq_d = report("lgbm_default", pred_default)
print(f"delta topq vs exp_048-ref: {topq_d - BASELINE_TOPQ:+.4f}", flush=True)
print(f"fit time so far: {time.time()-t0:.0f}s", flush=True)

print("\n=== exp_067b: LightGBM, few-shot config (arxiv 2411.04324) ===", flush=True)
clf_fewshot = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05,
    num_leaves=4, min_data_in_leaf=1, extra_trees=True,
    feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1,
    random_state=42, verbosity=-1, n_jobs=1,
)
clf_fewshot.fit(X_tr_aug, y_tr_aug)
pred_fewshot = clf_fewshot.predict(X_va)
plain_f, weighted_f, topq_f = report("lgbm_fewshot", pred_fewshot)
print(f"delta topq vs exp_048-ref (XGBoost baseline):  {topq_f - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)
print(f"delta topq vs lgbm_default (own ablation):      {topq_f - topq_d:+.4f}", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
