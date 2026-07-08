"""exp_065 (scheduler id exp_064, own): Confident Sinkhorn Allocation (CSA,
arxiv 2206.05880) for pseudo-label assignment.

Every prior pseudo-labeling variant tried this competition either:
  - took the raw argmax-confidence label above a threshold (exp_052/058,
    the only full-CV-confirmed positive lever so far, topq +0.0137 @0.60), or
  - reweighted the TRAINING loss/posterior globally by an estimated test
    class prior (exp_060 EM, exp_061 shrinkage-EM) -- both collapsed
    catastrophically (topq -0.1624 / -0.0855 to -0.0513), or
  - diffused labels via a kNN graph (exp_063 label propagation) -- flat to
    slightly negative.

CSA is a different mechanism: it does NOT touch model training at all. It
only changes which HARD LABEL gets assigned to an already confidence-
filtered pseudo-label candidate, by solving a small entropic-regularized
optimal-transport problem so the accepted candidate pool's class marginal
matches a target distribution (here: train's own class proportions). This
targets exactly this dataset's severe class imbalance (14/82 classes with
<10 samples) without the global-reweighting instability that sank exp_060/061.

Reuses exp_052/058's exact fold-honest harness (80/20 holdout, SEED,
more_oversample augmentation, XGBoost) for direct comparability to the
banked BASELINE_TOPQ=0.9573 reference and to exp_058's own +0.0171 probe
delta at thresh=0.75 / prior probe deltas at other thresholds.
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

# baseline reference: identical split/seed/features/model as exp_048-063
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

CONF_THRESHOLDS = (0.75, 0.60)  # the two full-CV-relevant thresholds from exp_056/058
SINKHORN_REG = 0.1
SINKHORN_ITERS = 100


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def sinkhorn_assign(proba, target_counts, reg=SINKHORN_REG, n_iters=SINKHORN_ITERS):
    """Reassign hard labels to `proba` rows (n_candidates x n_classes) so the
    resulting argmax counts approximately match `target_counts` (length
    n_classes, summing to n_candidates), via entropic-regularized optimal
    transport (Sinkhorn-Knopp). Returns hard label array (len n_candidates).
    """
    n = proba.shape[0]
    cost = -np.log(np.clip(proba, 1e-9, 1.0))
    K = np.exp(-cost / reg)
    row_target = np.ones(n)  # each candidate carries unit mass
    col_target = target_counts.astype(float)
    col_target = col_target * (n / col_target.sum())  # renormalize to n candidates
    u = np.ones(n)
    v = np.ones(proba.shape[1])
    for _ in range(n_iters):
        Ku = K * u[:, None]
        v = col_target / np.clip(Ku.sum(axis=0), 1e-12, None)
        Kv = K * v[None, :]
        u = row_target / np.clip(Kv.sum(axis=1), 1e-12, None)
    P = u[:, None] * K * v[None, :]
    return P.argmax(axis=1)


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

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048-063) ---
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

# --- single 80/20 holdout (same split/seed as exp_048-063) ---
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

# target class marginal for Sinkhorn: train's own (pre-augmentation) class proportions
train_class_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
train_class_props = train_class_counts / train_class_counts.sum()


def report(name, pred):
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print(f"\nbaseline (reference, exp_048 same split/model): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)

# stage-1 model: tr+aug only, used to pseudo-label the REAL unlabeled test set
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(X_tr_aug, y_tr_aug)
test_proba = clf_stage1.predict_proba(X_test_real)
test_conf = test_proba.max(axis=1)
test_argmax_pred = test_proba.argmax(axis=1)
print(f"\nstage-1 model fit ({time.time()-t0:.0f}s).", flush=True)

print("\n=== exp_065: XGBoost+aug+pseudo-label(real test), argmax vs Sinkhorn-reassigned labels ===", flush=True)
for thresh in CONF_THRESHOLDS:
    pl_mask = test_conf >= thresh
    n_pl = int(pl_mask.sum())
    if n_pl == 0:
        print(f"\nthreshold {thresh}: 0 pseudo-labeled rows, skipping", flush=True)
        continue
    X_pl = X_test_real[pl_mask]
    proba_pl = test_proba[pl_mask]

    # --- variant A: argmax (reproduces exp_052/058's known mechanism) ---
    y_pl_argmax = test_argmax_pred[pl_mask]
    X_tr_pl = np.vstack([X_tr_aug, X_pl])
    y_tr_pl = np.concatenate([y_tr_aug, y_pl_argmax])
    clf_a = make_xgb(n_classes)
    clf_a.fit(X_tr_pl, y_tr_pl)
    pred_a = clf_a.predict(X_va)
    print(f"\nthreshold {thresh}: {n_pl}/{len(test_conf)} real test rows pseudo-labeled ({time.time()-t0:.0f}s)", flush=True)
    plain_a, weighted_a, topq_a = report(f"argmax_thresh_{thresh}", pred_a)
    print(f"  delta topq (vs baseline): {topq_a - BASELINE_TOPQ:+.4f}  <- reproduction check", flush=True)

    # --- variant B: Sinkhorn-reassigned labels, target = train class marginal ---
    target_counts = train_class_props * n_pl
    y_pl_sinkhorn = sinkhorn_assign(proba_pl, target_counts)
    agree = (y_pl_sinkhorn == y_pl_argmax).mean()
    print(f"  Sinkhorn vs argmax label agreement on candidate pool: {agree:.3f}", flush=True)
    y_tr_pl_sk = np.concatenate([y_tr_aug, y_pl_sinkhorn])
    clf_b = make_xgb(n_classes)
    clf_b.fit(X_tr_pl, y_tr_pl_sk)
    pred_b = clf_b.predict(X_va)
    plain_b, weighted_b, topq_b = report(f"sinkhorn_thresh_{thresh}", pred_b)
    print(f"  delta topq (vs baseline):        {topq_b - BASELINE_TOPQ:+.4f}", flush=True)
    print(f"  delta topq (vs argmax variant):  {topq_b - topq_a:+.4f}  <- CSA's own contribution", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
