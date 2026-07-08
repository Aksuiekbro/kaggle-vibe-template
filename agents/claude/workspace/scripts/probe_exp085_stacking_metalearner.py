"""exp_085 (own, prepped ahead of exp_083 landing): trained stacking
meta-learner over the same 3 arms exp_083 blended by a swept scalar weight
(XGB grid+aug+PL, ExtraTrees grid+aug, XGB grid+scattering).

Motivation: PLAN.md Coordinator note #3 / STRATEGY.md item 0b flag that
prediction-level blending on this competition has failed 3/3 by simple
proba-weight sweep (exp_013, exp_044, exp_077) and that sub_029's weight-swept
blend reversed sign on the real LB (-0.0172) despite a full-CV shift-aware
gate -- and both call out a TRAINED meta-learner/stacking as a materially
different, not-yet-tried mechanism. A weight sweep is a 1-3 scalar
hyperparameter search that can overfit fold structure even when nested
(exp_084 showed nesting only halves the inflation, doesn't remove it). A
meta-learner fit by proper cross-validation is the standard fix for stacking
leakage specifically -- it learns a per-class/per-arm combination function
instead of one global scalar, using its own inner CV to avoid seeing the
outer-fold labels it's scored on.

Reuses exp_083's saved OOF arrays directly (proba_a/b/c, labels, fold_idx) --
zero retraining of the 3 base arms. Only adds: (1) recomputing top_mask via
the identical adversarial-validation procedure exp_083 used (not saved by
that script), (2) fitting a multinomial logistic-regression meta-learner on
concat([proba_a, proba_b, proba_c]) via leave-one-outer-fold-out (meta-learner
trained on 2 outer folds' OOF probs, scored on the 3rd -- same nesting
discipline as exp_083's weight search, but now the "hyperparameter" is a full
per-class linear model instead of one scalar).

Cheap: no XGBoost/ExtraTrees fits, just array loads + a handful of small
LogisticRegression fits (n<=2330, ~90 features in). Run only after exp_083's
.npy files exist.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
SEED = 42

t0 = time.time()
proba_a = np.load(GRID_DIR + "exp083_proba_a.npy")  # XGB grid+aug+PL
proba_b = np.load(GRID_DIR + "exp083_proba_b.npy")  # ExtraTrees grid+aug
proba_c = np.load(GRID_DIR + "exp083_proba_c.npy")  # XGB grid+scattering
y = np.load(GRID_DIR + "exp083_labels.npy")
fold_idx = np.load(GRID_DIR + "exp083_fold_idx.npy")
n_classes = proba_a.shape[1]
print(f"loaded exp_083 OOF arrays: n={len(y)} n_classes={n_classes} ({time.time()-t0:.0f}s)", flush=True)

# recompute top_mask exactly as exp_083 did (adversarial P(is_test) on grid features,
# top quartile by test-likeness) -- not saved by that script, cheap to redo (~85s)
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")
scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")
assert len(train) == len(y), "row count mismatch vs exp_083 arrays -- feature files changed underneath us"
X_grid = train[grid_feat_cols].values
X_test_grid = test[grid_feat_cols].values

X_adv = np.vstack([X_grid, X_test_grid])
y_adv = np.concatenate([np.zeros(len(X_grid)), np.ones(len(X_test_grid))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tr_a, va_a in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_a], y_adv[tr_a])
    p_test_all[va_a] = clf.predict_proba(X_adv[va_a])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
w = p_test_all[: len(X_grid)]
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
print(f"adversarial AUC (grid features): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)


def topq_on(mask, proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    sel = mask & top_mask
    return (correct[sel].mean() if sel.sum() else float("nan")), sel.sum()


N_SPLITS = int(fold_idx.max()) + 1
X_stack = np.concatenate([proba_a, proba_b, proba_c], axis=1)  # n x (3*n_classes)

# reference: exp_081's confirmed-best fixed pairwise weight (0.3 XGB+PL, 0.7 ET), same-fold scoring point of comparison
topq_ab_ref_all, _ = topq_on(np.ones(len(y), dtype=bool), 0.3 * proba_a + 0.7 * proba_b)
print(f"[A+B] exp_081 pairwise ref (w=0.3/0.7), all-rows topq={topq_ab_ref_all:.4f}", flush=True)

print("\n=== exp_085: nested (leave-one-outer-fold-out) trained meta-learner ===", flush=True)
held_correct_meta, held_correct_ab_ref = [], []
for f in range(N_SPLITS):
    other_mask = fold_idx != f
    held_mask = fold_idx == f

    meta = LogisticRegression(max_iter=2000, C=1.0)
    meta.fit(X_stack[other_mask], y[other_mask])
    proba_meta_held = meta.predict_proba(X_stack[held_mask])
    # LogisticRegression sorts classes_ ascending; y is already 0..n_classes-1 from exp_083's LabelEncoder, so this aligns
    assert list(meta.classes_) == list(range(n_classes)) or len(meta.classes_) < n_classes, "class subset in a fold -- check alignment"

    proba_ab_ref_held = 0.3 * proba_a[held_mask] + 0.7 * proba_b[held_mask]
    sel = held_mask & top_mask
    sel_local = sel[held_mask]

    pred_meta = np.full(held_mask.sum(), -1)
    # map meta.classes_ (may be a subset if a class is missing from `other_mask`) back to full label space
    pred_meta_sub = proba_meta_held.argmax(axis=1)
    pred_meta = np.array([meta.classes_[i] for i in pred_meta_sub])
    pred_ab = proba_ab_ref_held.argmax(axis=1)

    y_held = y[held_mask]
    held_correct_meta.append((pred_meta[sel_local] == y_held[sel_local]).astype(float))
    held_correct_ab_ref.append((pred_ab[sel_local] == y_held[sel_local]).astype(float))
    print(f"fold {f}: meta-learner trained on {other_mask.sum()} rows ({len(meta.classes_)}/{n_classes} classes seen), held-out n={sel_local.sum()}", flush=True)

nested_meta = np.concatenate(held_correct_meta).mean()
nested_ab_ref = np.concatenate(held_correct_ab_ref).mean()
nested_delta = nested_meta - nested_ab_ref
print(f"\nnested topq(meta-learner stack)={nested_meta:.4f}  nested topq(A+B ref)={nested_ab_ref:.4f}", flush=True)
print(f"NESTED delta vs A+B ref: {nested_delta:+.4f}", flush=True)
if nested_delta > 0.02:
    print("VERDICT: nested delta clears the ~0.01-0.02 noise floor (exp_084) -- candidate for pre-submit check.", flush=True)
else:
    print("VERDICT: nested delta does NOT clear the noise floor -- trained-meta-learner stacking axis closes same as the weight-sweep axis.", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
