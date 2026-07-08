"""Retrospective check (own, zero retraining cost): does leave-one-fold-out
(nested) blend-weight selection on exp_081's ALREADY-SAVED OOF arrays produce
a smaller/negative delta than the same-fold weight sweep did (topq +0.0034,
which reversed to -0.0172 on the real LB per blend-topq-lb-mismatch.md)?

Methodology fix per the failure card: choose the blend weight using topq
computed only on folds OTHER than the one being scored, then apply that
(possibly per-fold-different) weight to the held-out fold. This never lets
the weight-selection step see the data it's later scored on -- the exact gap
that let exp_081/sub_029 pass the coordinator's shift-aware gate and still
reverse sign on LB.

Fold assignment is not saved by full_cv_exp081_extratrees_blend.py, but it is
fully deterministic (StratifiedKFold(n_splits=3, shuffle=True,
random_state=SEED) on the same X, y loaded the same way) -- recomputed here
identically rather than rerun/retrained.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import SEED

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
N_SPLITS = 3
BLEND_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7)

train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
le = LabelEncoder()
y_recomputed = le.fit_transform(train["Pitch_ID"].values)
X = train[feat_cols].values
X_test_real = test[feat_cols].values

y = np.load(DIR + "exp081_labels.npy")
assert np.array_equal(y, y_recomputed), "label recomputation mismatch -- cannot trust recovered fold indices"
proba_xgb = np.load(DIR + "exp081_proba_xgb.npy")
proba_et = np.load(DIR + "exp081_proba_et.npy")

# adversarial weight for topq, recomputed identically to full_cv_exp081 (same seed/model/folds)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

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
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
print(f"adversarial AUC (recomputed): {adv_auc:.4f}", flush=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_idx = np.zeros(len(y), dtype=int)
for fold, (_, va_idx) in enumerate(skf.split(X, y)):
    fold_idx[va_idx] = fold
print(f"fold sizes: {[np.sum(fold_idx == f) for f in range(N_SPLITS)]}", flush=True)


def topq_on(mask, proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    sel = mask & top_mask
    return correct[sel].mean() if sel.sum() else float("nan"), sel.sum()


# --- (repro) same-fold weight sweep, as originally run ---
def topq_all(proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    return correct[top_mask].mean()


topq_xgb_all = topq_all(proba_xgb)
best_w_samefold, best_topq_samefold = None, -1.0
for wgt in BLEND_WEIGHTS:
    t = topq_all(wgt * proba_xgb + (1 - wgt) * proba_et)
    if t > best_topq_samefold:
        best_w_samefold, best_topq_samefold = wgt, t
print(f"\n[same-fold, original methodology] XGB alone topq={topq_xgb_all:.4f}; "
      f"best blend w_xgb={best_w_samefold} topq={best_topq_samefold:.4f} "
      f"delta={best_topq_samefold - topq_xgb_all:+.4f}  <- what sub_029 was submitted on (LB actually: -0.0172)", flush=True)

# --- nested (leave-one-fold-out) weight selection ---
held_out_correct = []
held_out_correct_xgb = []
chosen_weights = []
for f in range(N_SPLITS):
    other_mask = fold_idx != f
    held_mask = fold_idx == f

    best_w, best_t = None, -1.0
    for wgt in BLEND_WEIGHTS:
        t, n = topq_on(other_mask, wgt * proba_xgb + (1 - wgt) * proba_et)
        if t > best_t:
            best_w, best_t = wgt, t
    chosen_weights.append(best_w)

    proba_blend_held = best_w * proba_xgb + (1 - best_w) * proba_et
    sel = held_mask & top_mask
    pred_blend = proba_blend_held.argmax(axis=1)
    pred_xgb = proba_xgb.argmax(axis=1)
    correct_blend = (pred_blend[sel] == y[sel]).astype(float)
    correct_xgb = (pred_xgb[sel] == y[sel]).astype(float)
    held_out_correct.append(correct_blend)
    held_out_correct_xgb.append(correct_xgb)
    print(f"fold {f}: weight selected on OTHER folds = {best_w} (their topq={best_t:.4f}), "
          f"held-out fold topq(blend)={correct_blend.mean() if len(correct_blend) else float('nan'):.4f} "
          f"(n={len(correct_blend)}), held-out topq(xgb alone)={correct_xgb.mean() if len(correct_xgb) else float('nan'):.4f}", flush=True)

nested_blend = np.concatenate(held_out_correct).mean()
nested_xgb = np.concatenate(held_out_correct_xgb).mean()
print(f"\n[nested / leave-one-fold-out] weights chosen per fold: {chosen_weights}", flush=True)
print(f"nested topq(blend)={nested_blend:.4f}  nested topq(xgb alone)={nested_xgb:.4f}  "
      f"nested delta={nested_blend - nested_xgb:+.4f}  <- honest estimate, weight never saw its own scoring data", flush=True)
print(f"\nCOMPARISON: same-fold delta was {best_topq_samefold - topq_xgb_all:+.4f} (what got submitted, LB was -0.0172); "
      f"nested delta is {nested_blend - nested_xgb:+.4f}", flush=True)
