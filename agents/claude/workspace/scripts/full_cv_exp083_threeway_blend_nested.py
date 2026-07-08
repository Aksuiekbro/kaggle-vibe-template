"""exp_083 full 3-fold CV, WITH the nested weight-selection fix from
blend-topq-lb-mismatch.md / exp_084 built in from the start (not bolted on
after the fact like exp_081 was).

Probe (probe_exp083_threeway_blend.py, single 80/20 holdout, n_topq=117)
found the 3-way blend (drop the PL arm entirely: wA=0, wB=0.5 ExtraTrees+aug,
wC=0.5 XGB grid+scattering) beats exp_081's confirmed 2-way blend by +0.0085
topq -- but that number came from sweeping WEIGHT_COMBOS on the same holdout
it's scored on, the exact leakage exp_084 showed inflates deltas ~2x on this
competition's blend-weight searches.

Fix here: leave-one-fold-out nested selection. For each of the 3 outer CV
folds, the best weight combo is chosen by maximizing topq on the OTHER two
folds' OOF predictions only, then that (possibly per-fold-different) weight
is applied to and scored on the held-out fold. The reported delta never lets
the weight search see the data it's evaluated on. exp_084 also showed nested
deltas under ~0.01-0.02 topq should be treated as noise in this shift
regime -- apply that threshold when deciding whether this is submission-
worthy, do not submit on a bare positive sign alone.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
N_SPLITS = 3
PL_THRESH = 0.50
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)
# (w_xgb_grid_pl, w_et_grid, w_xgb_grid_scattering) -- must sum to 1
WEIGHT_COMBOS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.3, 0.7, 0.0), (0.7, 0.0, 0.3), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
    (0.3, 0.4, 0.3), (0.4, 0.3, 0.3), (0.5, 0.2, 0.3), (0.2, 0.5, 0.3),
    (0.4, 0.4, 0.2), (0.3, 0.3, 0.4),
]


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


def make_et():
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=1, random_state=42,
    )


t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X_grid = train[grid_feat_cols].values
X_gs = train[grid_feat_cols + scat_feat_cols].values
X_test_grid = test[grid_feat_cols].values
X_test_gs = test[grid_feat_cols + scat_feat_cols].values
print(f"merged rows: train={len(train)} test={len(test)} ({time.time()-t0:.0f}s)", flush=True)

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())

# adversarial P(is_test) on grid-only feature space, for the topq gating metric
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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rng_master = np.random.default_rng(SEED)
proba_a = np.zeros((len(y), n_classes))  # XGB grid+aug+PL
proba_b = np.zeros((len(y), n_classes))  # ExtraTrees grid+aug
proba_c = np.zeros((len(y), n_classes))  # XGB grid+scattering
fold_idx = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_grid, y)):
    t_fold = time.time()
    fold_idx[va_idx] = fold

    tr_rows = train.iloc[tr_idx]
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    aug_df = augment_rows(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
    X_tr_grid_aug = np.vstack([X_grid[tr_idx], aug_df[grid_feat_cols].values])
    y_tr_aug = np.concatenate([y[tr_idx], le.transform(aug_df["Pitch_ID"].values)])

    # arm A: XGBoost + grid-aug + pseudo-label(thresh=0.50) [banked pipeline]
    clf_stage1 = make_xgb(n_classes)
    clf_stage1.fit(X_tr_grid_aug, y_tr_aug)
    test_proba_stage1 = clf_stage1.predict_proba(X_test_grid)
    test_conf = test_proba_stage1.max(axis=1)
    test_pred = test_proba_stage1.argmax(axis=1)
    pl_mask = test_conf >= PL_THRESH
    X_pl, y_pl = X_test_grid[pl_mask], test_pred[pl_mask]
    X_tr_pl = np.vstack([X_tr_grid_aug, X_pl])
    y_tr_pl = np.concatenate([y_tr_aug, y_pl])
    clf_xgb_pl = make_xgb(n_classes)
    clf_xgb_pl.fit(X_tr_pl, y_tr_pl)
    proba_a[va_idx] = clf_xgb_pl.predict_proba(X_grid[va_idx])

    # arm B: ExtraTrees + grid-aug (no PL)
    clf_et = make_et()
    clf_et.fit(X_tr_grid_aug, y_tr_aug)
    proba_b[va_idx] = clf_et.predict_proba(X_grid[va_idx])

    # arm C: XGBoost on grid+scattering, no aug, no PL
    clf_gs = make_xgb(n_classes)
    clf_gs.fit(X_gs[tr_idx], y[tr_idx])
    proba_c[va_idx] = clf_gs.predict_proba(X_gs[va_idx])

    acc_a = accuracy_score(y[va_idx], proba_a[va_idx].argmax(axis=1))
    acc_b = accuracy_score(y[va_idx], proba_b[va_idx].argmax(axis=1))
    acc_c = accuracy_score(y[va_idx], proba_c[va_idx].argmax(axis=1))
    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): a_acc={acc_a:.4f} b_acc={acc_b:.4f} c_acc={acc_c:.4f} pl'd={pl_mask.sum()}/{len(test_conf)}", flush=True)


def topq_on(mask, proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    sel = mask & top_mask
    return (correct[sel].mean() if sel.sum() else float("nan")), sel.sum()


def topq_all(proba):
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    return correct[top_mask].mean()


print("\n=== exp_083 full CV (same-fold, for direct comparability to exp_081's original methodology) ===", flush=True)
topq_a_all = topq_all(proba_a)
topq_ab_ref = topq_all(0.3 * proba_a + 0.7 * proba_b)  # exp_081's confirmed-best pairwise weight
best_combo_samefold, best_topq_samefold = None, -1.0
for wa, wb, wc in WEIGHT_COMBOS:
    t = topq_all(wa * proba_a + wb * proba_b + wc * proba_c)
    if t > best_topq_samefold:
        best_combo_samefold, best_topq_samefold = (wa, wb, wc), t
print(f"[A] XGB+aug+PL alone: topq={topq_a_all:.4f}", flush=True)
print(f"[A+B] exp_081 pairwise ref (w=0.3/0.7): topq={topq_ab_ref:.4f}", flush=True)
print(f"[same-fold] best combo={best_combo_samefold} topq={best_topq_samefold:.4f} "
      f"delta vs A+B ref={best_topq_samefold - topq_ab_ref:+.4f}  <- what exp_081's methodology would report", flush=True)

print("\n=== exp_083 full CV (NESTED / leave-one-fold-out weight selection, the corrected methodology) ===", flush=True)
held_correct_blend, held_correct_ab_ref = [], []
chosen = []
for f in range(N_SPLITS):
    other_mask = fold_idx != f
    held_mask = fold_idx == f

    best_combo, best_t = None, -1.0
    for wa, wb, wc in WEIGHT_COMBOS:
        t, n = topq_on(other_mask, wa * proba_a + wb * proba_b + wc * proba_c)
        if t > best_t:
            best_combo, best_t = (wa, wb, wc), t
    chosen.append(best_combo)

    wa, wb, wc = best_combo
    proba_blend_held = wa * proba_a + wb * proba_b + wc * proba_c
    proba_ab_ref_held = 0.3 * proba_a + 0.7 * proba_b
    sel = held_mask & top_mask
    pred_blend = proba_blend_held.argmax(axis=1)
    pred_ab = proba_ab_ref_held.argmax(axis=1)
    held_correct_blend.append((pred_blend[sel] == y[sel]).astype(float))
    held_correct_ab_ref.append((pred_ab[sel] == y[sel]).astype(float))
    print(f"fold {f}: weight chosen on OTHER folds = {best_combo} (their topq={best_t:.4f}), "
          f"held-out n={sel.sum()}", flush=True)

nested_blend = np.concatenate(held_correct_blend).mean()
nested_ab_ref = np.concatenate(held_correct_ab_ref).mean()
nested_delta = nested_blend - nested_ab_ref
print(f"\nweights chosen per fold: {chosen}", flush=True)
print(f"nested topq(3-way blend)={nested_blend:.4f}  nested topq(A+B ref)={nested_ab_ref:.4f}", flush=True)
print(f"NESTED delta vs A+B ref: {nested_delta:+.4f}  <- honest, submission-worthiness gate", flush=True)
print(f"same-fold delta was: {best_topq_samefold - topq_ab_ref:+.4f} (inflated comparison point)", flush=True)
if nested_delta > 0.02:
    print("VERDICT: nested delta clears the ~0.01-0.02 noise floor (exp_084) -- candidate for pre-submit check.", flush=True)
else:
    print("VERDICT: nested delta does NOT clear the ~0.01-0.02 noise floor (exp_084) -- treat as noise, do not submit on this alone.", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)

np.save(GRID_DIR + "exp083_proba_a.npy", proba_a)
np.save(GRID_DIR + "exp083_proba_b.npy", proba_b)
np.save(GRID_DIR + "exp083_proba_c.npy", proba_c)
np.save(GRID_DIR + "exp083_labels.npy", y)
np.save(GRID_DIR + "exp083_fold_idx.npy", fold_idx)
