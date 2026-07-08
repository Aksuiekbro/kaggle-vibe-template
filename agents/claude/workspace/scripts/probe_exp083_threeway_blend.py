"""exp_083 (own, probe): does adding grid+scattering-XGBoost (confirmed new
feature family, exp_073/074, +0.0223 topq standalone vs grid-only, banked as
sub_028's diversification hedge) as a THIRD blend arm on top of exp_081's
confirmed 2-way blend (XGBoost+aug+PL(0.50) + ExtraTrees+aug, both grid-only,
full CV +0.0034) add further ensemble diversity?

This is the one composition mode not yet ruled out: exp_075 showed
pseudo-labeling composes NEGATIVELY with grid+scattering (-0.0085, likely
because scattering's near-perfect adv-AUC 0.9975 reinforces shift artifacts
when used to relabel training data), and exp_076 showed augmentation is
FLAT with it. But those are training-pipeline compositions (retraining on
altered data). Blending finished PREDICTIONS from an independently-trained
grid+scattering model is a different mechanism -- it never touches
scattering's shift-sensitivity during training, only combines its output
probabilities, which is exactly how exp_081's ExtraTrees blend added value
despite ExtraTrees also not composing with training-side levers.

exp_082 showed ExtraTrees itself does NOT transfer to grid+scattering
features (-0.0103), so ExtraTrees stays on grid-only in this 3-way blend;
only the grid+scattering XGBoost model is added as a new arm.

Full-data single 80/20 stratified holdout (not a row-subsample), matching
exp_081/012/040's fidelity lesson (probes on tiny subsamples flip sign at
full CV; this uses the complete dataset, just a single split rather than
3-fold, as the cheap first gate). Fold-safe: augmentation and pseudo-labeling
generated only from the train split.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
PL_THRESH = 0.50
MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

# (w_xgb_grid_pl, w_et_grid, w_xgb_grid_scattering) -- must sum to 1
WEIGHT_COMBOS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.3, 0.7, 0.0),   # exp_081's confirmed-best pairwise A+B
    (0.7, 0.0, 0.3),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.3, 0.4, 0.3),
    (0.4, 0.3, 0.3),
    (0.5, 0.2, 0.3),
    (0.2, 0.5, 0.3),
    (0.4, 0.4, 0.2),
    (0.3, 0.3, 0.4),
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

# same row-filter across all 3 arms so the holdout split is apples-to-apples
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

tr_idx, va_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
)
y_va = y[va_idx]

# adversarial P(is_test) on grid-only feature space (matches exp_081's gating metric)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold as SKF_ADV
from sklearn.metrics import accuracy_score, roc_auc_score

X_adv = np.vstack([X_grid, X_test_grid])
y_adv = np.concatenate([np.zeros(len(X_grid)), np.ones(len(X_test_grid))])
skf_adv = SKF_ADV(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tr_a, va_a in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_a], y_adv[tr_a])
    p_test_all[va_a] = clf.predict_proba(X_adv[va_a])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
w_all = p_test_all[: len(X_grid)]
w_va = w_all[va_idx]
q75 = np.quantile(w_va, 0.75)
top_mask_va = w_va >= q75
print(f"adversarial AUC (grid features): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

tr_rows = train.iloc[tr_idx]
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
rng = np.random.default_rng(SEED)
aug_df = augment_rows(rare_tr_rows, rng, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
X_tr_grid_aug = np.vstack([X_grid[tr_idx], aug_df[grid_feat_cols].values])
y_tr_aug = np.concatenate([y[tr_idx], le.transform(aug_df["Pitch_ID"].values)])
print(f"train {len(tr_idx)} rows -> {len(X_tr_grid_aug)} after rare-class augmentation ({time.time()-t0:.0f}s)", flush=True)

# --- arm A: XGBoost + grid-aug + pseudo-label(thresh=0.50) [banked pipeline] ---
t1 = time.time()
clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(X_tr_grid_aug, y_tr_aug)
test_proba_stage1 = clf_stage1.predict_proba(X_test_grid)
test_conf = test_proba_stage1.max(axis=1)
test_pred = test_proba_stage1.argmax(axis=1)
pl_mask = test_conf >= PL_THRESH
X_pl = X_test_grid[pl_mask]
y_pl = test_pred[pl_mask]
X_tr_pl = np.vstack([X_tr_grid_aug, X_pl])
y_tr_pl = np.concatenate([y_tr_aug, y_pl])
clf_xgb_pl = make_xgb(n_classes)
clf_xgb_pl.fit(X_tr_pl, y_tr_pl)
proba_a = clf_xgb_pl.predict_proba(X_grid[va_idx])
print(f"[A] XGBoost+grid-aug+PL(thr={PL_THRESH}) fit+predict done, {pl_mask.sum()}/{len(test_conf)} pseudo-labeled ({time.time()-t1:.0f}s)", flush=True)

# --- arm B: ExtraTrees + grid-aug (no PL) ---
t2 = time.time()
clf_et = make_et()
clf_et.fit(X_tr_grid_aug, y_tr_aug)
proba_b = clf_et.predict_proba(X_grid[va_idx])
print(f"[B] ExtraTrees+grid-aug fit+predict done ({time.time()-t2:.0f}s)", flush=True)

# --- arm C: XGBoost on grid+scattering, no aug, no PL ---
t3 = time.time()
clf_gs = make_xgb(n_classes)
clf_gs.fit(X_gs[tr_idx], y[tr_idx])
proba_c = clf_gs.predict_proba(X_gs[va_idx])
print(f"[C] XGBoost grid+scattering (no aug/PL) fit+predict done ({time.time()-t3:.0f}s)", flush=True)


def score(proba, name):
    pred = proba.argmax(axis=1)
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    weighted = np.average(correct, weights=w_va)
    topq = correct[top_mask_va].mean() if top_mask_va.sum() else float("nan")
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask_va.sum()})", flush=True)
    return topq


print("\n=== exp_083 probe: 3-way blend (XGB+grid-aug+PL / ExtraTrees+grid-aug / XGB grid+scattering) ===", flush=True)
topq_a = score(proba_a, "[A] XGBoost+grid-aug+PL(0.50) alone (banked pipeline)")
topq_b = score(proba_b, "[B] ExtraTrees+grid-aug alone")
topq_c = score(proba_c, "[C] XGBoost grid+scattering alone")
ab_ref = score(0.3 * proba_a + 0.7 * proba_b, "[A+B ref] exp_081 confirmed-best pairwise blend (w_xgb=0.3)")

best_combo, best_topq = None, -1.0
for wa, wb, wc in WEIGHT_COMBOS:
    proba_blend = wa * proba_a + wb * proba_b + wc * proba_c
    topq = score(proba_blend, f"blend(wA={wa},wB={wb},wC={wc})")
    if topq > best_topq:
        best_combo, best_topq = (wa, wb, wc), topq

print(f"\nbest combo: {best_combo}, topq={best_topq:.4f}", flush=True)
print(f"delta vs [A] alone: {best_topq - topq_a:+.4f}", flush=True)
print(f"delta vs [A+B] exp_081 pairwise reference: {best_topq - ab_ref:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
