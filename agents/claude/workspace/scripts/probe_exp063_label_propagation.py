"""exp_063 (scheduler id exp_062, own, new): graph-based label propagation
(sklearn LabelSpreading) as a genuinely new pseudo-labeling mechanism.

Every pseudo-labeling/prior-correction mechanism tried so far this
competition (exp_051-062) routes through a classifier trained ONLY on
train data, then asks "how much do I trust this classifier's own output on
test row X" (confidence threshold, cross-model agreement, or EM-reweighted
prior). Label propagation instead builds a kNN-affinity graph over the JOINT
train+test feature manifold and diffuses train labels by local similarity --
no train-only decision boundary is involved in the propagation step at all.
This sidesteps the specific failure modes already closed here: EM's noisy
per-class prior on thin classes (exp_059/060, topq -0.1624) and
single/dual-classifier confidence miscalibration under shift (exp_051-060's
whole threshold/agreement axis).

Two checks, in order:
1. Direct sanity check: can the graph alone (LabelSpreading, X_va masked as
   unlabeled) predict the held-out validation labels at all, and does it do
   worse specifically on rare classes (the a-priori risk: 14/82 classes have
   <10 train samples, so a kNN graph may lack same-class anchors nearby)?
   This is a standalone-model comparison against the stage-1 XGBoost's own
   accuracy on the identical split -- not the final gating metric.
2. Pipeline check (comparable to exp_051-062): use the graph's confident
   test-row propagated labels as pseudo-labels, add to train, retrain
   XGBoost (same architecture as every prior pseudo-labeling experiment),
   evaluate on X_va (never touched by propagation or pseudo-labeling).
   Scored on the SAME plain/weighted/topq metrics against the
   exp_048-reference baseline for direct comparability.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import RARE_THRESHOLD, SEED
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)

# baseline reference: identical split/seed/features/model as exp_048/049/050/057-062
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

LS_CONFIGS = [
    dict(n_neighbors=5, alpha=0.2),
    dict(n_neighbors=10, alpha=0.2),
    dict(n_neighbors=10, alpha=0.5),
]
CONF_THRESHOLDS = (0.5, 0.7)


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


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
rare_classes_ids = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes_ids)}/{len(full_counts)}", flush=True)

# --- adversarial P(is_test), reused only for topq weighting (same as exp_048-062) ---
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

# --- single 80/20 holdout (same split/seed as exp_048-062) ---
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

clf_stage1 = make_xgb(n_classes)
clf_stage1.fit(X_tr_aug, y_tr_aug)
pred_va_xgb = clf_stage1.predict(X_va)
print(f"\nstage-1 XGBoost fit ({time.time()-t0:.0f}s)", flush=True)
report("stage-1 XGBoost (this holdout's own fit)", pred_va_xgb)

scaler = StandardScaler()
scaler.fit(np.vstack([X_tr_aug, X_va, X_test_real]))
X_tr_aug_s = scaler.transform(X_tr_aug)
X_va_s = scaler.transform(X_va)
X_test_real_s = scaler.transform(X_test_real)

print("\n=== exp_063: graph-based label propagation (sklearn LabelSpreading) ===", flush=True)

print("\n--- check 1: direct graph-only accuracy on held-out X_va (never touched, masked unlabeled) ---", flush=True)
for cfg in LS_CONFIGS:
    X_graph = np.vstack([X_tr_aug_s, X_va_s])
    y_graph = np.concatenate([y_tr_aug, -1 * np.ones(len(X_va_s), dtype=int)])
    ls = LabelSpreading(kernel="knn", n_neighbors=cfg["n_neighbors"], alpha=cfg["alpha"], max_iter=30)
    ls.fit(X_graph, y_graph)
    pred_va_graph = ls.transduction_[len(X_tr_aug_s):]
    print(f"\nn_neighbors={cfg['n_neighbors']} alpha={cfg['alpha']} ({time.time()-t0:.0f}s)", flush=True)
    report("graph-only", pred_va_graph)

print("\n--- check 2: pseudo-label pipeline (graph propagates labels onto real test, "
      "XGBoost retrains on train+aug+pseudo, evaluated on X_va) ---", flush=True)
best_cfg = LS_CONFIGS[1]  # n_neighbors=10, alpha=0.2 -- middle-ground default
X_graph_test = np.vstack([X_tr_aug_s, X_test_real_s])
y_graph_test = np.concatenate([y_tr_aug, -1 * np.ones(len(X_test_real_s), dtype=int)])
ls_test = LabelSpreading(kernel="knn", n_neighbors=best_cfg["n_neighbors"], alpha=best_cfg["alpha"], max_iter=30)
ls_test.fit(X_graph_test, y_graph_test)
test_dist = ls_test.label_distributions_[len(X_tr_aug_s):]
test_dist = test_dist / np.clip(test_dist.sum(axis=1, keepdims=True), 1e-12, None)
test_conf = test_dist.max(axis=1)
test_pred_graph = ls_test.transduction_[len(X_tr_aug_s):]
print(f"\nlabel-propagation graph fit on train+real-test ({time.time()-t0:.0f}s). "
      f"test confidence p50={np.percentile(test_conf,50):.3f} p90={np.percentile(test_conf,90):.3f}", flush=True)

for thresh in CONF_THRESHOLDS:
    pl_mask = test_conf >= thresh
    n_pl = pl_mask.sum()
    if n_pl == 0:
        print(f"\nthreshold {thresh}: 0 pseudo-labeled rows, skipping", flush=True)
        continue
    X_pl = X_test_real[pl_mask]
    y_pl = test_pred_graph[pl_mask]
    X_tr_pl = np.vstack([X_tr_aug, X_pl])
    y_tr_pl = np.concatenate([y_tr_aug, y_pl])
    clf = make_xgb(n_classes)
    clf.fit(X_tr_pl, y_tr_pl)
    pred = clf.predict(X_va)
    print(f"\nthreshold {thresh}: {n_pl}/{len(test_conf)} real test rows pseudo-labeled via graph, "
          f"fit done ({time.time()-t0:.0f}s)", flush=True)
    plain, weighted, topq = report(f"graph_pseudolabel_thresh_{thresh}", pred)
    print(f"delta plain    vs no-pl:      {plain - BASELINE_PLAIN:+.4f}", flush=True)
    print(f"delta weighted vs no-pl:      {weighted - BASELINE_WEIGHTED:+.4f}", flush=True)
    print(f"delta topq     vs no-pl:      {topq - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
