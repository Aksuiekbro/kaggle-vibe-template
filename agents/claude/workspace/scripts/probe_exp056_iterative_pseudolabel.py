"""exp_056 (own, new): ITERATIVE (multi-round) self-training, vs exp_051/052's
single-round fold-honest pseudo-labeling and exp_055's density-regularized
single-round acceptance.

Genuinely new mechanism on the pseudo-labeling axis: every prior pseudo-label
experiment this competition (exp_051/052 confidence threshold, exp_055
density-regularized score) picks pseudo-labels ONCE from a single stage-1
model and stops. This tests whether a second (and third) round -- re-scoring
the real unlabeled test set with the model that already includes round-1's
pseudo-labels, and adding newly-confident rows on top -- compounds the small
existing gain (exp_051 full CV: topq +0.0017) or instead compounds noise
(a known self-training failure mode: confident-but-wrong labels reinforcing
themselves round over round).

Same 80/20 holdout split/seed/features/augmentation config as exp_048-055 for
direct comparability. Uses threshold=0.85 (exp_051/052's full-CV-confirmed
winner) as the per-round acceptance rule -- isolates "does iterating help"
from "which acceptance rule is best" (exp_055 already covers the latter).
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

BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

THRESH = 0.85
N_ROUNDS = 3


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
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

idx = np.arange(len(y))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va = X[tr_idx], X[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
w_full = p_test_all[: len(X)]
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


print(f"\nbaseline (no pseudo-label, reference): plain={BASELINE_PLAIN:.4f} "
      f"weighted={BASELINE_WEIGHTED:.4f} topq={BASELINE_TOPQ:.4f}", flush=True)
print("exp_052 thresh=0.85 (single-round, reference): plain=0.9485 weighted=0.8383 "
      "topq=0.9658  <- comparison target (round_2 below should reproduce this: round_1 is "
      "the no-pseudo-label sanity check, round_2 is the first PL round, round_3+ is new)", flush=True)

print(f"\n=== exp_056: {N_ROUNDS}-round iterative self-training, thresh={THRESH} per round "
      f"(round_1=no-PL baseline check, round_2=single-round PL, round_3+=iterative/new) ===", flush=True)

X_cur = X_tr_aug
y_cur = y_tr_aug
labeled_mask = np.zeros(len(X_test_real), dtype=bool)  # test rows already absorbed into training
prev_topq = None

for round_i in range(1, N_ROUNDS + 1):
    clf = make_xgb(n_classes)
    clf.fit(X_cur, y_cur)

    # score this round's model on the real held-out labels (never touched by pseudo-labeling)
    pred_va = clf.predict(X_va)
    plain, weighted, topq = report(f"round_{round_i}", pred_va)
    d_topq = topq - BASELINE_TOPQ
    print(f"round_{round_i} delta topq vs no-pl: {d_topq:+.4f}  <- coordinator's gating metric "
          f"({time.time()-t0:.0f}s elapsed)", flush=True)
    if prev_topq is not None:
        print(f"round_{round_i} delta topq vs round_{round_i-1}: {topq - prev_topq:+.4f}", flush=True)
    prev_topq = topq

    if round_i == N_ROUNDS:
        break

    # re-score the REAL unlabeled test set with this round's model, add newly-confident
    # rows (not already absorbed) on top of the existing pseudo-labeled set
    test_proba = clf.predict_proba(X_test_real)
    test_conf = test_proba.max(axis=1)
    test_pred = test_proba.argmax(axis=1)
    new_mask = (test_conf >= THRESH) & (~labeled_mask)
    n_new = new_mask.sum()
    print(f"round_{round_i}->  {n_new} newly-confident test rows added "
          f"(cumulative will be {labeled_mask.sum() + n_new}/{len(X_test_real)})", flush=True)
    if n_new == 0:
        print("no new confident rows -- stopping early, further rounds would be identical", flush=True)
        break
    labeled_mask |= new_mask
    X_cur = np.vstack([X_tr_aug, X_test_real[labeled_mask]])
    y_cur = np.concatenate([y_tr_aug, test_pred[labeled_mask]])

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
