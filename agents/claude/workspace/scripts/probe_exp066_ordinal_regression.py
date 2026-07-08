"""exp_066: treat Pitch_ID as an ordinal regression target instead of a
nominal 82-way classification target.

Genuinely new mechanism family (STRATEGY.md block 37: feature-transform,
model-family, ensembling, hyperparameters, domain-adaptation, and most of
pseudo-labeling are all closed -- this axis has never been tried).

Pitch_ID is not an arbitrary category id: train.csv's Pitch_ID spans exactly
0..81 (82 contiguous integers, confirmed via `df['Pitch_ID'].describe()`), and
exp_005's own grid-feature extractor builds candidates via the MIDI-to-
frequency formula `f0 = 440 * 2**((d-69)/12)` over a contiguous note range --
i.e. the label is a discretized pitch axis, adjacent ids are one semitone
apart. Nominal multiclass softmax (every pair of classes equally "different")
throws this structure away; with severe class imbalance (14/82 classes <10
samples, avg ~28/class) a regression objective could let the model share
information across neighboring pitches (a candidate near the true pitch has
correlated grid-feature evidence) instead of learning 82 independent decision
surfaces from very few examples for the thin classes.

Mechanism hypothesis: XGBoost regression on the integer-encoded Pitch_ID
(LabelEncoder is order-preserving here since all 82 values 0..81 are present
and already sorted) learns splits on a single continuous target, letting
gradient signal from nearby-pitch rows reinforce each other. Cheapest test
first (C4): plain regression + round-and-clip predictions vs the existing
multi:softmax XGBoost classifier, same split/seed/augmentation as exp_048-065
for direct topq comparability. If this shows any signal, a follow-up
(exp_067) would try the classic Frank & Hall ordinal-binary-decomposition
trick (K-1 binary "class > k" classifiers) which doesn't force a linear
metric-distance assumption the way raw regression does.

Risk (flagged before running): squared-error regression assumes a linear
penalty in pitch-index units, which need not match musical/perceptual
distance (e.g. octave wraparound, harmonic confusability between a true pitch
and its octave-neighbor candidate) -- if this hurts, it's evidence the
*labeling* is ordinal but the *error cost* isn't simply |predicted-true|.
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

# baseline reference: identical split/seed/features/model as exp_048-065
BASELINE_PLAIN = 0.9442
BASELINE_WEIGHTED = 0.8223
BASELINE_TOPQ = 0.9573

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
assert list(le.classes_) == list(range(n_classes)), "Pitch_ID not a contiguous 0..N-1 ordinal range -- ordinal framing invalid"
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

print("\n=== exp_066: XGBoost regression on ordinal-encoded Pitch_ID (round+clip) ===", flush=True)
reg = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
    random_state=42, n_jobs=1, tree_method="hist", verbosity=0,
)
reg.fit(X_tr_aug, y_tr_aug.astype(float))
pred_raw = reg.predict(X_va)
pred_va = np.clip(np.round(pred_raw), 0, n_classes - 1).astype(int)
print(f"\nregression fit ({time.time()-t0:.0f}s)", flush=True)
mae = np.mean(np.abs(pred_raw - y_va))
print(f"raw MAE (pitch-index units, pre-round): {mae:.3f}", flush=True)
plain_r, weighted_r, topq_r = report("ordinal_regression", pred_va)
print(f"delta plain    vs exp_048-ref: {plain_r - BASELINE_PLAIN:+.4f}", flush=True)
print(f"delta weighted vs exp_048-ref: {weighted_r - BASELINE_WEIGHTED:+.4f}", flush=True)
print(f"delta topq     vs exp_048-ref: {topq_r - BASELINE_TOPQ:+.4f}  <- coordinator's gating metric", flush=True)

print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
