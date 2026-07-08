"""exp_035 (full fidelity): does the octave-ratio feature (probe delta +0.0172,
scheduler-promoted) survive the exp_006-style shift-aware check, for LGBM-ALONE
(no MLP, no blend, no augmentation -- isolating this one feature's effect)?

oct_ratio_d = harm_low[d] / (harm_low[d-12] + harm_low[d+12] + eps), targeting
exp_034's confirmed octave-confusion error mode (58% of OOF errors at
7/12/19/24-semitone harmonic-ratio distances vs 27% null).

Per PLAN.md coordinator note #2 (submission freeze on untrusted CV proxy):
"submit ONLY when the exp_006-style shift-aware holdout (most-test-like-
quartile accuracy) improves over the sub_020 equivalent." This script produces
that comparison: grid-only (sub_020-equivalent) vs grid+oct_ratio, both
LGBM-alone, under adversarial test-likeness weighting. Same structure/params as
exp_033 for direct comparability.
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
N_SPLITS = 3
SEED = 42
EPS = 1e-8
MIDI_LO, MIDI_HI = 30, 110

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def add_oct_ratio_features(df):
    df = df.copy()
    for d in range(MIDI_LO, MIDI_HI + 1):
        low_col = f"d{d}_harm_low"
        neighbors = [df[f"d{nd}_harm_low"] for nd in (d - 12, d + 12) if MIDI_LO <= nd <= MIDI_HI]
        neighbor_sum = sum(neighbors) if neighbors else pd.Series(0.0, index=df.index)
        df[f"d{d}_oct_ratio"] = df[low_col] / (neighbor_sum + EPS)
    return df


t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
train_oct = add_oct_ratio_features(train)
test_oct = add_oct_ratio_features(test)

feat_cols_base = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
feat_cols_oct = [c for c in train_oct.columns if c not in ("Path", "Pitch_ID")]

le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X_base = train[feat_cols_base].values
X_oct = train_oct[feat_cols_oct].values

# --- 1. adversarial classifier: train vs test, OOF P(is_test) per train row (grid-only features, matches exp_033) ---
X_adv = np.vstack([X_base, test[feat_cols_base].values])
y_adv = np.concatenate([np.zeros(len(X_base)), np.ones(len(test))])
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
w = p_test_all[: len(X_base)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# --- 2. fold-safe 3-fold CV: grid-only baseline vs grid+oct_ratio, LGBM-alone ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
pred_base = np.zeros(len(y), dtype=int)
pred_oct = np.zeros(len(y), dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base, y)):
    t_fold = time.time()
    clf_base = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_base.fit(X_base[tr_idx], y[tr_idx])
    pred_base[va_idx] = clf_base.predict(X_base[va_idx])

    clf_oct = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_oct.fit(X_oct[tr_idx], y[tr_idx])
    pred_oct[va_idx] = clf_oct.predict(X_oct[va_idx])

    print(f"fold {fold} done ({time.time()-t_fold:.0f}s): "
          f"base_acc={accuracy_score(y[va_idx], pred_base[va_idx]):.4f} "
          f"oct_acc={accuracy_score(y[va_idx], pred_oct[va_idx]):.4f}", flush=True)

correct_base = (pred_base == y).astype(float)
correct_oct = (pred_oct == y).astype(float)

q75 = np.quantile(w, 0.75)
top_mask = w >= q75


def report(name, correct):
    plain = correct.mean()
    weighted = np.average(correct, weights=w)
    topq = correct[top_mask].mean()
    print(f"{name}: plain={plain:.4f} weighted={weighted:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})", flush=True)
    return plain, weighted, topq


print("\n=== shift-aware comparison: LGBM-alone, grid-only vs grid+oct_ratio ===", flush=True)
plain_b, weighted_b, topq_b = report("grid-only (sub_020-equivalent)", correct_base)
plain_o, weighted_o, topq_o = report("grid+oct_ratio", correct_oct)
print(f"\ndelta plain:   {plain_o - plain_b:+.4f}", flush=True)
print(f"delta weighted:{weighted_o - weighted_b:+.4f}", flush=True)
print(f"delta topq:    {topq_o - topq_b:+.4f}  <- coordinator's gating metric", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
