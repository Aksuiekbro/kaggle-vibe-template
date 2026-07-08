"""exp_006: shift-aware validation for exp_005 (grid-harmonic features).

Coordinator PLAN.md priority 1: CV-vs-public-LB sign agreement is only 54% on
gemini's 14 submissions (verifiers.py cv-lb) -- plain StratifiedKFold CV isn't
trustworthy under the measured train/test shift (adversarial AUC ~0.998, exp_003).

Method: fit an adversarial train-vs-test classifier on the exp_005 grid
features (same family exp_003 used), get per-train-row P(is_test). Reproduce
exp_005's full 3-fold OOF using the same split (random_state=42) and compare:
  - plain (unweighted) OOF accuracy               <- what we reported before
  - test-likeness-weighted OOF accuracy            <- weights each row by
    P(is_test), so rows that resemble the test distribution count more
  - OOF accuracy restricted to the most test-like quartile of train rows
If the weighted/quartile numbers track close to plain OOF, the shift mostly
doesn't interact with class-relevant signal and plain CV is fine to trust.
If they diverge a lot, that's the mechanism behind the measured CV-LB gap.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")

feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
assert feat_cols == [c for c in test.columns if c not in ("Path", "Pitch_ID")]

class_counts = train["Pitch_ID"].value_counts()
N_SPLITS = 3
keep_classes = class_counts[class_counts >= N_SPLITS].index
train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)

# --- 1. adversarial classifier: train vs test, get P(is_test) for every train row (OOF, to avoid leakage) ---
X_adv = np.vstack([X, test[feat_cols].values])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(test))])
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
p_test_train = p_test_all[: len(X)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f}")
print(f"P(is_test) on train rows: min={p_test_train.min():.3f} mean={p_test_train.mean():.3f} max={p_test_train.max():.3f}")

# --- 2. reproduce exp_005's full 3-fold OOF (same split/params) ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_pred = np.zeros(len(y), dtype=int)
for tr_idx, va_idx in skf.split(X, y):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=len(np.unique(y)), random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X[tr_idx], y[tr_idx])
    oof_pred[va_idx] = clf.predict(X[va_idx])

correct = (oof_pred == y).astype(float)
plain_acc = correct.mean()

# --- 3. test-likeness-weighted accuracy ---
w = p_test_train
weighted_acc = np.average(correct, weights=w)

# --- 4. most-test-like quartile only ---
q75 = np.quantile(w, 0.75)
top_mask = w >= q75
topq_acc = correct[top_mask].mean()
q25 = np.quantile(w, 0.25)
bottom_mask = w <= q25
bottomq_acc = correct[bottom_mask].mean()

print(f"\nplain OOF accuracy:                    {plain_acc:.4f}  (n={len(y)})")
print(f"test-likeness-weighted OOF accuracy:    {weighted_acc:.4f}")
print(f"most-test-like quartile OOF accuracy:   {topq_acc:.4f}  (n={top_mask.sum()})")
print(f"least-test-like quartile OOF accuracy:  {bottomq_acc:.4f}  (n={bottom_mask.sum()})")
print(f"gap (plain - most-test-like quartile):  {plain_acc - topq_acc:+.4f}")
