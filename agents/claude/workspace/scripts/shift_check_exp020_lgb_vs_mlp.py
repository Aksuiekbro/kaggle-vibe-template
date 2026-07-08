"""exp_020: is the MLP component of exp_017's LGBM+MLP blend more shift-fragile
than the LGBM component, on the *same* grid features? exp_006 only checked
LGBM (gap -0.0095, effectively shift-robust). sub_022 (the blend, CV 0.9476)
scored WORSE on public LB (0.87931) than sub_020 (LGBM-only, CV 0.9086, LB
0.91954) -- a CV-up/LB-down sign flip that exp_017's proper 3-fold CV can't
explain by itself (fold std was only 0.0088, not probe noise).

Method: reuse exp_017's saved per-model OOF probability arrays
(oof_lgb_exp017.npy, oof_mlp_exp017.npy, y_exp017.npy) -- no refitting needed.
Refit only the adversarial train-vs-test classifier (same recipe as exp_006)
to get P(is_test) per train row, then compare the plain-vs-most-test-like-
quartile accuracy gap separately for LGBM-only and MLP-only predictions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

oof_lgb = np.load(DIR + "oof_lgb_exp017.npy")
oof_mlp = np.load(DIR + "oof_mlp_exp017.npy")
y = np.load(DIR + "y_exp017.npy")
assert len(train) == len(y), "row count mismatch vs exp_017 saved arrays"

X = train[feat_cols].values
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
w = p_test_all[: len(X)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f}")

q75 = np.quantile(w, 0.75)
q25 = np.quantile(w, 0.25)
top_mask = w >= q75
bottom_mask = w <= q25


def report(name, oof_proba):
    pred = np.argmax(oof_proba, axis=1)
    correct = (pred == y).astype(float)
    plain_acc = correct.mean()
    weighted_acc = np.average(correct, weights=w)
    topq_acc = correct[top_mask].mean()
    bottomq_acc = correct[bottom_mask].mean()
    gap = plain_acc - topq_acc
    print(f"\n{name}")
    print(f"  plain OOF acc:                   {plain_acc:.4f}")
    print(f"  test-likeness-weighted acc:      {weighted_acc:.4f}")
    print(f"  most-test-like quartile acc:     {topq_acc:.4f}  (n={top_mask.sum()})")
    print(f"  least-test-like quartile acc:    {bottomq_acc:.4f}  (n={bottom_mask.sum()})")
    print(f"  gap (plain - most-test-like):    {gap:+.4f}")
    return gap


gap_lgb = report("LGBM-only (exp_017 oof_lgb)", oof_lgb)
gap_mlp = report("MLP-only (exp_017 oof_mlp)", oof_mlp)
blend = 0.4 * oof_lgb + 0.6 * oof_mlp
gap_blend = report("LGBM+MLP blend w_lgb=0.4 (sub_022 config)", blend)

print("\n=== summary ===")
print(f"gap_lgb={gap_lgb:+.4f}  gap_mlp={gap_mlp:+.4f}  gap_blend={gap_blend:+.4f}")
print(f"exp_006 reference gap_lgb (own refit, different params): -0.0095")
print(f"MLP more shift-fragile than LGBM: {gap_mlp < gap_lgb - 0.01}")
