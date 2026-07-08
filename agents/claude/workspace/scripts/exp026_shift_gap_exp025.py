"""exp_026 (own, cheap, no retraining): does exp_025's augmented LGBM+MLP
blend (candidate to replace sub_022, full CV 0.9524 vs sub_022's 0.9476)
show the same "clean gap despite real LB drop" pattern that exp_020/exp_023
found for sub_022's un-augmented blend? Reuses exp_025's saved OOF arrays
(oof_lgb_exp025.npy, oof_mlp_exp025.npy, y_exp025.npy) and exp_020's exact
adversarial-quartile-gap recipe -- no refitting of LGBM/MLP needed, only the
adversarial train-vs-test classifier.

This does NOT resolve whether exp_025 will transfer to LB (exp_020/023
already showed this diagnostic family fails to predict the sub_022 drop) --
it is a consistency check, not a green light. Recorded as a methodology data
point either way.
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

oof_lgb = np.load(DIR + "oof_lgb_exp025.npy")
oof_mlp = np.load(DIR + "oof_mlp_exp025.npy")
y = np.load(DIR + "y_exp025.npy")
assert len(train) == len(y), "row count mismatch vs exp_025 saved arrays"

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


gap_lgb = report("LGBM(aug)-only (exp_025 oof_lgb)", oof_lgb)
gap_mlp = report("MLP(aug)-only (exp_025 oof_mlp)", oof_mlp)
blend = 0.4 * oof_lgb + 0.6 * oof_mlp
gap_blend = report("LGBM(aug)+MLP(aug) blend w_lgb=0.4 (exp_025 config)", blend)

print("\n=== summary ===")
print(f"gap_lgb={gap_lgb:+.4f}  gap_mlp={gap_mlp:+.4f}  gap_blend={gap_blend:+.4f}")
print(f"exp_020 reference (un-augmented blend): gap_lgb=-0.0087 gap_mlp=-0.0129 gap_blend=-0.0078")
print(f"MLP more shift-fragile than LGBM: {gap_mlp < gap_lgb - 0.01}")
print(f"augmentation changed the gap picture materially: {abs(gap_blend - (-0.0078)) > 0.01}")
