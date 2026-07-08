"""exp_023: shift-aware blend-weight search between exp_017's oof_lgb and
oof_mlp (grid features, no retraining -- reuses saved OOF arrays).

sub_022 (plain-OOF-optimal w_lgb=0.4) scored WORSE on LB (0.87931) than
sub_020 (w_lgb=1.0, LGBM-only, LB 0.91954), despite a higher CV (0.9476 vs
0.9090). exp_011 already falsified "shift-aware scoring picks a different
weight" for a grid-vs-generic-feature blend. This tests the same mechanism
for a DIFFERENT blend pair: LGBM vs MLP, both on the same shift-robust grid
features (exp_006's finding). If shift-aware (most-test-like-quartile)
accuracy peaks near w=1.0 while plain OOF peaks near w=0.4, that is direct
evidence shift-aware scoring would have avoided the sub_022 LB drop.
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
w_test = p_test_all[: len(X)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f}")

q75 = np.quantile(w_test, 0.75)
topq_mask = w_test >= q75

print(f"\n{'w_lgb':>8} {'plain_acc':>10} {'shift_wtd_acc':>14} {'topq_acc':>10}")
results = []
for w in np.arange(0.0, 1.01, 0.1):
    proba_blend = w * oof_lgb + (1 - w) * oof_mlp
    pred = proba_blend.argmax(axis=1)
    correct = (pred == y).astype(float)
    plain_acc = correct.mean()
    shift_wtd_acc = np.average(correct, weights=w_test)
    topq_acc = correct[topq_mask].mean()
    results.append((w, plain_acc, shift_wtd_acc, topq_acc))
    print(f"{w:8.1f} {plain_acc:10.4f} {shift_wtd_acc:14.4f} {topq_acc:10.4f}")

best_plain = max(results, key=lambda r: r[1])
best_shift = max(results, key=lambda r: r[2])
best_topq = max(results, key=lambda r: r[3])
print(f"\nbest w_lgb by plain OOF accuracy:       w={best_plain[0]:.1f} (plain_acc={best_plain[1]:.4f})")
print(f"best w_lgb by shift-aware weighted acc: w={best_shift[0]:.1f} (shift_wtd_acc={best_shift[2]:.4f})")
print(f"best w_lgb by most-test-like quartile:  w={best_topq[0]:.1f} (topq_acc={best_topq[3]:.4f})")
print(f"\nsub_022 config w_lgb=0.4: plain={results[4][1]:.4f} topq={results[4][3]:.4f}")
print(f"sub_020 config w_lgb=1.0: plain={results[10][1]:.4f} topq={results[10][3]:.4f}")
print(f"shift-aware/topq would have picked closer to sub_020's weight: {best_topq[0] > 0.6}")
