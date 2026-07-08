"""exp_021: logistic-regression stacking meta-learner on exp_017's saved
LGBM+MLP OOF probabilities, in place of the fixed scalar blend weight
(w_lgb=0.4, best-blend OOF 0.9476). Nested 3-fold CV on the meta-learner
itself, reusing the same StratifiedKFold(3, random_state=42) row groups so
the meta-learner is never fit on a row it's evaluated on.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

oof_lgb = np.load(DIR + "oof_lgb_exp017.npy")
oof_mlp = np.load(DIR + "oof_mlp_exp017.npy")
y = np.load(DIR + "y_exp017.npy")

X_meta = np.hstack([oof_lgb, oof_mlp])
n = len(y)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
meta_pred = np.zeros(n, dtype=int)
for tr_idx, va_idx in skf.split(X_meta, y):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_meta[tr_idx], y[tr_idx])
    meta_pred[va_idx] = clf.predict(X_meta[va_idx])

acc_meta = accuracy_score(y, meta_pred)

blend_fixed = 0.4 * oof_lgb + 0.6 * oof_mlp
acc_fixed = accuracy_score(y, np.argmax(blend_fixed, axis=1))

print(f"fixed-weight blend (w_lgb=0.4) OOF acc: {acc_fixed:.4f}")
print(f"logreg-stacked meta-learner OOF acc:    {acc_meta:.4f}")
print(f"delta (meta - fixed-weight blend):      {acc_meta - acc_fixed:+.4f}")
