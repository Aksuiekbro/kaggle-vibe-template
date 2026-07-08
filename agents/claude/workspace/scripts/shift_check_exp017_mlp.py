"""exp_020: is the MLP component of exp_017's LGBM+MLP blend more shift-
fragile than LGBM, on the SAME grid features exp_006 already checked?

sub_022 (LGBM+MLP blend, CV OOF 0.9476) scored WORSE on public LB (0.87931)
than sub_020 (LGBM-only, CV OOF 0.9086, LB 0.91954) -- a CV-up/LB-down sign
disagreement. exp_006 found LGBM-alone has almost no accuracy gap between
test-like and non-test-like train quartiles (gap -0.0095). This reruns that
same adversarial-weighting methodology separately on the already-saved
oof_lgb_exp017.npy and oof_mlp_exp017.npy to see whether MLP's gap is larger.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

X = train[feat_cols].values
Xte = test[feat_cols].values

X_adv = np.vstack([X, Xte])
y_adv = np.concatenate([np.zeros(len(X)), np.ones(len(Xte))])
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
print(f"adversarial AUC (train vs test): {adv_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# exp_017's saved OOF proba arrays -- same row order as train_grid_features.parquet
oof_lgb = np.load(DIR + "oof_lgb_exp017.npy")
oof_mlp = np.load(DIR + "oof_mlp_exp017.npy")
y = np.load(DIR + "y_exp017.npy")

pred_lgb = np.argmax(oof_lgb, axis=1)
pred_mlp = np.argmax(oof_mlp, axis=1)
W_LGB = 0.4
pred_blend = np.argmax(W_LGB * oof_lgb + (1 - W_LGB) * oof_mlp, axis=1)

w = p_test_train
q75 = np.quantile(w, 0.75)
q25 = np.quantile(w, 0.25)
top_mask = w >= q75
bottom_mask = w <= q25

for name, pred in [("LGBM", pred_lgb), ("MLP", pred_mlp), ("blend(w=0.4)", pred_blend)]:
    correct = (pred == y).astype(float)
    plain_acc = correct.mean()
    topq_acc = correct[top_mask].mean()
    bottomq_acc = correct[bottom_mask].mean()
    gap = plain_acc - topq_acc
    print(f"{name}: plain={plain_acc:.4f} most-test-like-quartile={topq_acc:.4f} "
          f"least-test-like-quartile={bottomq_acc:.4f} gap={gap:+.4f}", flush=True)

print(f"total time: {time.time()-t0:.0f}s", flush=True)
