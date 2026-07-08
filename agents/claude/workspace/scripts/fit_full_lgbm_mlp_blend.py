"""exp_017 follow-through: fit LGBM + MLP on the FULL train set (grid features,
exp_005 feature family) and predict test with the w_lgb=0.4 soft-vote blend
that exp_017's 3-fold CV found best (OOF 0.9476 vs LGBM-only 0.9090 /
MLP-only 0.9288). Writes a submission.csv for the score gate.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
W_LGB = 0.4

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

Xtr = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
ytr = le.fit_transform(y_raw)
n_classes = len(np.unique(ytr))

Xte = test[feat_cols].values

t0 = time.time()
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
)
clf_lgb.fit(Xtr, ytr)
proba_lgb = clf_lgb.predict_proba(Xte)
full_proba_lgb = np.zeros((len(Xte), n_classes))
full_proba_lgb[:, clf_lgb.classes_] = proba_lgb
print(f"LGBM fit+predict done ({time.time()-t0:.0f}s)", flush=True)

t1 = time.time()
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
    early_stopping=True, random_state=42,
)
clf_mlp.fit(Xtr_s, ytr)
proba_mlp = clf_mlp.predict_proba(Xte_s)
full_proba_mlp = np.zeros((len(Xte), n_classes))
full_proba_mlp[:, clf_mlp.classes_] = proba_mlp
print(f"MLP fit+predict done ({time.time()-t1:.0f}s)", flush=True)

blend_proba = W_LGB * full_proba_lgb + (1 - W_LGB) * full_proba_mlp
blend_pred_idx = np.argmax(blend_proba, axis=1)
blend_pred = le.inverse_transform(blend_pred_idx)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": blend_pred})
out_path = DATA_DIR + "submission_exp017_blend.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows")
