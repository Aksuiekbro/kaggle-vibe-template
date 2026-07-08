"""exp_030: fit MLP-ONLY (no LGBM blend) on the FULL train set (grid features,
exp_005 feature family) and predict test. Direct empirical test of whether the
blend ARCHITECTURE (not model family) explains the CV-up/LB-down pattern seen
on sub_022/sub_024 -- sub_020 (LGBM-alone) is the only single-model submission
so far and is also the LB-best. MLP-alone was never itself submitted.
CV reference: exp_017's saved oof_mlp_exp017.npy gives OOF acc 0.9288 (3-fold).
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

Xtr = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
ytr = le.fit_transform(y_raw)

Xte = test[feat_cols].values

t0 = time.time()
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
    early_stopping=True, random_state=42,
)
clf_mlp.fit(Xtr_s, ytr)
proba_mlp = clf_mlp.predict_proba(Xte_s)
n_classes = len(le.classes_)
full_proba_mlp = np.zeros((len(Xte), n_classes))
full_proba_mlp[:, clf_mlp.classes_] = proba_mlp
print(f"MLP fit+predict done ({time.time()-t0:.0f}s)", flush=True)

pred_idx = np.argmax(full_proba_mlp, axis=1)
pred = le.inverse_transform(pred_idx)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": pred})
out_path = DATA_DIR + "submission_exp030_mlp_only.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows")
