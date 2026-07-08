"""exp_016: MLP-alone vs LGBM-alone on FULL grid features (2330 rows/82
classes), single stratified 80/20 holdout (exp_012's cost-fix -- full
row/class coverage, cut fold count not row count). Resolves whether
probe_mlp_grid.py's tiny-subsample verdict (MLP OOF 0.2096 vs LGBM 0.5876)
holds at full scale, or reverses like exp_010's LGBM hparam probe did.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

X = train[feat_cols].values
y_raw = train["Pitch_ID"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(np.unique(y))

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

t0 = time.time()
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
)
clf_lgb.fit(Xtr, ytr)
acc_lgb = accuracy_score(yva, clf_lgb.predict(Xva))
print(f"LGBM holdout acc: {acc_lgb:.4f} ({time.time()-t0:.0f}s)")

t1 = time.time()
sc = StandardScaler().fit(Xtr)
Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
    early_stopping=True, random_state=42,
)
clf_mlp.fit(Xtr_s, ytr)
acc_mlp = accuracy_score(yva, clf_mlp.predict(Xva_s))
print(f"MLP holdout acc:  {acc_mlp:.4f} ({time.time()-t1:.0f}s)")

print(f"\nn_train={len(Xtr)} n_holdout={len(Xva)} classes={n_classes} n_feats={len(feat_cols)}")
print(f"delta (mlp - lgbm): {acc_mlp - acc_lgb:+.4f}")
print(f"total time: {time.time()-t0:.0f}s")
