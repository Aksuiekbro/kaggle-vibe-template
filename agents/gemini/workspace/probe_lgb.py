import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))
X_train_v4 = np.concatenate([X_train_base, X_train_pitch], axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train_base, y_train))

lgb_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'learning_rate': 0.0446,
    'n_estimators': 160,
    'max_depth': 3,
    'num_leaves': 24,
    'min_child_samples': 6,
    'subsample': 0.985,
    'colsample_bytree': 0.545,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Evaluate LGBM on Fold 0 for base features
print("Evaluating base LightGBM...")
t0 = time.time()
model_v1 = lgb.LGBMClassifier(**lgb_params)
model_v1.fit(
    X_train_base[train_idx], y_train[train_idx],
    eval_set=[(X_train_base[val_idx], y_train[val_idx])],
    callbacks=[lgb.early_stopping(15, verbose=False)]
)
preds_v1 = model_v1.predict(X_train_base[val_idx])
acc_v1 = accuracy_score(y_train[val_idx], preds_v1)
print(f"Base LightGBM Accuracy: {acc_v1:.5f} in {time.time() - t0:.2f}s")

# Evaluate LGBM on Fold 0 for v4 features
print("\nEvaluating LightGBM with pitch features...")
t0 = time.time()
model_v4 = lgb.LGBMClassifier(**lgb_params)
model_v4.fit(
    X_train_v4[train_idx], y_train[train_idx],
    eval_set=[(X_train_v4[val_idx], y_train[val_idx])],
    callbacks=[lgb.early_stopping(15, verbose=False)]
)
preds_v4 = model_v4.predict(X_train_v4[val_idx])
acc_v4 = accuracy_score(y_train[val_idx], preds_v4)
print(f"LightGBM with pitch features Accuracy: {acc_v4:.5f} in {time.time() - t0:.2f}s")

delta = acc_v4 - acc_v1
print(f"\nDelta Accuracy: {delta:+.5f}")
