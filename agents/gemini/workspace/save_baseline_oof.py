import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load cached features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train.npy")
X_test_path = os.path.join(cache_dir, "X_test.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Baseline features not found.")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
print(f"Loaded features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'learning_rate': 0.044647267426869655,
    'n_estimators': 160,
    'max_depth': 3,
    'num_leaves': 24,
    'min_child_samples': 6,
    'subsample': 0.9845881298914211,
    'colsample_bytree': 0.5445651668276759,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Train the model to generate OOF and test probabilities
print("Training Baseline LightGBM model...", flush=True)
oof_probs = np.zeros((len(X_train), 82))
test_probs = np.zeros((len(X_test), 82))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    val_probs = model.predict_proba(X_va)
    oof_probs[val_idx] = val_probs
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}", flush=True)
    
    test_probs += model.predict_proba(X_test) / 5.0

print(f"\nFinal CV Accuracy: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")

# Save outputs
np.save(os.path.join(cache_dir, "oof_probs_baseline_lgb.npy"), oof_probs)
np.save(os.path.join(cache_dir, "test_probs_baseline_lgb.npy"), test_probs)
print("Saved baseline probabilities to agents/gemini/workspace/")
