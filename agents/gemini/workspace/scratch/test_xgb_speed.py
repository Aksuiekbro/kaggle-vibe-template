import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

cache_dir = "agents/gemini/workspace"
X_train = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

print(f"Loaded features shapes: Train {X_train.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': 82,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'n_estimators': 40,
    'max_depth': 3,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 5
}

cv_scores = []
t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = xgb.XGBClassifier(**params)
    fold_t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} accuracy: {acc:.4f} in {time.time() - fold_t0:.2f}s (best iteration: {model.best_iteration})")

elapsed = time.time() - t0
print(f"5-fold CV completed in {elapsed:.2f} seconds.")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")
