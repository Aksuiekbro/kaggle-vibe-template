import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

# Load cached features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train_exp003.npy")
X_test_path = os.path.join(cache_dir, "X_test_exp003.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Combined clean features not found.")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
print(f"Loaded features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 82,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 80, 250),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(15, verbose=False)]
        )
        
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        
    return np.mean(cv_scores)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")

def logging_callback(study, trial):
    print(f"Trial {trial.number:02d} completed. Value: {trial.value:.4f}. Best Value so far: {study.best_value:.4f}", flush=True)

print("Starting Optuna hyperparameter search on combined clean features (15 trials)...", flush=True)
t0 = time.time()
study.optimize(objective, n_trials=15, callbacks=[logging_callback])
elapsed = time.time() - t0
print(f"Finished 15 trials in {elapsed:.2f}s", flush=True)

best_params = study.best_params
best_score = study.best_value
print(f"\nBest CV Accuracy: {best_score:.4f}", flush=True)

# Train the final model with best params to generate OOF and test probabilities
print("\nTraining final LightGBM model with best parameters...", flush=True)
oof_probs = np.zeros((len(X_train), 82))
test_probs = np.zeros((len(X_test), 82))
cv_scores = []

final_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    **best_params
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**final_params)
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
np.save(os.path.join(cache_dir, "oof_probs_clean_lgb.npy"), oof_probs)
np.save(os.path.join(cache_dir, "test_probs_clean_lgb.npy"), test_probs)

results = {
    "best_cv": best_score,
    "best_params": best_params,
    "cv_scores": cv_scores,
    "cv_mean": np.mean(cv_scores),
    "cv_std": np.std(cv_scores)
}

with open(os.path.join(cache_dir, "results_clean_lgb.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved tuned results and probabilities to agents/gemini/workspace/")
