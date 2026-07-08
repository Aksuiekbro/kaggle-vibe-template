import os
import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

# Load cached baseline features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Baseline features not found. Run train_optimized.py first.")

X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
print(f"Loaded train features shape: {X_train.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 82,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 80, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'min_child_samples': trial.suggest_int('min_child_samples', 4, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    cv_scores = []
    # Stratified 5-Fold evaluation (mandatory k>=5)
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
    # Save incremental study results
    results = {
        "best_cv": study.best_value,
        "best_params": study.best_params,
        "trials_completed": len(study.trials)
    }
    with open("agents/gemini/workspace/results_exp002.json", "w") as f:
        json.dump(results, f, indent=2)

print("Starting Optuna hyperparameter search (12 trials max)...", flush=True)
t0 = time.time()
study.optimize(objective, n_trials=12, callbacks=[logging_callback])
elapsed = time.time() - t0
print(f"Finished 12 trials in {elapsed:.2f}s", flush=True)

best_params = study.best_params
best_score = study.best_value
print(f"\nBest CV Accuracy: {best_score:.4f}", flush=True)
print("Best parameters:", flush=True)
for k, v in best_params.items():
    print(f"  '{k}': {v},", flush=True)

