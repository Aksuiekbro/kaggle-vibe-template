import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Load clean concatenated features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train_exp003.npy")
X_test_path = os.path.join(cache_dir, "X_test_exp003.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(X_test_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Clean concatenated features not found. Run adv_selection.py first.")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
print(f"Loaded clean features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 82,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 80, 220),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 12),
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

print("Bypassing Optuna hyperparameter search to run a fast probe...", flush=True)
best_params = {
    'learning_rate': 0.044647267426869655,
    'n_estimators': 160,
    'max_depth': 3,
    'num_leaves': 24,
    'min_child_samples': 6,
    'subsample': 0.9845881298914211,
    'colsample_bytree': 0.5445651668276759
}
best_score = 0.0

# Final evaluation with best params
print("\nTraining final model with best parameters...", flush=True)
cv_scores = []
test_preds_probs = np.zeros((len(df_test), 82))
oof_preds = np.zeros(len(df_train))

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
    t_start = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    preds = model.predict(X_va)
    oof_preds[val_idx] = preds
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f} in {time.time() - t_start:.2f}s", flush=True)
    
    test_preds_probs += model.predict_proba(X_test) / 5.0

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\nFinal CV Accuracy Mean: {cv_mean:.4f} Std: {cv_std:.4f}", flush=True)

# Save submission
sub_path = "agents/gemini/submissions/submission_exp003.csv"
final_preds = np.argmax(test_preds_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path}", flush=True)

# Save results
results = {
    "cv_scores": cv_scores,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "best_params": best_params,
    "submission_file": sub_path
}
with open("agents/gemini/workspace/results_exp003_final.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved final results to agents/gemini/workspace/results_exp003_final.json", flush=True)
