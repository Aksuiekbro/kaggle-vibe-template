import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Load cached features
cache_dir = "agents/gemini/workspace"
X_train_1 = np.load(os.path.join(cache_dir, "X_train.npy"))
X_test_1 = np.load(os.path.join(cache_dir, "X_test.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_2 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_test_2 = np.load(os.path.join(cache_dir, "X_test_exp003.npy"))

print(f"Model 1 (Baseline) Train shape: {X_train_1.shape}, Test shape: {X_test_1.shape}")
print(f"Model 2 (Combined Clean) Train shape: {X_train_2.shape}, Test shape: {X_test_2.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Best params from exp002
params_1 = {
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

# Use similar robust parameters for Model 2
params_2 = {
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

oof_probs_1 = np.zeros((len(df_train), 82))
oof_probs_2 = np.zeros((len(df_train), 82))
test_probs_1 = np.zeros((len(df_test), 82))
test_probs_2 = np.zeros((len(df_test), 82))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_1, y_train)):
    print(f"\n--- Training Fold {fold} ---")
    
    # Model 1
    print("Training Model 1 (Baseline features)...")
    X_tr1, y_tr1 = X_train_1[train_idx], y_train[train_idx]
    X_va1, y_va1 = X_train_1[val_idx], y_train[val_idx]
    
    model1 = lgb.LGBMClassifier(**params_1)
    t0 = time.time()
    model1.fit(
        X_tr1, y_tr1,
        eval_set=[(X_va1, y_va1)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    oof_probs_1[val_idx] = model1.predict_proba(X_va1)
    test_probs_1 += model1.predict_proba(X_test_1) / 5.0
    print(f"Model 1 trained in {time.time() - t0:.2f}s")
    
    # Model 2
    print("Training Model 2 (Combined Clean features)...")
    X_tr2, y_tr2 = X_train_2[train_idx], y_train[train_idx]
    X_va2, y_va2 = X_train_2[val_idx], y_train[val_idx]
    
    model2 = lgb.LGBMClassifier(**params_2)
    t0 = time.time()
    model2.fit(
        X_tr2, y_tr2,
        eval_set=[(X_va2, y_va2)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    oof_probs_2[val_idx] = model2.predict_proba(X_va2)
    test_probs_2 += model2.predict_proba(X_test_2) / 5.0
    print(f"Model 2 trained in {time.time() - t0:.2f}s")

# Let's find the best blending weight
print("\n--- Finding Best Blending Weight ---")
best_w = 0.5
best_acc = 0.0

# Scan weights for Model 1 vs Model 2
for w in np.linspace(0, 1, 21):
    blend_probs = w * oof_probs_1 + (1 - w) * oof_probs_2
    preds = np.argmax(blend_probs, axis=1)
    acc = accuracy_score(y_train, preds)
    print(f"Weight {w:.2f} (Model 1): OOF Accuracy = {acc:.6f}")
    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"\nBest Blending Weight (Model 1): {best_w:.2f} with OOF Accuracy = {best_acc:.6f}")

# Generate final blended predictions
final_blend_probs = best_w * test_probs_1 + (1 - best_w) * test_probs_2
final_preds = np.argmax(final_blend_probs, axis=1)

# Fold-by-fold accuracy for the best blend weight
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_1, y_train)):
    fold_probs = best_w * oof_probs_1[val_idx] + (1 - best_w) * oof_probs_2[val_idx]
    fold_preds = np.argmax(fold_probs, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.4f}")

# Save submission
os.makedirs("agents/gemini/submissions", exist_ok=True)
sub_path = "agents/gemini/submissions/submission_ensemble.csv"
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

# Output results in standard JSON format
results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": best_acc,
    "cv_std": np.std(blend_cv_scores),
    "best_weight_model1": best_w,
    "submission_file": sub_path
}
with open("agents/gemini/workspace/results_ensemble.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved ensemble results to agents/gemini/workspace/results_ensemble.json")
