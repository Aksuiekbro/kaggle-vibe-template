import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_test_exp003 = np.load(os.path.join(cache_dir, "X_test_exp003.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_grid = np.load(os.path.join(cache_dir, "X_train_grid.npy"))
X_test_grid = np.load(os.path.join(cache_dir, "X_test_grid.npy"))

# Concatenate features
X_train = np.hstack([X_train_exp003, X_train_grid])
X_test = np.hstack([X_test_exp003, X_test_grid])

print(f"Concatenated features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

seeds = [42, 100, 2026, 777, 999]
oof_probs_ms = np.zeros((len(X_train), 82))
test_probs_ms = np.zeros((len(X_test), 82))
mlp_cv_scores = []

# Base parameters for MLP
params_base = {
    'hidden_layer_sizes': (512, 256),  # Larger network for more features
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'batch_size': 128,
    'learning_rate_init': 0.002,
    'max_iter': 120,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10
}

print(f"Training Multi-Seed MLP Classifier (seeds: {seeds}) with 5-fold CV...")
t0 = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)
    
    fold_val_probs = np.zeros((len(val_idx), 82))
    fold_te_probs = np.zeros((len(X_test), 82))
    fold_t0 = time.time()
    
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **params_base)
        model.fit(X_tr_scaled, y_tr)
        
        fold_val_probs += model.predict_proba(X_va_scaled) / len(seeds)
        fold_te_probs += model.predict_proba(X_te_scaled) / len(seeds)
        
    oof_probs_ms[val_idx] = fold_val_probs
    test_probs_ms += fold_te_probs / 5.0
    
    preds = np.argmax(fold_val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    mlp_cv_scores.append(acc)
    print(f"Fold {fold} MLP Multi-Seed Accuracy: {acc:.5f} in {time.time() - fold_t0:.2f}s")

ms_cv_mean = np.mean(mlp_cv_scores)
ms_cv_std = np.std(mlp_cv_scores)
print(f"\nMulti-Seed MLP CV Accuracy: {ms_cv_mean:.5f} (std: {ms_cv_std:.5f}) in {time.time() - t0:.1f}s")

# Let's save these MLP probabilities
np.save(os.path.join(cache_dir, "oof_probs_ms_mlp_grid.npy"), oof_probs_ms)
np.save(os.path.join(cache_dir, "test_probs_ms_mlp_grid.npy"), test_probs_ms)

# Now train LightGBM on the concatenated features
print("\nTraining LightGBM on concatenated features...")
oof_probs_lgb = np.zeros((len(X_train), 82))
test_probs_lgb = np.zeros((len(X_test), 82))
lgb_cv_scores = []

# LightGBM params
lgb_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'n_estimators': 250,
    'max_depth': 4,
    'num_leaves': 15,
    'min_child_samples': 5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    fold_t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    
    val_probs = model.predict_proba(X_va)
    oof_probs_lgb[val_idx] = val_probs
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    lgb_cv_scores.append(acc)
    print(f"LGB Fold {fold} Accuracy: {acc:.5f} in {time.time() - fold_t0:.2f}s")
    
    test_probs_lgb += model.predict_proba(X_test) / 5.0

lgb_cv_mean = np.mean(lgb_cv_scores)
lgb_cv_std = np.std(lgb_cv_scores)
print(f"LGB CV Accuracy: {lgb_cv_mean:.5f} (std: {lgb_cv_std:.5f})")

# Save LGB probabilities
np.save(os.path.join(cache_dir, "oof_probs_lgb_grid.npy"), oof_probs_lgb)
np.save(os.path.join(cache_dir, "test_probs_lgb_grid.npy"), test_probs_lgb)

# Perform ensembling
print("\n--- Finding Best Blending Weight ---")
best_w = 1.0
best_acc = 0.0

for w in np.linspace(0, 1, 101):
    blend_probs = w * oof_probs_ms + (1 - w) * oof_probs_lgb
    preds = np.argmax(blend_probs, axis=1)
    acc = accuracy_score(y_train, preds)
    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"Best MLP Weight: {best_w:.2f} with Blended OOF Accuracy = {best_acc:.6f}")

blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_probs = best_w * oof_probs_ms[val_idx] + (1 - best_w) * oof_probs_lgb[val_idx]
    fold_preds = np.argmax(fold_probs, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.5f}")

blend_cv_mean = np.mean(blend_cv_scores)
blend_cv_std = np.std(blend_cv_scores)
print(f"Blended CV Accuracy: {blend_cv_mean:.5f} (std: {blend_cv_std:.5f})")

# Save submission
sub_path = "agents/gemini/submissions/submission_ensemble_grid.csv"
final_blend_probs = best_w * test_probs_ms + (1 - best_w) * test_probs_lgb
final_preds = np.argmax(final_blend_probs, axis=1)

df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

# Save results
results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": blend_cv_mean,
    "cv_std": blend_cv_std,
    "best_weight_mlp": best_w,
    "submission_file": sub_path
}
with open(os.path.join(cache_dir, "results_ensemble_grid.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved ensemble results to agents/gemini/workspace/results_ensemble_grid.json")
