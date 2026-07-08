import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_test_exp003 = np.load(os.path.join(cache_dir, "X_test_exp003.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
X_test_grid = np.load(os.path.join(cache_dir, "X_test_enhanced_grid.npy"))

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
    'hidden_layer_sizes': (512, 256),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.02,
    'batch_size': 128,
    'learning_rate_init': 0.002,
    'max_iter': 130,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10
}

print(f"Training Multi-Seed MLP Classifier with PCA=512 (seeds: {seeds}) with 5-fold CV...")
t0 = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Preprocessing: StandardScaler + PCA(512)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=512, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_va_pca = pca.transform(X_va_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    
    fold_val_probs = np.zeros((len(val_idx), 82))
    fold_te_probs = np.zeros((len(X_test), 82))
    fold_t0 = time.time()
    
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **params_base)
        model.fit(X_tr_pca, y_tr)
        
        fold_val_probs += model.predict_proba(X_va_pca) / len(seeds)
        fold_te_probs += model.predict_proba(X_te_pca) / len(seeds)
        
    oof_probs_ms[val_idx] = fold_val_probs
    test_probs_ms += fold_te_probs / 5.0
    
    preds = np.argmax(fold_val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    mlp_cv_scores.append(acc)
    print(f"Fold {fold} MLP Multi-Seed Accuracy: {acc:.5f} in {time.time() - fold_t0:.2f}s")

ms_cv_mean = np.mean(mlp_cv_scores)
ms_cv_std = np.std(mlp_cv_scores)
print(f"\nMulti-Seed PCA-MLP CV Accuracy: {ms_cv_mean:.5f} (std: {ms_cv_std:.5f}) in {time.time() - t0:.1f}s")

# Save probabilities
np.save(os.path.join(cache_dir, "oof_probs_ms_mlp_pca512.npy"), oof_probs_ms)
np.save(os.path.join(cache_dir, "test_probs_ms_mlp_pca512.npy"), test_probs_ms)

# Save submission
sub_path = "agents/gemini/submissions/submission_ms_mlp_pca512.csv"
final_preds = np.argmax(test_probs_ms, axis=1)

df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved PCA-MLP submission to {sub_path}")

# Save results
results = {
    "cv_scores": mlp_cv_scores,
    "cv_mean": ms_cv_mean,
    "cv_std": ms_cv_std,
    "submission_file": sub_path
}
with open(os.path.join(cache_dir, "results_ms_mlp_pca512.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved PCA-MLP results to agents/gemini/workspace/results_ms_mlp_pca512.json")
