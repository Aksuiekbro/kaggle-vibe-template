import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Load cached combined clean features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train_exp003.npy")
X_test_path = os.path.join(cache_dir, "X_test_exp003.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(X_test_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Combined clean features not found.")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
print(f"Loaded features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

seeds = [42, 100, 2026, 777, 999]
oof_probs_ms = np.zeros((len(X_train), 82))
test_probs_ms = np.zeros((len(X_test), 82))
cv_scores = []

# Base parameters
params_base = {
    'hidden_layer_sizes': (256, 128),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'batch_size': 128,
    'learning_rate_init': 0.002,
    'max_iter': 100,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 8
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
    cv_scores.append(acc)
    print(f"Fold {fold} Multi-Seed Accuracy: {acc:.5f} in {time.time() - fold_t0:.2f}s")

ms_cv_mean = np.mean(cv_scores)
ms_cv_std = np.std(cv_scores)
print(f"\nMulti-Seed MLP CV Accuracy: {ms_cv_mean:.5f} (std: {ms_cv_std:.5f}) in {time.time() - t0:.1f}s")

# Save multi-seed MLP OOF and test probabilities
np.save(os.path.join(cache_dir, "oof_probs_ms_mlp.npy"), oof_probs_ms)
np.save(os.path.join(cache_dir, "test_probs_ms_mlp.npy"), test_probs_ms)

# Load LightGBM OOF and test probabilities if available
oof_probs_lgb_path = os.path.join(cache_dir, "oof_probs_exp003_lgb.npy")
test_probs_lgb_path = os.path.join(cache_dir, "test_probs_exp003_lgb.npy")

if os.path.exists(oof_probs_lgb_path) and os.path.exists(test_probs_lgb_path):
    oof_probs_lgb = np.load(oof_probs_lgb_path)
    test_probs_lgb = np.load(test_probs_lgb_path)
    
    # Blending multi-seed MLP with LGB
    print("\n--- Finding Best Blending Weight between Multi-Seed MLP and LightGBM ---")
    best_w = 1.0
    best_acc = 0.0
    
    for w in np.linspace(0, 1, 101):
        blend_probs = w * oof_probs_ms + (1 - w) * oof_probs_lgb
        preds = np.argmax(blend_probs, axis=1)
        acc = accuracy_score(y_train, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            
    print(f"Best Multi-Seed MLP Weight: {best_w:.2f} with Blended OOF Accuracy = {best_acc:.6f}")
    
    # Fold-by-fold accuracy for the best blend weight
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
    
    # Save ensemble submission
    os.makedirs("agents/gemini/submissions", exist_ok=True)
    sub_path = "agents/gemini/submissions/submission_ensemble_ms_mlp_lgb.csv"
    
    final_blend_probs = best_w * test_probs_ms + (1 - best_w) * test_probs_lgb
    final_preds = np.argmax(final_blend_probs, axis=1)
    
    df_sub = pd.DataFrame({
        'Path': df_test['Path'],
        'Pitch_ID': final_preds
    })
    df_sub.to_csv(sub_path, index=False)
    print(f"\nSaved ensemble submission to {sub_path}")
    
    results = {
        "cv_scores": blend_cv_scores,
        "cv_mean": blend_cv_mean,
        "cv_std": blend_cv_std,
        "best_weight_ms_mlp": best_w,
        "submission_file": sub_path
    }
    with open(os.path.join(cache_dir, "results_ensemble_ms_mlp_lgb.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved ensemble results to agents/gemini/workspace/results_ensemble_ms_mlp_lgb.json")
else:
    print("\nLightGBM probabilities not found, skipping ensembling step.")
