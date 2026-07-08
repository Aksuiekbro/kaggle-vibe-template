import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")
df_test = pd.read_csv(test_csv)

# Load base model OOF probabilities
oof_mlp = np.load(os.path.join(cache_dir, "oof_probs_ms_mlp_enhanced_grid.npy"))
oof_svm = np.load(os.path.join(cache_dir, "oof_probs_svm_linear.npy"))
oof_lgb = np.load(os.path.join(cache_dir, "oof_probs_lgb_enhanced_grid.npy"))

test_mlp = np.load(os.path.join(cache_dir, "test_probs_ms_mlp_enhanced_grid.npy"))
test_svm = np.load(os.path.join(cache_dir, "test_probs_svm_linear.npy"))
test_lgb = np.load(os.path.join(cache_dir, "test_probs_lgb_enhanced_grid.npy"))

# Concatenate OOF probabilities
X_oof = np.hstack([oof_mlp, oof_svm, oof_lgb])
X_test = np.hstack([test_mlp, test_svm, test_lgb])

print(f"OOF Concatenated shape: {X_oof.shape}")
print(f"Test Concatenated shape: {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

meta_configs = [
    {
        "name": "LogisticRegression_L2_C=1.0",
        "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    },
    {
        "name": "LogisticRegression_L2_C=0.1",
        "model": LogisticRegression(max_iter=1000, C=0.1, random_state=42, n_jobs=-1)
    },
    {
        "name": "LogisticRegression_L2_C=10.0",
        "model": LogisticRegression(max_iter=1000, C=10.0, random_state=42, n_jobs=-1)
    },
    {
        "name": "MLPClassifier_64",
        "model": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True, random_state=42)
    },
    {
        "name": "MLPClassifier_128_64",
        "model": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=42)
    }
]

best_score = 0.0
best_meta = None

# Simple Weighted Ensemble Baseline (for comparison under the same 5-folds)
w_mlp, w_svm, w_lgb = 0.20, 0.74, 0.06
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_train)):
    fold_probs = (
        w_mlp * oof_mlp[val_idx] +
        w_svm * oof_svm[val_idx] +
        w_lgb * oof_lgb[val_idx]
    )
    fold_preds = np.argmax(fold_probs, axis=1)
    blend_cv_scores.append(accuracy_score(y_train[val_idx], fold_preds))
print(f"Weighted Blend Baseline 5-Fold CV: {np.mean(blend_cv_scores):.5f} (std: {np.std(blend_cv_scores):.5f})")

for cfg in meta_configs:
    name = cfg["name"]
    model = cfg["model"]
    
    cv_scores = []
    oof_preds = np.zeros(len(y_train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_train)):
        X_tr, y_tr = X_oof[train_idx], y_train[train_idx]
        X_va, y_va = X_oof[val_idx], y_train[val_idx]
        
        # Scaling is helpful for LR/MLP meta-classifiers
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        oof_preds[val_idx] = preds
        cv_scores.append(accuracy_score(y_train[val_idx], preds))
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"Meta-Classifier {name} -> Mean 5-Fold CV: {mean_acc:.5f} (std: {std_acc:.5f})")
    
    if mean_acc > best_score:
        best_score = mean_acc
        best_meta = name
        best_oof_preds = oof_preds
        best_model = model

print(f"\nBest Stacking Meta-Classifier: {best_meta} with CV: {best_score:.5f}")

if best_score > np.mean(blend_cv_scores):
    print("Stacking outperforms simple weighted blend. Training final stack and generating submission...")
    
    # Train meta-classifier on all OOF data to predict test set
    scaler = StandardScaler()
    X_oof_scaled = scaler.fit_transform(X_oof)
    X_test_scaled = scaler.transform(X_test)
    
    # Best model fit on full OOF
    best_model.fit(X_oof_scaled, y_train)
    final_preds = best_model.predict(X_test_scaled)
    
    # Save submission
    sub_path = "agents/gemini/submissions/submission_stacking.csv"
    df_sub = pd.DataFrame({
        'Path': df_test['Path'],
        'Pitch_ID': final_preds
    })
    df_sub.to_csv(sub_path, index=False)
    print(f"Saved stacking submission to {sub_path}")
    
    # Save results
    results = {
        "cv_scores": blend_cv_scores,
        "cv_mean": best_score,
        "cv_std": np.std(blend_cv_scores),
        "best_meta_classifier": best_meta,
        "submission_file": sub_path
    }
    with open(os.path.join(cache_dir, "results_stacking.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved stacking results to agents/gemini/workspace/results_stacking.json")
else:
    print("Stacking does not outperform simple weighted blend. Keep the current ensemble.")
