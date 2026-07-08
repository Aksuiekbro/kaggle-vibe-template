import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

oof_files = {
    "pca_mlp_512": "oof_probs_mlp_pca_norm.npy",
    "pca_mlp_256": "oof_probs_mlp_pca_256_v3.npy",
    "raw_mlp_512_256": "oof_probs_mlp_raw_norm.npy",
    "raw_mlp_512_128": "oof_probs_mlp_raw_512_v3.npy",
    "svm_pca": "oof_probs_svm_pca_norm.npy",
    "lgb": "oof_probs_lgb_norm.npy",
    "grid_mlp": "oof_probs_mlp_grid_only.npy",
    "grid_svm": "oof_probs_svm_grid_only.npy",
    "grid_lgb": "oof_probs_lgb_grid_only.npy"
}

test_files = {
    "pca_mlp_512": "test_probs_mlp_pca_norm.npy",
    "pca_mlp_256": "test_probs_mlp_pca_256_v3.npy",
    "raw_mlp_512_256": "test_probs_mlp_raw_norm.npy",
    "raw_mlp_512_128": "test_probs_mlp_raw_512_v3.npy",
    "svm_pca": "test_probs_svm_pca_norm.npy",
    "lgb": "test_probs_lgb_norm.npy",
    "grid_mlp": "test_probs_mlp_grid_only.npy",
    "grid_svm": "test_probs_svm_grid_only.npy",
    "grid_lgb": "test_probs_lgb_grid_only.npy"
}

models = {}
test_probs = {}

for name, filename in oof_files.items():
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        models[name] = np.load(path)
        test_probs[name] = np.load(os.path.join(cache_dir, test_files[name]))
    else:
        print(f"Warning: {name} OOF file not found!")

model_names = list(models.keys())
num_models = len(model_names)
print(f"Loaded {num_models} models: {model_names}")

def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalize
    
    blend = np.zeros_like(models[model_names[0]])
    for i, name in enumerate(model_names):
        blend += weights[i] * models[name]
        
    preds = np.argmax(blend, axis=1)
    acc = accuracy_score(y_train, preds)
    return -acc

best_acc = 0.0
best_weights = None

for start_idx in range(num_models):
    init_weights = np.zeros(num_models)
    init_weights[start_idx] = 1.0
    
    res = minimize(
        objective, 
        init_weights, 
        method='COBYLA', 
        constraints={'type': 'ineq', 'fun': lambda w: w},
        options={'maxiter': 1000}
    )
    
    opt_w = res.x
    opt_w = opt_w / np.sum(opt_w)
    
    acc = -objective(opt_w)
    if acc > best_acc:
        best_acc = acc
        best_weights = opt_w

best_weights = np.array(best_weights)
best_weights = best_weights / np.sum(best_weights)

print("\n--- Optimized Blending ---")
for name, weight in zip(model_names, best_weights):
    if weight > 1e-4:
        print(f"Weight for {name:<15}: {weight:.4f}")
print(f"Blended OOF Accuracy: {best_acc:.6f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(models[model_names[0]], y_train)):
    fold_blend = np.zeros((len(val_idx), 82))
    for i, name in enumerate(model_names):
        fold_blend += best_weights[i] * models[name][val_idx]
        
    fold_preds = np.argmax(fold_blend, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.5f}")

mean_cv = np.mean(blend_cv_scores)
std_cv = np.std(blend_cv_scores)
print(f"\nBlended 5-Fold CV Accuracy: {mean_cv:.5f} (std: {std_cv:.5f})")

# Let's save a new submission if it beats the current best CV (0.968242)
data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")
df_test = pd.read_csv(test_csv)

sub_path = "agents/gemini/submissions/submission_super_blend.csv"
final_probs = np.zeros((len(df_test), 82))
for i, name in enumerate(model_names):
    final_probs += best_weights[i] * test_probs[name]

final_preds = np.argmax(final_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"Saved super blend submission to {sub_path}")

results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": mean_cv,
    "cv_std": std_cv,
    "weights": {name: float(w) for name, w in zip(model_names, best_weights)},
    "submission_file": sub_path
}
results_path = os.path.join(cache_dir, "results_super_blend.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
