import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")
df_test = pd.read_csv(test_csv)

models = {
    "mlp_pca512": {
        "oof": np.load(os.path.join(cache_dir, "oof_probs_ms_mlp_pca512.npy")),
        "test": np.load(os.path.join(cache_dir, "test_probs_ms_mlp_pca512.npy"))
    },
    "svm_pca512": {
        "oof": np.load(os.path.join(cache_dir, "oof_probs_svm_linear_c0.05_pca512.npy")),
        "test": np.load(os.path.join(cache_dir, "test_probs_svm_linear_c0.05_pca512.npy"))
    },
    "mlp_raw": {
        "oof": np.load(os.path.join(cache_dir, "oof_probs_ms_mlp_enhanced_grid.npy")),
        "test": np.load(os.path.join(cache_dir, "test_probs_ms_mlp_enhanced_grid.npy"))
    },
    "lgb_raw": {
        "oof": np.load(os.path.join(cache_dir, "oof_probs_lgb_enhanced_grid.npy")),
        "test": np.load(os.path.join(cache_dir, "test_probs_lgb_enhanced_grid.npy"))
    },
    "svm_raw": {
        "oof": np.load(os.path.join(cache_dir, "oof_probs_svm_linear.npy")),
        "test": np.load(os.path.join(cache_dir, "test_probs_svm_linear.npy"))
    }
}

for name, data in models.items():
    acc = accuracy_score(y_train, np.argmax(data["oof"], axis=1))
    print(f"Model {name} OOF Accuracy: {acc:.6f}")

model_names = list(models.keys())
num_models = len(model_names)

# Define objective function: negative accuracy (accuracy is not differentiable, so we could optimize soft cross-entropy or negative accuracy directly using COBYLA/Nelder-Mead)
def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalize
    
    blend = np.zeros_like(models[model_names[0]]["oof"])
    for i, name in enumerate(model_names):
        blend += weights[i] * models[name]["oof"]
        
    preds = np.argmax(blend, axis=1)
    acc = accuracy_score(y_train, preds)
    return -acc

# Run optimization using different starting weights
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
        options={'maxiter': 500}
    )
    
    opt_w = res.x
    opt_w = opt_w / np.sum(opt_w)
    
    # Calculate final accuracy
    acc = -objective(opt_w)
    if acc > best_acc:
        best_acc = acc
        best_weights = opt_w

best_weights = np.array(best_weights)
best_weights = best_weights / np.sum(best_weights)

print("\n--- Optimized Blending ---")
for name, weight in zip(model_names, best_weights):
    print(f"Weight for {name}: {weight:.4f}")
print(f"Blended OOF Accuracy: {best_acc:.6f}")

# Calculate CV scores per fold with these weights
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
blend_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(y_train, y_train)):
    fold_blend = np.zeros((len(val_idx), 82))
    for i, name in enumerate(model_names):
        fold_blend += best_weights[i] * models[name]["oof"][val_idx]
        
    fold_preds = np.argmax(fold_blend, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.5f}")

mean_cv = np.mean(blend_cv_scores)
std_cv = np.std(blend_cv_scores)
print(f"\nBlended 5-Fold CV Accuracy: {mean_cv:.5f} (std: {std_cv:.5f})")

# Save blended submissions
sub_path = "agents/gemini/submissions/submission_ensemble_all.csv"
final_probs = np.zeros((len(df_test), 82))
for i, name in enumerate(model_names):
    final_probs += best_weights[i] * models[name]["test"]

final_preds = np.argmax(final_probs, axis=1)

df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

# Save results
results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": mean_cv,
    "cv_std": std_cv,
    "weights": {name: float(w) for name, w in zip(model_names, best_weights)},
    "submission_file": sub_path
}
results_path = os.path.join(cache_dir, "results_ensemble_all.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
