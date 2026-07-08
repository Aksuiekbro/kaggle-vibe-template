import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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

# Sweep C values to find the best linear SVM
c_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
best_c = 1.0
best_cv_acc = 0.0
best_oof_probs = None
best_test_probs = None
best_cv_scores = []

for c in c_values:
    print(f"\nEvaluating Linear SVM with C={c}...")
    oof_probs = np.zeros((len(X_train), 82))
    test_probs = np.zeros((len(X_test), 82))
    cv_scores = []
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_test)
        
        model = SVC(kernel='linear', C=c, probability=True, random_state=42)
        model.fit(X_tr_scaled, y_tr)
        
        val_p = model.predict_proba(X_va_scaled)
        oof_probs[val_idx] = val_p
        test_probs += model.predict_proba(X_te_scaled) / 5.0
        
        preds = np.argmax(val_p, axis=1)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"C={c} CV Accuracy: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
    
    if mean_acc > best_cv_acc:
        best_cv_acc = mean_acc
        best_c = c
        best_oof_probs = oof_probs
        best_test_probs = test_probs
        best_cv_scores = cv_scores

print(f"\nBest SVM C is {best_c} with CV Accuracy: {best_cv_acc:.5f}")

# Save SVM probabilities
np.save(os.path.join(cache_dir, "oof_probs_svm_linear.npy"), best_oof_probs)
np.save(os.path.join(cache_dir, "test_probs_svm_linear.npy"), best_test_probs)

# Now, load MLP and LGBM out-of-fold probabilities for ensembling
oof_mlp = np.load(os.path.join(cache_dir, "oof_probs_ms_mlp_enhanced_grid.npy"))
test_mlp = np.load(os.path.join(cache_dir, "test_probs_ms_mlp_enhanced_grid.npy"))

oof_lgb = np.load(os.path.join(cache_dir, "oof_probs_lgb_enhanced_grid.npy"))
test_lgb = np.load(os.path.join(cache_dir, "test_probs_lgb_enhanced_grid.npy"))

# Let's perform a grid search for ensembling weights (w_mlp, w_svm, w_lgb)
print("\n--- Optimizing Ensemble Blending Weights ---")
best_ensemble_acc = 0.0
best_weights = (0, 0, 0)

# Grid search with step 0.02
for w_mlp in np.linspace(0, 1, 51):
    for w_svm in np.linspace(0, 1 - w_mlp, int(round((1 - w_mlp) * 50)) + 1):
        w_lgb = 1.0 - w_mlp - w_svm
        if w_lgb < -1e-6:
            continue
        
        blend_probs = w_mlp * oof_mlp + w_svm * best_oof_probs + w_lgb * oof_lgb
        preds = np.argmax(blend_probs, axis=1)
        acc = accuracy_score(y_train, preds)
        
        if acc > best_ensemble_acc:
            best_ensemble_acc = acc
            best_weights = (w_mlp, w_svm, w_lgb)

w_mlp, w_svm, w_lgb = best_weights
print(f"Optimal weights: MLP={w_mlp:.3f}, SVM={w_svm:.3f}, LGB={w_lgb:.3f}")
print(f"Ensemble OOF Accuracy: {best_ensemble_acc:.6f}")

# Calculate CV scores per fold with these weights
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_probs = (
        w_mlp * oof_mlp[val_idx] +
        w_svm * best_oof_probs[val_idx] +
        w_lgb * oof_lgb[val_idx]
    )
    fold_preds = np.argmax(fold_probs, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.5f}")

ensemble_cv_mean = np.mean(blend_cv_scores)
ensemble_cv_std = np.std(blend_cv_scores)
print(f"Ensemble CV Accuracy: {ensemble_cv_mean:.5f} (std: {ensemble_cv_std:.5f})")

# Save submission
sub_path = "agents/gemini/submissions/submission_ensemble_mlp_svm_lgb.csv"
final_blend_probs = w_mlp * test_mlp + w_svm * best_test_probs + w_lgb * test_lgb
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
    "cv_mean": ensemble_cv_mean,
    "cv_std": ensemble_cv_std,
    "weights": {
        "mlp": w_mlp,
        "svm": w_svm,
        "lgb": w_lgb
    },
    "best_svm_c": best_c,
    "svm_cv_mean": best_cv_acc,
    "submission_file": sub_path
}
with open(os.path.join(cache_dir, "results_ensemble_mlp_svm_lgb.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved ensemble results to agents/gemini/workspace/results_ensemble_mlp_svm_lgb.json")
