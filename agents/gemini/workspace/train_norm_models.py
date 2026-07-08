import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from scipy.optimize import minimize

cache_dir = "agents/gemini/workspace"
data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

X_train = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_test = np.load(os.path.join(cache_dir, "X_test_norm.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

print(f"Loaded normalized features shape: Train {X_train.shape}, Test {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
seeds = [42, 100, 2026, 777, 999]

# Define base MLP parameters
mlp_params = {
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

# Define LightGBM parameters
lgb_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'learning_rate': 0.0446,
    'n_estimators': 160,
    'max_depth': 3,
    'num_leaves': 24,
    'min_child_samples': 6,
    'subsample': 0.985,
    'colsample_bytree': 0.545,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Initialize OOF and test probability arrays
oof_mlp_pca = np.zeros((len(X_train), 82))
test_mlp_pca = np.zeros((len(X_test), 82))

oof_svm_pca = np.zeros((len(X_train), 82))
test_svm_pca = np.zeros((len(X_test), 82))

oof_mlp_raw = np.zeros((len(X_train), 82))
test_mlp_raw = np.zeros((len(X_test), 82))

oof_lgb = np.zeros((len(X_train), 82))
test_lgb = np.zeros((len(X_test), 82))

# Train models fold by fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Training Fold {fold} ---")
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=512, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_va_pca = pca.transform(X_va_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    
    # 1. Train PCA-MLP (5 seeds)
    t_start = time.time()
    fold_val_p = np.zeros((len(val_idx), 82))
    fold_te_p = np.zeros((len(X_test), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params)
        model.fit(X_tr_pca, y_tr)
        fold_val_p += model.predict_proba(X_va_pca) / len(seeds)
        fold_te_p += model.predict_proba(X_te_pca) / len(seeds)
    oof_mlp_pca[val_idx] = fold_val_p
    test_mlp_pca += fold_te_p / 5.0
    acc_mlp_pca = accuracy_score(y_va, np.argmax(fold_val_p, axis=1))
    print(f"Fold {fold} PCA-MLP (5 seeds) Accuracy: {acc_mlp_pca:.5f} in {time.time() - t_start:.2f}s")
    
    # 2. Train PCA-SVM (linear, C=0.05)
    t_start = time.time()
    model_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
    model_svm.fit(X_tr_pca, y_tr)
    val_svm_p = model_svm.predict_proba(X_va_pca)
    oof_svm_pca[val_idx] = val_svm_p
    test_svm_pca += model_svm.predict_proba(X_te_pca) / 5.0
    acc_svm_pca = accuracy_score(y_va, np.argmax(val_svm_p, axis=1))
    print(f"Fold {fold} PCA-SVM Accuracy: {acc_svm_pca:.5f} in {time.time() - t_start:.2f}s")
    
    # 3. Train Raw MLP (5 seeds, no PCA)
    t_start = time.time()
    fold_val_p_raw = np.zeros((len(val_idx), 82))
    fold_te_p_raw = np.zeros((len(X_test), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params)
        model.fit(X_tr_scaled, y_tr)
        fold_val_p_raw += model.predict_proba(X_va_scaled) / len(seeds)
        fold_te_p_raw += model.predict_proba(X_te_scaled) / len(seeds)
    oof_mlp_raw[val_idx] = fold_val_p_raw
    test_mlp_raw += fold_te_p_raw / 5.0
    acc_mlp_raw = accuracy_score(y_va, np.argmax(fold_val_p_raw, axis=1))
    print(f"Fold {fold} Raw-MLP (5 seeds) Accuracy: {acc_mlp_raw:.5f} in {time.time() - t_start:.2f}s")
    
    # 4. Train LightGBM Classifier
    t_start = time.time()
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    val_lgb_p = model_lgb.predict_proba(X_va)
    oof_lgb[val_idx] = val_lgb_p
    test_lgb += model_lgb.predict_proba(X_test) / 5.0
    acc_lgb = accuracy_score(y_va, np.argmax(val_lgb_p, axis=1))
    print(f"Fold {fold} LightGBM Accuracy: {acc_lgb:.5f} in {time.time() - t_start:.2f}s")

# Print OOF accuracies of individual models
print("\n--- OOF Accuracies ---")
print(f"PCA-MLP (5 seeds) OOF Accuracy: {accuracy_score(y_train, np.argmax(oof_mlp_pca, axis=1)):.6f}")
print(f"PCA-SVM OOF Accuracy: {accuracy_score(y_train, np.argmax(oof_svm_pca, axis=1)):.6f}")
print(f"Raw MLP (5 seeds) OOF Accuracy: {accuracy_score(y_train, np.argmax(oof_mlp_raw, axis=1)):.6f}")
print(f"LightGBM OOF Accuracy: {accuracy_score(y_train, np.argmax(oof_lgb, axis=1)):.6f}")

# Save model OOF/test probabilities
np.save(os.path.join(cache_dir, "oof_probs_mlp_pca_norm.npy"), oof_mlp_pca)
np.save(os.path.join(cache_dir, "test_probs_mlp_pca_norm.npy"), test_mlp_pca)

np.save(os.path.join(cache_dir, "oof_probs_svm_pca_norm.npy"), oof_svm_pca)
np.save(os.path.join(cache_dir, "test_probs_svm_pca_norm.npy"), test_svm_pca)

np.save(os.path.join(cache_dir, "oof_probs_mlp_raw_norm.npy"), oof_mlp_raw)
np.save(os.path.join(cache_dir, "test_probs_mlp_raw_norm.npy"), test_mlp_raw)

np.save(os.path.join(cache_dir, "oof_probs_lgb_norm.npy"), oof_lgb)
np.save(os.path.join(cache_dir, "test_probs_lgb_norm.npy"), test_lgb)

# Optimize blend weights
models = {
    "mlp_pca_norm": oof_mlp_pca,
    "svm_pca_norm": oof_svm_pca,
    "mlp_raw_norm": oof_mlp_raw,
    "lgb_norm": oof_lgb
}

test_probs_dict = {
    "mlp_pca_norm": test_mlp_pca,
    "svm_pca_norm": test_svm_pca,
    "mlp_raw_norm": test_mlp_raw,
    "lgb_norm": test_lgb
}

model_names = list(models.keys())
num_models = len(model_names)

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
        options={'maxiter': 500}
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
    print(f"Weight for {name}: {weight:.4f}")
print(f"Blended OOF Accuracy: {best_acc:.6f}")

# Calculate CV scores per fold with these weights
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
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

# Save final submission
os.makedirs("agents/gemini/submissions", exist_ok=True)
sub_path = "agents/gemini/submissions/submission_norm_blend.csv"
final_probs = np.zeros((len(df_test), 82))
for i, name in enumerate(model_names):
    final_probs += best_weights[i] * test_probs_dict[name]

final_preds = np.argmax(final_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": mean_cv,
    "cv_std": std_cv,
    "weights": {name: float(w) for name, w in zip(model_names, best_weights)},
    "submission_file": sub_path
}
results_path = os.path.join(cache_dir, "results_norm_blend.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
