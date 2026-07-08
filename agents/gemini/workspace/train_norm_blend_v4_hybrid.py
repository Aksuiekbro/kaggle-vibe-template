import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# 1. Load features for LightGBM (concatenated features)
X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_test_base = np.load(os.path.join(cache_dir, "X_test_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))
X_test_pitch = np.load(os.path.join(cache_dir, "X_test_pitch_only.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_lgb = np.concatenate([X_train_base, X_train_pitch], axis=1)
X_test_lgb = np.concatenate([X_test_base, X_test_pitch], axis=1)

print(f"LGBM features shape: Train {X_train_lgb.shape}, Test {X_test_lgb.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

oof_lgb = np.zeros((len(X_train_lgb), 82))
test_lgb = np.zeros((len(X_test_lgb), 82))

# Train LightGBM fold by fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgb, y_train)):
    t_start = time.time()
    print(f"--- Training LGBM Fold {fold} ---")
    X_tr, y_tr = X_train_lgb[train_idx], y_train[train_idx]
    X_va, y_va = X_train_lgb[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    val_lgb_p = model_lgb.predict_proba(X_va)
    oof_lgb[val_idx] = val_lgb_p
    test_lgb += model_lgb.predict_proba(X_test_lgb) / 5.0
    acc = accuracy_score(y_va, np.argmax(val_lgb_p, axis=1))
    print(f"Fold {fold} LGBM Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")

# Save LGBM predictions
np.save(os.path.join(cache_dir, "oof_probs_lgb_pitch.npy"), oof_lgb)
np.save(os.path.join(cache_dir, "test_probs_lgb_pitch.npy"), test_lgb)

lgb_oof_acc = accuracy_score(y_train, np.argmax(oof_lgb, axis=1))
print(f"\nLGBM OOF Accuracy with Pitch Features: {lgb_oof_acc:.6f}")

# 2. Load cached predictions for the MLP/SVM models (trained on baseline features)
print("\nLoading baseline models OOF and test predictions...")
oof_mlp_pca_512 = np.load(os.path.join(cache_dir, "oof_probs_mlp_pca_norm.npy"))
test_mlp_pca_512 = np.load(os.path.join(cache_dir, "test_probs_mlp_pca_norm.npy"))

oof_mlp_pca_256 = np.load(os.path.join(cache_dir, "oof_probs_mlp_pca_256_v3.npy"))
test_mlp_pca_256 = np.load(os.path.join(cache_dir, "test_probs_mlp_pca_256_v3.npy"))

oof_mlp_raw_512_256 = np.load(os.path.join(cache_dir, "oof_probs_mlp_raw_norm.npy"))
test_mlp_raw_512_256 = np.load(os.path.join(cache_dir, "test_probs_mlp_raw_norm.npy"))

oof_mlp_raw_512_128 = np.load(os.path.join(cache_dir, "oof_probs_mlp_raw_512_v3.npy"))
test_mlp_raw_512_128 = np.load(os.path.join(cache_dir, "test_probs_mlp_raw_512_v3.npy"))

oof_svm_pca = np.load(os.path.join(cache_dir, "oof_probs_svm_pca_norm.npy"))
test_svm_pca = np.load(os.path.join(cache_dir, "test_probs_svm_pca_norm.npy"))

# Define full dictionary of models to blend (substituting new lgb)
models = {
    "pca_mlp_512_256": oof_mlp_pca_512,
    "pca_mlp_256_256_128": oof_mlp_pca_256,
    "raw_mlp_512_256": oof_mlp_raw_512_256,
    "raw_mlp_512_256_128": oof_mlp_raw_512_128,
    "svm_pca": oof_svm_pca,
    "lgb_pitch": oof_lgb
}

test_probs_dict = {
    "pca_mlp_512_256": test_mlp_pca_512,
    "pca_mlp_256_256_128": test_mlp_pca_256,
    "raw_mlp_512_256": test_mlp_raw_512_256,
    "raw_mlp_512_256_128": test_mlp_raw_512_128,
    "svm_pca": test_svm_pca,
    "lgb_pitch": test_lgb
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
    print(f"Weight for {name:<22}: {weight:.4f}")
print(f"Blended OOF Accuracy: {best_acc:.6f}")

# Calculate CV scores per fold with these weights
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgb, y_train)):
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
sub_path = "agents/gemini/submissions/submission_norm_blend_v4_hybrid.csv"
final_probs = np.zeros((len(df_test), 82))
for i, name in enumerate(model_names):
    final_probs += best_weights[i] * test_probs_dict[name]

final_preds = np.argmax(final_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved hybrid ensemble submission to {sub_path}")

results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": mean_cv,
    "cv_std": std_cv,
    "weights": {name: float(w) for name, w in zip(model_names, best_weights)},
    "submission_file": sub_path
}
results_path = os.path.join(cache_dir, "results_norm_blend_v4_hybrid.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
