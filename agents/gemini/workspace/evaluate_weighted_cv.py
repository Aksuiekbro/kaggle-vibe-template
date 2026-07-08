import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

# Load concatenated features
X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_test_base = np.load(os.path.join(cache_dir, "X_test_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))
X_test_pitch = np.load(os.path.join(cache_dir, "X_test_pitch_only.npy"))

X_train = np.concatenate([X_train_base, X_train_pitch], axis=1)
X_test = np.concatenate([X_test_base, X_test_pitch], axis=1)

print(f"Features shape: Train {X_train.shape}, Test {X_test.shape}")

# Adversarial validation set
Xa = np.vstack([X_train, X_test])
ya = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(ya))
oof_lr = np.zeros(len(ya))

scaler = StandardScaler()
Xa_scaled = scaler.fit_transform(Xa)

for fold, (tr_idx, va_idx) in enumerate(skf.split(Xa, ya)):
    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    clf_lgb.fit(Xa[tr_idx], ya[tr_idx])
    oof_lgb[va_idx] = clf_lgb.predict_proba(Xa[va_idx])[:, 1]
    
    # 2. Logistic Regression (L1 regularized for sparsity & robustness)
    clf_lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
    clf_lr.fit(Xa_scaled[tr_idx], ya[tr_idx])
    oof_lr[va_idx] = clf_lr.predict_proba(Xa_scaled[va_idx])[:, 1]

auc_lgb = roc_auc_score(ya, oof_lgb)
auc_lr = roc_auc_score(ya, oof_lr)
print(f"Adversarial Validation AUC (LightGBM): {auc_lgb:.4f}")
print(f"Adversarial Validation AUC (Logistic Regression L1): {auc_lr:.4f}")

# Let's extract adversarial validation probabilities for the train set
p_train_lgb = oof_lgb[:len(X_train)]
p_train_lr = oof_lr[:len(X_train)]

# We will calculate sample weights using density ratio: w = p / (1 - p)
# To avoid extreme values (e.g. division by zero or very large weights), we will clip p.
p_train_lgb_clipped = np.clip(p_train_lgb, 0.01, 0.99)
weights_lgb = p_train_lgb_clipped / (1.0 - p_train_lgb_clipped)
weights_lgb = weights_lgb / np.sum(weights_lgb) * len(X_train)

p_train_lr_clipped = np.clip(p_train_lr, 0.01, 0.99)
weights_lr = p_train_lr_clipped / (1.0 - p_train_lr_clipped)
weights_lr = weights_lr / np.sum(weights_lr) * len(X_train)

print(f"\nWeights statistics (LGBM): Mean={np.mean(weights_lgb):.4f}, Std={np.std(weights_lgb):.4f}, Min={np.min(weights_lgb):.4f}, Max={np.max(weights_lgb):.4f}")
print(f"Weights statistics (LR): Mean={np.mean(weights_lr):.4f}, Std={np.std(weights_lr):.4f}, Min={np.min(weights_lr):.4f}, Max={np.max(weights_lr):.4f}")

# Now let's load OOF files and evaluate them
# 1. sub_017 OOF files (norm_blend version 1)
# 2. sub_018 OOF files (norm_blend version 3 / v3)
# Let's check which files exist

oof_files_v3 = {
    "pca_mlp_512_256": "oof_probs_mlp_pca_norm.npy",
    "pca_mlp_256_256_128": "oof_probs_mlp_pca_256_v3.npy",
    "raw_mlp_512_256": "oof_probs_mlp_raw_norm.npy",
    "raw_mlp_512_256_128": "oof_probs_mlp_raw_512_v3.npy",
    "svm_pca": "oof_probs_svm_pca_norm.npy",
    "lgb": "oof_probs_lgb_norm.npy"
}

oof_files_v1 = {
    "mlp_pca_norm": "oof_probs_mlp_pca_norm.npy",
    "svm_pca_norm": "oof_probs_svm_pca_norm.npy",
    "mlp_raw_norm": "oof_probs_mlp_raw_norm.npy",
    "lgb_norm": "oof_probs_lgb_norm.npy"
}

# Load the weights from sub_017 results to reconstruct the ensemble OOF
results_v3_path = os.path.join(cache_dir, "results_norm_blend_v3.json")
if os.path.exists(results_v3_path):
    with open(results_v3_path) as f:
        res_v3 = json.load(f)
    print("\nReconstructing sub_018 (norm_blend_v3) OOF...")
    w_dict = res_v3["weights"]
    
    # Reconstruct OOF
    oof_v3 = np.zeros((len(X_train), 82))
    for name, weight in w_dict.items():
        filename = oof_files_v3.get(name)
        if filename and os.path.exists(os.path.join(cache_dir, filename)):
            oof_v3 += weight * np.load(os.path.join(cache_dir, filename))
        else:
            print(f"  Warning: OOF file for {name} ({filename}) not found!")
            
    acc_v3_unweighted = accuracy_score(y_train, np.argmax(oof_v3, axis=1))
    acc_v3_weighted_lgb = np.sum((np.argmax(oof_v3, axis=1) == y_train) * weights_lgb) / np.sum(weights_lgb)
    acc_v3_weighted_lr = np.sum((np.argmax(oof_v3, axis=1) == y_train) * weights_lr) / np.sum(weights_lr)
    
    print(f"sub_018 (LB: 0.82758) - Unweighted CV: {acc_v3_unweighted:.6f}")
    print(f"sub_018 (LB: 0.82758) - Weighted CV (LGBM): {acc_v3_weighted_lgb:.6f}")
    print(f"sub_018 (LB: 0.82758) - Weighted CV (LR): {acc_v3_weighted_lr:.6f}")

# Let's load sub_017 results and reconstruct it
results_v1_path = os.path.join(cache_dir, "results_norm_blend.json")
if os.path.exists(results_v1_path):
    with open(results_v1_path) as f:
        res_v1 = json.load(f)
    print("\nReconstructing sub_017 (norm_blend) OOF...")
    w_dict = res_v1["weights"]
    
    # Reconstruct OOF
    oof_v1 = np.zeros((len(X_train), 82))
    for name, weight in w_dict.items():
        filename = oof_files_v1.get(name)
        if filename and os.path.exists(os.path.join(cache_dir, filename)):
            oof_v1 += weight * np.load(os.path.join(cache_dir, filename))
        else:
            print(f"  Warning: OOF file for {name} ({filename}) not found!")
            
    acc_v1_unweighted = accuracy_score(y_train, np.argmax(oof_v1, axis=1))
    acc_v1_weighted_lgb = np.sum((np.argmax(oof_v1, axis=1) == y_train) * weights_lgb) / np.sum(weights_lgb)
    acc_v1_weighted_lr = np.sum((np.argmax(oof_v1, axis=1) == y_train) * weights_lr) / np.sum(weights_lr)
    
    print(f"sub_017 (LB: 0.85632) - Unweighted CV: {acc_v1_unweighted:.6f}")
    print(f"sub_017 (LB: 0.85632) - Weighted CV (LGBM): {acc_v1_weighted_lgb:.6f}")
    print(f"sub_017 (LB: 0.85632) - Weighted CV (LR): {acc_v1_weighted_lr:.6f}")

# Save weights to disk for future use
np.save(os.path.join(cache_dir, "adv_weights_lgb.npy"), weights_lgb)
np.save(os.path.join(cache_dir, "adv_weights_lr.npy"), weights_lr)
print("\nSaved adversarial weights to disk.")
