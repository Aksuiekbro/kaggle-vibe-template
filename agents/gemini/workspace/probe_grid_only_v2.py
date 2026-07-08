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
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")

df_train = pd.read_csv(train_csv)

# Load base features and pitch features
X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

# Select ONLY grid features (indices 494 to 1627)
X_train_grid = X_train_base[:, 494:]

# Select ONLY robust pitch features: yin_midi (idx 1) and hps_midi (idx 4)
X_train_robust_pitch = X_train_pitch[:, [1, 4]]

# Concatenate robust features
X_train = np.concatenate([X_train_grid, X_train_robust_pitch], axis=1)

print(f"Grid-only features shape: Train {X_train.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
seeds = [42, 100, 2026, 777, 999]

# Define base MLP parameters
mlp_params_pca_512_256 = {
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

mlp_params_pca_256_256_128 = {
    'hidden_layer_sizes': (256, 256, 128),
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

mlp_params_raw_512_256 = {
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

mlp_params_raw_512_256_128 = {
    'hidden_layer_sizes': (512, 256, 128),
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

# Define LightGBM parameters (specifically tuned for robust pitch grid)
lgb_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'n_estimators': 120,
    'max_depth': 3,
    'num_leaves': 20,
    'min_child_samples': 6,
    'subsample': 0.95,
    'colsample_bytree': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# We only train Fold 0 for the probe
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    if fold > 0:
        break
    print(f"\n--- Training Probe Fold {fold} ---")
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    # PCA 512
    pca512 = PCA(n_components=512, random_state=42)
    X_tr_pca512 = pca512.fit_transform(X_tr_scaled)
    X_va_pca512 = pca512.transform(X_va_scaled)
    
    # PCA 256
    pca256 = PCA(n_components=256, random_state=42)
    X_tr_pca256 = pca256.fit_transform(X_tr_scaled)
    X_va_pca256 = pca256.transform(X_va_scaled)
    
    models = {}
    
    # 1. PCA-MLP (512, 256) (5 seeds)
    t_start = time.time()
    fold_val = np.zeros((len(val_idx), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params_pca_512_256)
        model.fit(X_tr_pca512, y_tr)
        fold_val += model.predict_proba(X_va_pca512) / len(seeds)
    models["pca_mlp_512_256"] = fold_val
    acc = accuracy_score(y_va, np.argmax(fold_val, axis=1))
    print(f"PCA-MLP (512, 256) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")
    
    # 2. PCA-MLP (256, 256, 128) (5 seeds)
    t_start = time.time()
    fold_val = np.zeros((len(val_idx), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params_pca_256_256_128)
        model.fit(X_tr_pca256, y_tr)
        fold_val += model.predict_proba(X_va_pca256) / len(seeds)
    models["pca_mlp_256_256_128"] = fold_val
    acc = accuracy_score(y_va, np.argmax(fold_val, axis=1))
    print(f"PCA-MLP (256, 256, 128) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")
    
    # 3. Train Raw MLP (512, 256) (5 seeds)
    t_start = time.time()
    fold_val = np.zeros((len(val_idx), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params_raw_512_256)
        model.fit(X_tr_scaled, y_tr)
        fold_val += model.predict_proba(X_va_scaled) / len(seeds)
    models["raw_mlp_512_256"] = fold_val
    acc = accuracy_score(y_va, np.argmax(fold_val, axis=1))
    print(f"Raw MLP (512, 256) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")
    
    # 4. Train Raw MLP (512, 256, 128) (5 seeds)
    t_start = time.time()
    fold_val = np.zeros((len(val_idx), 82))
    for seed in seeds:
        model = MLPClassifier(random_state=seed, **mlp_params_raw_512_256_128)
        model.fit(X_tr_scaled, y_tr)
        fold_val += model.predict_proba(X_va_scaled) / len(seeds)
    models["raw_mlp_512_256_128"] = fold_val
    acc = accuracy_score(y_va, np.argmax(fold_val, axis=1))
    print(f"Raw MLP (512, 256, 128) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")
    
    # 5. Train PCA-SVM (512 components, C=0.05)
    t_start = time.time()
    model_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
    model_svm.fit(X_tr_pca512, y_tr)
    val_svm_p = model_svm.predict_proba(X_va_pca512)
    models["svm_pca_512"] = val_svm_p
    acc = accuracy_score(y_va, np.argmax(val_svm_p, axis=1))
    print(f"PCA-SVM (512) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")

    # 6. Train PCA-SVM (256 components, C=0.05)
    t_start = time.time()
    model_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
    model_svm.fit(X_tr_pca256, y_tr)
    val_svm_p = model_svm.predict_proba(X_va_pca256)
    models["svm_pca_256"] = val_svm_p
    acc = accuracy_score(y_va, np.argmax(val_svm_p, axis=1))
    print(f"PCA-SVM (256) Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")
    
    # 7. Train LightGBM Classifier (runs on raw, unscaled features)
    t_start = time.time()
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    val_lgb_p = model_lgb.predict_proba(X_va)
    models["lgb"] = val_lgb_p
    acc = accuracy_score(y_va, np.argmax(val_lgb_p, axis=1))
    print(f"LightGBM Accuracy: {acc:.5f} in {time.time() - t_start:.2f}s")

    # Optimize blend weights on Fold 0
    model_names = list(models.keys())
    num_models = len(model_names)
    
    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # normalize
        
        blend = np.zeros_like(models[model_names[0]])
        for i, name in enumerate(model_names):
            blend += weights[i] * models[name]
            
        preds = np.argmax(blend, axis=1)
        acc = accuracy_score(y_va, preds)
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
    
    print("\n--- Optimized Blending (Fold 0) ---")
    for name, weight in zip(model_names, best_weights):
        print(f"Weight for {name:<22}: {weight:.4f}")
    print(f"Blended Fold 0 Accuracy: {best_acc:.6f}")
