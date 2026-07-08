import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import lightgbm as lgb
from sklearn.svm import SVC

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
X_train = np.load(os.path.join(cache_dir, "X_train_norm.npy"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train, y_train))

X_tr, y_tr = X_train[train_idx], y_train[train_idx]
X_va, y_va = X_train[val_idx], y_train[val_idx]

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)

pca = PCA(n_components=512, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_scaled)
X_va_pca = pca.transform(X_va_scaled)

seeds = [42, 100, 2026, 777, 999]

# --- 1. Evaluate baseline PCA-MLP (512, 256) (5 seeds) ---
print("Evaluating Baseline PCA-MLP (512, 256)...")
t0 = time.time()
p_va_base_pca = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_pca, y_tr)
    p_va_base_pca += model.predict_proba(X_va_pca) / len(seeds)
acc_base_pca = accuracy_score(y_va, np.argmax(p_va_base_pca, axis=1))
print(f"Baseline PCA-MLP Fold 0 Accuracy: {acc_base_pca:.5f} in {time.time() - t0:.2f}s")

# --- 2. Evaluate new PCA-MLP (256, 256, 128) (5 seeds) ---
print("\nEvaluating New PCA-MLP (256, 256, 128)...")
t0 = time.time()
p_va_new_pca = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(256, 256, 128), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_pca, y_tr)
    p_va_new_pca += model.predict_proba(X_va_pca) / len(seeds)
acc_new_pca = accuracy_score(y_va, np.argmax(p_va_new_pca, axis=1))
print(f"New PCA-MLP Fold 0 Accuracy: {acc_new_pca:.5f} in {time.time() - t0:.2f}s")

# --- 3. Evaluate baseline Raw-MLP (512, 256) (5 seeds) ---
print("\nEvaluating Baseline Raw-MLP (512, 256)...")
t0 = time.time()
p_va_base_raw = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_scaled, y_tr)
    p_va_base_raw += model.predict_proba(X_va_scaled) / len(seeds)
acc_base_raw = accuracy_score(y_va, np.argmax(p_va_base_raw, axis=1))
print(f"Baseline Raw-MLP Fold 0 Accuracy: {acc_base_raw:.5f} in {time.time() - t0:.2f}s")

# --- 4. Evaluate new Raw-MLP (512, 256, 128) (5 seeds) ---
print("\nEvaluating New Raw-MLP (512, 256, 128)...")
t0 = time.time()
p_va_new_raw = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256, 128), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_scaled, y_tr)
    p_va_new_raw += model.predict_proba(X_va_scaled) / len(seeds)
acc_new_raw = accuracy_score(y_va, np.argmax(p_va_new_raw, axis=1))
print(f"New Raw-MLP Fold 0 Accuracy: {acc_new_raw:.5f} in {time.time() - t0:.2f}s")

# --- 5. Evaluate blends on Fold 0 ---
print("\nEvaluating Blend Combinations on Fold 0...")
# Load PCA-SVM and LGBM on Fold 0
print("Training SVC and LightGBM on Fold 0...")
model_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
model_svm.fit(X_tr_pca, y_tr)
p_va_svm = model_svm.predict_proba(X_va_pca)

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
model_lgb = lgb.LGBMClassifier(**lgb_params)
model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
p_va_lgb = model_lgb.predict_proba(X_va)

# Baseline 4-model blend
models_base = [p_va_base_pca, p_va_svm, p_va_base_raw, p_va_lgb]
# New 6-model blend
models_new = [p_va_base_pca, p_va_new_pca, p_va_base_raw, p_va_new_raw, p_va_svm, p_va_lgb]

def optimize_blend(model_probs, y):
    num_models = len(model_probs)
    def objective(w):
        w = np.array(w)
        w = w / np.sum(w)
        blend = np.zeros_like(model_probs[0])
        for i in range(num_models):
            blend += w[i] * model_probs[i]
        preds = np.argmax(blend, axis=1)
        return -accuracy_score(y, preds)
    
    best_acc = 0.0
    best_w = None
    for start in range(num_models):
        init = np.zeros(num_models)
        init[start] = 1.0
        res = minimize(objective, init, method='COBYLA', constraints={'type': 'ineq', 'fun': lambda w: w})
        opt_w = res.x / np.sum(res.x)
        acc = -objective(opt_w)
        if acc > best_acc:
            best_acc = acc
            best_w = opt_w
    return best_acc, best_w

acc_base_blend, w_base = optimize_blend(models_base, y_va)
acc_new_blend, w_new = optimize_blend(models_new, y_va)

print(f"\nBaseline 4-model blend Fold 0 Accuracy: {acc_base_blend:.5f}")
print(f"New 6-model blend Fold 0 Accuracy: {acc_new_blend:.5f}")
delta = acc_new_blend - acc_base_blend
print(f"Delta Blend Accuracy: {delta:+.5f}")
