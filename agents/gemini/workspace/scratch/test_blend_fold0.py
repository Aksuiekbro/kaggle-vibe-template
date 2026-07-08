import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from scipy.optimize import minimize

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

# Models probability dictionaries
val_probs = {}

print("Training Fold 0 models...")

# 1. PCA-MLP (512, 256) (5 seeds)
t0 = time.time()
p_mlp_pca1 = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_pca, y_tr)
    p_mlp_pca1 += model.predict_proba(X_va_pca) / len(seeds)
val_probs["pca_mlp_512_256"] = p_mlp_pca1
print(f"PCA-MLP (512, 256) Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_mlp_pca1, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 2. PCA-MLP (256, 256, 128) (5 seeds)
t0 = time.time()
p_mlp_pca2 = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(256, 256, 128), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_pca, y_tr)
    p_mlp_pca2 += model.predict_proba(X_va_pca) / len(seeds)
val_probs["pca_mlp_256_256_128"] = p_mlp_pca2
print(f"PCA-MLP (256, 256, 128) Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_mlp_pca2, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 3. PCA-SVM (linear, C=0.05)
t0 = time.time()
model_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
model_svm.fit(X_tr_pca, y_tr)
p_svm_pca = model_svm.predict_proba(X_va_pca)
val_probs["pca_svm"] = p_svm_pca
print(f"PCA-SVM Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_svm_pca, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 4. Raw-MLP (512, 256, 128) (5 seeds)
t0 = time.time()
p_mlp_raw1 = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256, 128), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_scaled, y_tr)
    p_mlp_raw1 += model.predict_proba(X_va_scaled) / len(seeds)
val_probs["raw_mlp_512_256_128"] = p_mlp_raw1
print(f"Raw-MLP (512, 256, 128) Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_mlp_raw1, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 5. Raw-MLP (512, 256) (5 seeds)
t0 = time.time()
p_mlp_raw2 = np.zeros((len(val_idx), 82))
for seed in seeds:
    model = MLPClassifier(random_state=seed, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
    model.fit(X_tr_scaled, y_tr)
    p_mlp_raw2 += model.predict_proba(X_va_scaled) / len(seeds)
val_probs["raw_mlp_512_256"] = p_mlp_raw2
print(f"Raw-MLP (512, 256) Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_mlp_raw2, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 6. Raw SVM (linear, C=0.05)
t0 = time.time()
model_raw_svm = SVC(kernel='linear', C=0.05, probability=True, random_state=42)
model_raw_svm.fit(X_tr_scaled, y_tr)
p_raw_svm = model_raw_svm.predict_proba(X_va_scaled)
val_probs["raw_svm"] = p_raw_svm
print(f"Raw SVM Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_raw_svm, axis=1)):.5f} in {time.time() - t0:.2f}s")

# 7. Raw LGBM
t0 = time.time()
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
p_lgb = model_lgb.predict_proba(X_va)
val_probs["lgb"] = p_lgb
print(f"LightGBM Done. Fold 0 Accuracy: {accuracy_score(y_va, np.argmax(p_lgb, axis=1)):.5f} in {time.time() - t0:.2f}s")

# Optimize blend weights on Fold 0
model_names = list(val_probs.keys())
num_models = len(model_names)

def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = np.zeros_like(val_probs[model_names[0]])
    for i, name in enumerate(model_names):
        blend += weights[i] * val_probs[name]
    preds = np.argmax(blend, axis=1)
    return -accuracy_score(y_va, preds)

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
    opt_w = res.x / np.sum(res.x)
    acc = -objective(opt_w)
    if acc > best_acc:
        best_acc = acc
        best_weights = opt_w

print("\n--- Optimized Blending on Fold 0 ---")
for name, weight in zip(model_names, best_weights):
    print(f"Weight for {name:<22}: {weight:.4f}")
print(f"Blended Fold 0 Accuracy: {best_acc:.6f}")
