import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
X_train = np.load(os.path.join(cache_dir, "X_train_norm.npy")) # Use V1 features which were best

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

# Baseline models to compare
print("--- Baseline Models on Fold 0 ---")
# 1. PCA-MLP (seed 42)
model = MLPClassifier(random_state=42, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
model.fit(X_tr_pca, y_tr)
acc = accuracy_score(y_va, model.predict(X_va_pca))
print(f"PCA-MLP (seed 42) Accuracy: {acc:.5f}")

# 2. PCA-SVM (linear, C=0.05)
model = SVC(kernel='linear', C=0.05, random_state=42)
model.fit(X_tr_pca, y_tr)
acc = accuracy_score(y_va, model.predict(X_va_pca))
print(f"PCA-SVM (linear, C=0.05) Accuracy: {acc:.5f}")

# 3. Raw LGBM
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
acc = accuracy_score(y_va, model_lgb.predict(X_va))
print(f"Raw LGBM Accuracy: {acc:.5f}")

print("\n--- Candidate Models on Fold 0 ---")
# 4. PCA-LGBM
model_lgb_pca = lgb.LGBMClassifier(**lgb_params)
model_lgb_pca.fit(X_tr_pca, y_tr, eval_set=[(X_va_pca, y_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
acc = accuracy_score(y_va, model_lgb_pca.predict(X_va_pca))
print(f"PCA-LGBM Accuracy: {acc:.5f}")

# 5. PCA-SVM (RBF, C=1.0)
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_tr_pca, y_tr)
acc = accuracy_score(y_va, model.predict(X_va_pca))
print(f"PCA-SVM (RBF, C=1.0) Accuracy: {acc:.5f}")

# 6. PCA-SVM (linear, tune C)
for c in [0.01, 0.1, 0.5, 1.0]:
    model = SVC(kernel='linear', C=c, random_state=42)
    model.fit(X_tr_pca, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va_pca))
    print(f"PCA-SVM (linear, C={c}) Accuracy: {acc:.5f}")

# 7. Raw SVM (linear, C=0.05)
model = SVC(kernel='linear', C=0.05, random_state=42)
model.fit(X_tr_scaled, y_tr)
acc = accuracy_score(y_va, model.predict(X_va_scaled))
print(f"Raw SVM (linear, C=0.05) Accuracy: {acc:.5f}")

# 8. KNN (metric=cosine)
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    model.fit(X_tr_pca, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va_pca))
    print(f"PCA-KNN (cosine, k={k}) Accuracy: {acc:.5f}")

# 9. KNN (metric=euclidean)
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    model.fit(X_tr_pca, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va_pca))
    print(f"PCA-KNN (euclidean, k={k}) Accuracy: {acc:.5f}")
