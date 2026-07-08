import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

architectures = [
    (1024, 512),
    (512, 256),  # baseline
    (256, 128),
    (512, 512),
    (256, 256, 128),
    (512, 256, 128)
]

print("--- Testing PCA-MLP Architectures ---")
for arch in architectures:
    t0 = time.time()
    model = MLPClassifier(
        random_state=42,
        hidden_layer_sizes=arch,
        alpha=0.02,
        batch_size=128,
        learning_rate_init=0.002,
        max_iter=130,
        early_stopping=True
    )
    model.fit(X_tr_pca, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va_pca))
    print(f"PCA-MLP {str(arch):<16} | Accuracy: {acc:.5f} | Time: {time.time() - t0:.2f}s")

print("\n--- Testing Raw-MLP Architectures ---")
for arch in architectures:
    t0 = time.time()
    model = MLPClassifier(
        random_state=42,
        hidden_layer_sizes=arch,
        alpha=0.02,
        batch_size=128,
        learning_rate_init=0.002,
        max_iter=130,
        early_stopping=True
    )
    model.fit(X_tr_scaled, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va_scaled))
    print(f"Raw-MLP {str(arch):<16} | Accuracy: {acc:.5f} | Time: {time.time() - t0:.2f}s")
