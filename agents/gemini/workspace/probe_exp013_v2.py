import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

# Load base features and pitch features
X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train_base, y_train))

# Evaluate PCA-MLP on Fold 0 for V1 (baseline)
print("Evaluating V1 (baseline)...")
scaler1 = StandardScaler()
X_tr_s1 = scaler1.fit_transform(X_train_base[train_idx])
X_va_s1 = scaler1.transform(X_train_base[val_idx])

pca1 = PCA(n_components=512, random_state=42)
X_tr_p1 = pca1.fit_transform(X_tr_s1)
X_va_p1 = pca1.transform(X_va_s1)

t0 = time.time()
model_v1 = MLPClassifier(random_state=42, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
model_v1.fit(X_tr_p1, y_train[train_idx])
preds_v1 = model_v1.predict(X_va_p1)
acc_v1 = accuracy_score(y_train[val_idx], preds_v1)
print(f"V1 Fold 0 PCA-MLP Accuracy: {acc_v1:.5f} in {time.time() - t0:.2f}s")

# Evaluate V4_v2: Scale pitch features, run PCA on base features, and concatenate them
print("\nEvaluating V4_v2 (with pitch features outside PCA)...")
# Scale base
scaler_base = StandardScaler()
X_tr_base_s = scaler_base.fit_transform(X_train_base[train_idx])
X_va_base_s = scaler_base.transform(X_train_base[val_idx])

# PCA on base
pca_base = PCA(n_components=512, random_state=42)
X_tr_base_pca = pca_base.fit_transform(X_tr_base_s)
X_va_base_pca = pca_base.transform(X_va_base_s)

# Scale pitch features
scaler_pitch = StandardScaler()
X_tr_pitch_s = scaler_pitch.fit_transform(X_train_pitch[train_idx])
X_va_pitch_s = scaler_pitch.transform(X_train_pitch[val_idx])

# Concatenate PCA components and scaled pitch features
X_tr_v4 = np.concatenate([X_tr_base_pca, X_tr_pitch_s], axis=1)
X_va_v4 = np.concatenate([X_va_base_pca, X_va_pitch_s], axis=1)

t0 = time.time()
model_v4 = MLPClassifier(random_state=42, hidden_layer_sizes=(512, 256), alpha=0.02, batch_size=128, learning_rate_init=0.002, max_iter=130, early_stopping=True)
model_v4.fit(X_tr_v4, y_train[train_idx])
preds_v4 = model_v4.predict(X_va_v4)
acc_v4 = accuracy_score(y_train[val_idx], preds_v4)
print(f"V4_v2 Fold 0 PCA-MLP Accuracy: {acc_v4:.5f} in {time.time() - t0:.2f}s")

delta = acc_v4 - acc_v1
print(f"\nDelta Accuracy: {delta:+.5f}")
if delta > 0:
    print("Signal confirmed! Feature integration outside PCA improves performance.")
elif delta == 0:
    print("Neutral. No direct change on Fold 0.")
else:
    print("Negative. Still degraded performance on Fold 0.")
