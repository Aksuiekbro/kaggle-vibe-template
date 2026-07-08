import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train = np.hstack([X_train_exp003, X_train_grid])
print(f"Loaded features shape: {X_train.shape}")

# Use 2 folds for probe speed
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

configs = [
    # MLPs
    {"name": "MLP_base (512, 256) a=0.02", "model": MLPClassifier(hidden_layer_sizes=(512, 256), alpha=0.02, learning_rate_init=0.002, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    {"name": "MLP_wider (1024, 512) a=0.02", "model": MLPClassifier(hidden_layer_sizes=(1024, 512), alpha=0.02, learning_rate_init=0.002, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    {"name": "MLP_deeper (512, 256, 128) a=0.02", "model": MLPClassifier(hidden_layer_sizes=(512, 256, 128), alpha=0.02, learning_rate_init=0.002, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    {"name": "MLP_deep_wide (1024, 512, 256) a=0.02", "model": MLPClassifier(hidden_layer_sizes=(1024, 512, 256), alpha=0.02, learning_rate_init=0.002, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    {"name": "MLP_deep_wide (1024, 512, 256) a=0.05", "model": MLPClassifier(hidden_layer_sizes=(1024, 512, 256), alpha=0.05, learning_rate_init=0.002, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    {"name": "MLP_deep_wide (1024, 512, 256) a=0.05 lr=0.001", "model": MLPClassifier(hidden_layer_sizes=(1024, 512, 256), alpha=0.05, learning_rate_init=0.001, max_iter=130, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42)},
    
    # KNNs
    {"name": "KNN (k=3)", "model": KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)},
    {"name": "KNN (k=5)", "model": KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)},
    {"name": "KNN (k=7)", "model": KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)},
    
    # SVMs
    {"name": "SVM_linear C=1.0", "model": SVC(kernel='linear', C=1.0, probability=True, random_state=42)},
    {"name": "SVM_rbf C=1.0", "model": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)},
    {"name": "SVM_rbf C=10.0", "model": SVC(kernel='rbf', C=10.0, probability=True, random_state=42)}
]

for cfg in configs:
    name = cfg["name"]
    model = cfg["model"]
    t0 = time.time()
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        # Scale features for models that require scaling
        if "KNN" in name or "SVM" in name or "MLP" in name:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
            
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"{name:<40} -> Mean 2-Fold CV: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
