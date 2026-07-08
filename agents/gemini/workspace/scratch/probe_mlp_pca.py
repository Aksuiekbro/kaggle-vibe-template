import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train = np.hstack([X_train_exp003, X_train_grid])
print(f"Loaded features shape: {X_train.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_fold(fold, train_idx, val_idx, params):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    
    # Optional PCA
    if "pca" in params:
        pca = PCA(n_components=params["pca"], random_state=42)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)
        
    model = MLPClassifier(
        hidden_layer_sizes=params.get("hidden", (512, 256)),
        alpha=params.get("alpha", 0.02),
        learning_rate_init=0.002,
        max_iter=130,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    return acc

configs = [
    {"name": "MLP_no_PCA_alpha=0.02", "params": {}},
    {"name": "MLP_no_PCA_alpha=0.05", "params": {"alpha": 0.05}},
    {"name": "MLP_PCA=512_alpha=0.02", "params": {"pca": 512}},
    {"name": "MLP_PCA=256_alpha=0.02", "params": {"pca": 256}},
    {"name": "MLP_wider_no_PCA_alpha=0.02", "params": {"hidden": (1024, 512)}},
]

for cfg in configs:
    name = cfg["name"]
    params = cfg["params"]
    
    print(f"\nEvaluating {name} in parallel across 5 folds...")
    t0 = time.time()
    
    cv_scores = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(fold, train_idx, val_idx, params)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))
    )
    
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"{name} -> Mean 5-Fold CV: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
