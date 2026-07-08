import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import Parallel, delayed

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train = np.hstack([X_train_exp003, X_train_grid])
print(f"Loaded features shape: {X_train.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_fold(fold, train_idx, val_idx, model_type, params):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Preprocessing
    if model_type != "HGB":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        
        if "pca" in params:
            pca = PCA(n_components=params["pca"], random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_va = pca.transform(X_va)
            
    # Model definition
    if model_type == "SVM":
        model = SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            probability=False,
            random_state=42
        )
    elif model_type == "HGB":
        model = HistGradientBoostingClassifier(
            learning_rate=params.get("lr", 0.05),
            max_iter=params.get("max_iter", 100),
            max_depth=params.get("max_depth", 4),
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        )
        
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    return acc

configs = [
    {"name": "SVM_linear_C=0.05", "model_type": "SVM", "params": {"kernel": "linear", "C": 0.05}},
    {"name": "SVM_rbf_C=1.0", "model_type": "SVM", "params": {"kernel": "rbf", "C": 1.0}},
    {"name": "SVM_rbf_C=5.0", "model_type": "SVM", "params": {"kernel": "rbf", "C": 5.0}},
    {"name": "SVM_rbf_C=10.0", "model_type": "SVM", "params": {"kernel": "rbf", "C": 10.0}},
    {"name": "SVM_rbf_C=10.0_PCA=128", "model_type": "SVM", "params": {"kernel": "rbf", "C": 10.0, "pca": 128}},
    {"name": "HGB_base", "model_type": "HGB", "params": {"lr": 0.05, "max_iter": 100, "max_depth": 4}},
]

for cfg in configs:
    name = cfg["name"]
    model_type = cfg["model_type"]
    params = cfg["params"]
    
    print(f"\nEvaluating {name} in parallel across 5 folds...")
    t0 = time.time()
    
    cv_scores = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(fold, train_idx, val_idx, model_type, params)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))
    )
    
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"{name} -> Mean 5-Fold CV: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
