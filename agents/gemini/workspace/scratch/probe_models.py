import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train = np.hstack([X_train_exp003, X_train_grid])
print(f"Loaded features shape: {X_train.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    "SVM_linear_C=0.05": SVC(kernel='linear', C=0.05, probability=True, random_state=42),
    "SVM_rbf_C=1.0": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "SVM_rbf_C=5.0": SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
    "SVM_rbf_C=10.0": SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
    "SVM_rbf_C=50.0": SVC(kernel='rbf', C=50.0, probability=True, random_state=42),
}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    t0 = time.time()
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]
        
        fold_t0 = time.time()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        print(f"Fold {fold} Acc: {acc:.5f} in {time.time() - fold_t0:.1f}s")
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"{name} -> Mean 5-Fold CV: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
