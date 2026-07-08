import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

cache_dir = "agents/gemini/workspace"
X_train_exp003 = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
X_test_exp003 = np.load(os.path.join(cache_dir, "X_test_exp003.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

X_train_grid = np.load(os.path.join(cache_dir, "X_train_enhanced_grid.npy"))
X_test_grid = np.load(os.path.join(cache_dir, "X_test_enhanced_grid.npy"))

# Concatenate features
X_train = np.hstack([X_train_exp003, X_train_grid])
X_test = np.hstack([X_test_exp003, X_test_grid])

print(f"Concatenated features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Configurations to test: (kernel, C, pca_components)
configs = [
    ('linear', 0.05, 512),
    ('linear', 0.1, 512),
    ('linear', 0.5, 512),
    ('linear', 1.0, 512),
    ('rbf', 1.0, 512),
    ('rbf', 5.0, 512),
    ('rbf', 10.0, 512),
]

best_cv_acc = 0.0
best_config = None
best_oof_probs = None
best_test_probs = None
best_cv_scores = []

for kernel, c, n_comp in configs:
    print(f"\nEvaluating SVM(kernel={kernel}, C={c}, PCA={n_comp})...")
    oof_probs = np.zeros((len(X_train), 82))
    test_probs = np.zeros((len(X_test), 82))
    cv_scores = []
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_test)
        
        # PCA
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_va_pca = pca.transform(X_va_scaled)
        X_te_pca = pca.transform(X_te_scaled)
        
        model = SVC(kernel=kernel, C=c, probability=True, random_state=42)
        model.fit(X_tr_pca, y_tr)
        
        val_p = model.predict_proba(X_va_pca)
        oof_probs[val_idx] = val_p
        test_probs += model.predict_proba(X_te_pca) / 5.0
        
        preds = np.argmax(val_p, axis=1)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"SVM(kernel={kernel}, C={c}, PCA={n_comp}) CV Accuracy: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
    
    if mean_acc > best_cv_acc:
        best_cv_acc = mean_acc
        best_config = (kernel, c, n_comp)
        best_oof_probs = oof_probs
        best_test_probs = test_probs
        best_cv_scores = cv_scores

print(f"\nBest SVM config: {best_config} with CV Accuracy: {best_cv_acc:.5f}")

# Save the best SVM probabilities
kernel, c, n_comp = best_config
np.save(os.path.join(cache_dir, f"oof_probs_svm_{kernel}_c{c}_pca{n_comp}.npy"), best_oof_probs)
np.save(os.path.join(cache_dir, f"test_probs_svm_{kernel}_c{c}_pca{n_comp}.npy"), best_test_probs)

# Save results json
results = {
    "cv_scores": best_cv_scores,
    "cv_mean": best_cv_acc,
    "cv_std": float(np.std(best_cv_scores)),
    "config": {
        "kernel": kernel,
        "C": c,
        "n_components": n_comp
    }
}
results_path = os.path.join(cache_dir, f"results_svm_{kernel}_c{c}_pca{n_comp}.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
