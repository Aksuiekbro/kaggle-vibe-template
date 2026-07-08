import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

cache_dir = "agents/gemini/workspace"
X_train = np.load(os.path.join(cache_dir, "X_train_exp003.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

print(f"Loaded features shapes: Train {X_train.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale features (very important for neural networks)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    
    # Fast MLP config
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.005,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )
    
    fold_t0 = time.time()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"  Fold {fold} accuracy: {acc:.4f} in {time.time() - fold_t0:.2f}s (epochs: {model.n_iter_})")

elapsed = time.time() - t0
print(f"MLP 5-fold CV completed in {elapsed:.2f} seconds.")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")
