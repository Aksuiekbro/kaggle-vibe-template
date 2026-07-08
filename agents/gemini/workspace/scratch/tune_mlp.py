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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = [
    {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "lr": 0.005},
    {"hidden_layer_sizes": (512, 256), "alpha": 0.01, "lr": 0.005},
    {"hidden_layer_sizes": (256, 128), "alpha": 0.05, "lr": 0.005},
    {"hidden_layer_sizes": (256, 128), "alpha": 0.01, "lr": 0.002},
    {"hidden_layer_sizes": (512, 256, 128), "alpha": 0.01, "lr": 0.005},
]

for i, cfg in enumerate(configs):
    cv_scores = []
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        
        model = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation='relu',
            solver='adam',
            alpha=cfg["alpha"],
            batch_size=128,
            learning_rate_init=cfg["lr"],
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8
        )
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        cv_scores.append(acc)
        
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"Config {i}: Arch={cfg['hidden_layer_sizes']}, alpha={cfg['alpha']}, lr={cfg['lr']} -> Mean CV: {mean_acc:.5f} (std: {std_acc:.5f}) in {time.time() - t0:.1f}s")
