import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
X_train_v2 = np.load(os.path.join(cache_dir, "X_train_norm_v2.npy"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train_v2, y_train))

# Define subsets
subsets = {
    "V1_only": X_train_v2[:, :1628],
    "V1_plus_CQT": X_train_v2[:, :1796],
    "V1_plus_CQT_Chroma": X_train_v2[:, :1820],
    "V1_plus_all_new": X_train_v2
}

pca_options = [256, 512, 768, "None"]

results = []

for subset_name, X_data in subsets.items():
    for pca_comp in pca_options:
        # Split fold
        X_tr = X_data[train_idx]
        X_va = X_data[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        # PCA
        if pca_comp != "None":
            pca = PCA(n_components=int(pca_comp), random_state=42)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_va_p = pca.transform(X_va_s)
        else:
            X_tr_p = X_tr_s
            X_va_p = X_va_s
            
        # Model
        model = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=(512, 256),
            alpha=0.02,
            batch_size=128,
            learning_rate_init=0.002,
            max_iter=130,
            early_stopping=True
        )
        
        model.fit(X_tr_p, y_train[train_idx])
        preds = model.predict(X_va_p)
        acc = accuracy_score(y_train[val_idx], preds)
        
        print(f"Subset: {subset_name:<20} | PCA: {str(pca_comp):<5} | Accuracy: {acc:.5f}")
        results.append((subset_name, pca_comp, acc))

best_res = max(results, key=lambda x: x[2])
print(f"\nBest configuration on Fold 0:")
print(f"Subset: {best_res[0]} | PCA: {best_res[1]} | Accuracy: {best_res[2]:.5f}")
