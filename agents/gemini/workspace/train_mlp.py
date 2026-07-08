import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Load cached combined clean features
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train_exp003.npy")
X_test_path = os.path.join(cache_dir, "X_test_exp003.npy")
y_train_path = os.path.join(cache_dir, "y_train.npy")

if not os.path.exists(X_train_path) or not os.path.exists(X_test_path) or not os.path.exists(y_train_path):
    raise FileNotFoundError("Combined clean features not found.")

X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
print(f"Loaded features shapes: Train {X_train.shape}, Test {X_test.shape}")

# Stratified 5-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = np.zeros((len(X_train), 82))
test_probs = np.zeros((len(X_test), 82))
cv_scores = []

# Config 3 parameters: (256, 128) layers, alpha=0.01, lr=0.002, early stopping
params = {
    'hidden_layer_sizes': (256, 128),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'batch_size': 128,
    'learning_rate_init': 0.002,
    'max_iter': 100,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 8
}

print("Training MLPClassifier with 5-fold CV...")
t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test)
    
    model = MLPClassifier(**params)
    fold_t0 = time.time()
    model.fit(X_tr_scaled, y_tr)
    
    val_probs = model.predict_proba(X_va_scaled)
    oof_probs[val_idx] = val_probs
    
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f} in {time.time() - fold_t0:.2f}s (epochs: {model.n_iter_})")
    
    test_probs += model.predict_proba(X_te_scaled) / 5.0

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\nFinal CV Accuracy: {cv_mean:.5f} (std: {cv_std:.5f}) in {time.time() - t0:.1f}s")

# Save OOF and test probabilities
np.save(os.path.join(cache_dir, "oof_probs_mlp.npy"), oof_probs)
np.save(os.path.join(cache_dir, "test_probs_mlp.npy"), test_probs)

# Save submission
os.makedirs("agents/gemini/submissions", exist_ok=True)
sub_path = "agents/gemini/submissions/submission_mlp.csv"
final_preds = np.argmax(test_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"Saved MLP submission to {sub_path}")

# Output results in standard JSON format
results = {
    "cv_scores": cv_scores,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "best_params": params,
    "submission_file": sub_path
}
with open(os.path.join(cache_dir, "results_mlp.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved MLP results to agents/gemini/workspace/results_mlp.json")
