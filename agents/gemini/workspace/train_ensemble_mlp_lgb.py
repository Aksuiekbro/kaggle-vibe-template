import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

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

# Best params from exp003 LightGBM
best_params = {
    'learning_rate': 0.044647267426869655,
    'n_estimators': 160,
    'max_depth': 3,
    'num_leaves': 24,
    'min_child_samples': 6,
    'subsample': 0.9845881298914211,
    'colsample_bytree': 0.5445651668276759
}

final_params = {
    'objective': 'multiclass',
    'num_class': 82,
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    **best_params
}

print("Training LightGBM on clean features to extract OOF probabilities...")
oof_probs_lgb = np.zeros((len(X_train), 82))
test_probs_lgb = np.zeros((len(X_test), 82))
lgb_cv_scores = []

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**final_params)
    fold_t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(15, verbose=False)]
    )
    
    val_probs = model.predict_proba(X_va)
    oof_probs_lgb[val_idx] = val_probs
    
    preds = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_va, preds)
    lgb_cv_scores.append(acc)
    print(f"LGB Fold {fold} Accuracy: {acc:.4f} in {time.time() - fold_t0:.2f}s")
    
    test_probs_lgb += model.predict_proba(X_test) / 5.0

lgb_cv_mean = np.mean(lgb_cv_scores)
lgb_cv_std = np.std(lgb_cv_scores)
print(f"\nLGB CV Accuracy: {lgb_cv_mean:.5f} (std: {lgb_cv_std:.5f}) in {time.time() - t0:.1f}s")

# Save LGB OOF and test probabilities
np.save(os.path.join(cache_dir, "oof_probs_exp003_lgb.npy"), oof_probs_lgb)
np.save(os.path.join(cache_dir, "test_probs_exp003_lgb.npy"), test_probs_lgb)

# Load MLP OOF and test probabilities
oof_probs_mlp_path = os.path.join(cache_dir, "oof_probs_mlp.npy")
test_probs_mlp_path = os.path.join(cache_dir, "test_probs_mlp.npy")

if not os.path.exists(oof_probs_mlp_path) or not os.path.exists(test_probs_mlp_path):
    raise FileNotFoundError("MLP OOF/test probabilities not found. Run train_mlp.py first.")

oof_probs_mlp = np.load(oof_probs_mlp_path)
test_probs_mlp = np.load(test_probs_mlp_path)

# Let's perform probability ensembling
print("\n--- Finding Best Blending Weight between MLP and LightGBM ---")
best_w = 1.0  # Weight for MLP
best_acc = 0.0

# Search weights for MLP (w) vs LightGBM (1 - w)
for w in np.linspace(0, 1, 101):
    blend_probs = w * oof_probs_mlp + (1 - w) * oof_probs_lgb
    preds = np.argmax(blend_probs, axis=1)
    acc = accuracy_score(y_train, preds)
    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"Best MLP Weight: {best_w:.2f} with Blended OOF Accuracy = {best_acc:.6f}")

# Fold-by-fold accuracy for the best blend weight
blend_cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_probs = best_w * oof_probs_mlp[val_idx] + (1 - best_w) * oof_probs_lgb[val_idx]
    fold_preds = np.argmax(fold_probs, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.4f}")

blend_cv_mean = np.mean(blend_cv_scores)
blend_cv_std = np.std(blend_cv_scores)
print(f"Blended CV Accuracy: {blend_cv_mean:.5f} (std: {blend_cv_std:.5f})")

# Save ensemble submission
os.makedirs("agents/gemini/submissions", exist_ok=True)
sub_path = "agents/gemini/submissions/submission_ensemble_mlp_lgb.csv"

final_blend_probs = best_w * test_probs_mlp + (1 - best_w) * test_probs_lgb
final_preds = np.argmax(final_blend_probs, axis=1)

df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

# Output results in standard JSON format
results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": blend_cv_mean,
    "cv_std": blend_cv_std,
    "best_weight_mlp": best_w,
    "submission_file": sub_path
}
with open(os.path.join(cache_dir, "results_ensemble_mlp_lgb.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved ensemble results to agents/gemini/workspace/results_ensemble_mlp_lgb.json")
