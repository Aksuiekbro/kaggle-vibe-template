import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load Gemini features
cache_dir = "agents/gemini/workspace"
X_train_gem = np.load(os.path.join(cache_dir, "X_train.npy"))
X_test_gem = np.load(os.path.join(cache_dir, "X_test.npy"))
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

# Load Claude features
df_train_claude = pd.read_parquet("agents/claude/workspace/kernel_output/train_features.parquet")
df_test_claude = pd.read_parquet("agents/claude/workspace/kernel_output/test_features.parquet")

claude_cols = [c for c in df_train_claude.columns if c not in ("Path", "Pitch_ID")]
X_train_claude = df_train_claude[claude_cols].values
X_test_claude = df_test_claude[claude_cols].values

# Concatenate features
X_train = np.hstack([X_train_gem, X_train_claude])
X_test = np.hstack([X_test_gem, X_test_claude])
feature_names = [f"gemini_{i}" for i in range(X_train_gem.shape[1])] + claude_cols

print(f"Concatenated Train shape: {X_train.shape}")
print(f"Concatenated Test shape: {X_test.shape}")

def evaluate_adv_auc(X_tr, X_te):
    Xa = np.vstack([X_tr, X_te])
    ya = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    # 3-Fold Stratified CV for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(ya))
    for tr_idx, va_idx in skf.split(Xa, ya):
        clf = lgb.LGBMClassifier(
            n_estimators=60,
            learning_rate=0.08,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        clf.fit(Xa[tr_idx], ya[tr_idx])
        oof[va_idx] = clf.predict_proba(Xa[va_idx])[:, 1]
    return roc_auc_score(ya, oof)

initial_auc = evaluate_adv_auc(X_train, X_test)
print(f"Initial concatenated Adversarial AUC: {initial_auc:.4f}")

# Adversarial Feature Selection Loop
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()
active_indices = list(range(X_train.shape[1]))
dropped_features = []

threshold = 0.75
max_drops = 150
drop_chunk = 15

for step in range(max_drops // drop_chunk):
    current_auc = evaluate_adv_auc(X_train_clean, X_test_clean)
    print(f"Step {step}: active features = {len(active_indices)}, Adversarial AUC = {current_auc:.4f}")
    if current_auc <= threshold:
        print("Adversarial AUC is below threshold. Stopping.")
        break
        
    # Fit full model to get importances
    Xa = np.vstack([X_train_clean, X_test_clean])
    ya = np.concatenate([np.zeros(len(X_train_clean)), np.ones(len(X_test_clean))])
    clf = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.08, random_state=42, verbosity=-1, n_jobs=-1)
    clf.fit(Xa, ya)
    importances = clf.feature_importances_
    
    # Get top features to drop
    top_indices_local = np.argsort(importances)[::-1][:drop_chunk]
    top_indices_global = [active_indices[i] for i in top_indices_local]
    
    # Drop them
    for idx in top_indices_global:
        dropped_features.append(feature_names[idx])
        active_indices.remove(idx)
        
    # Rebuild clean arrays
    X_train_clean = X_train[:, active_indices]
    X_test_clean = X_test[:, active_indices]

print(f"Dropped {len(dropped_features)} features.")
print(f"Top 20 dropped features: {dropped_features[:20]}")
print(f"Remaining features count: {X_train_clean.shape[1]}")

# Save clean features
np.save(os.path.join(cache_dir, "X_train_exp003.npy"), X_train_clean)
np.save(os.path.join(cache_dir, "X_test_exp003.npy"), X_test_clean)
print("Saved exp003 features to disk.")
