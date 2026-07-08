import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

cache_dir = "agents/gemini/workspace"
X_train_base = np.load(os.path.join(cache_dir, "X_train_norm.npy"))
X_test_base = np.load(os.path.join(cache_dir, "X_test_norm.npy"))
X_train_pitch = np.load(os.path.join(cache_dir, "X_train_pitch_only.npy"))
X_test_pitch = np.load(os.path.join(cache_dir, "X_test_pitch_only.npy"))

X_train = np.concatenate([X_train_base, X_train_pitch], axis=1)
X_test = np.concatenate([X_test_base, X_test_pitch], axis=1)

print(f"Features shape: Train {X_train.shape}, Test {X_test.shape}")

# Adversarial validation set
Xa = np.vstack([X_train, X_test])
ya = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(ya))

for tr_idx, va_idx in skf.split(Xa, ya):
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    clf.fit(Xa[tr_idx], ya[tr_idx])
    oof[va_idx] = clf.predict_proba(Xa[va_idx])[:, 1]

auc = roc_auc_score(ya, oof)
print(f"Concatenated Normalized Features Adversarial AUC: {auc:.4f}")

# Train full model to find the features causing covariate shift
clf_full = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1)
clf_full.fit(Xa, ya)
importances = clf_full.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
print("\nTop 15 adversarial features index and importances:")
for idx in top_idx:
    print(f"  Index {idx}: importance = {importances[idx]}")
