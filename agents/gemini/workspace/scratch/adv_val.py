import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

cache_dir = "agents/gemini/workspace"
X_train = np.load(os.path.join(cache_dir, "X_train.npy"))
X_test = np.load(os.path.join(cache_dir, "X_test.npy"))

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Adversarial validation
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
print(f"Gemini baseline features Adversarial AUC: {auc:.4f}")

# Train a full model to get feature importances
clf_full = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1)
clf_full.fit(Xa, ya)
importances = clf_full.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
print("Top 15 adversarial features indices and importances:")
for idx in top_idx:
    print(f"  Index {idx}: {importances[idx]}")
