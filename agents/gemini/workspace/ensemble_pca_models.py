import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

cache_dir = "agents/gemini/workspace"
y_train = np.load(os.path.join(cache_dir, "y_train.npy"))

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")
df_test = pd.read_csv(test_csv)

# Load MLP probabilities (PCA 512, 5 seeds)
oof_mlp = np.load(os.path.join(cache_dir, "oof_probs_ms_mlp_pca512.npy"))
test_mlp = np.load(os.path.join(cache_dir, "test_probs_ms_mlp_pca512.npy"))

# Load SVM probabilities (PCA 512, linear, C=0.05)
oof_svm = np.load(os.path.join(cache_dir, "oof_probs_svm_linear_c0.05_pca512.npy"))
test_svm = np.load(os.path.join(cache_dir, "test_probs_svm_linear_c0.05_pca512.npy"))

print("MLP OOF Accuracy:", accuracy_score(y_train, np.argmax(oof_mlp, axis=1)))
print("SVM OOF Accuracy:", accuracy_score(y_train, np.argmax(oof_svm, axis=1)))

# Optimize blend weights
best_acc = 0.0
best_w = 1.0

for w in np.linspace(0, 1, 101):
    blend_oof = w * oof_mlp + (1 - w) * oof_svm
    acc = accuracy_score(y_train, np.argmax(blend_oof, axis=1))
    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"\nBest Blend Weight (MLP): {best_w:.2f}")
print(f"Blended OOF Accuracy: {best_acc:.6f}")

# Calculate CV scores per fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
blend_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(oof_mlp, y_train)):
    fold_probs = best_w * oof_mlp[val_idx] + (1 - best_w) * oof_svm[val_idx]
    fold_preds = np.argmax(fold_probs, axis=1)
    fold_acc = accuracy_score(y_train[val_idx], fold_preds)
    blend_cv_scores.append(fold_acc)
    print(f"Fold {fold} Blend Accuracy: {fold_acc:.5f}")

mean_cv = np.mean(blend_cv_scores)
std_cv = np.std(blend_cv_scores)
print(f"\nBlended 5-Fold CV Accuracy: {mean_cv:.5f} (std: {std_cv:.5f})")

# Save blended submissions
sub_path = "agents/gemini/submissions/submission_ensemble_pca_mlp_svm.csv"
final_probs = best_w * test_mlp + (1 - best_w) * test_svm
final_preds = np.argmax(final_probs, axis=1)

df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"\nSaved ensemble submission to {sub_path}")

# Save results
results = {
    "cv_scores": blend_cv_scores,
    "cv_mean": mean_cv,
    "cv_std": std_cv,
    "weights": {
        "mlp_pca512": best_w,
        "svm_pca512": 1 - best_w
    },
    "submission_file": sub_path
}
results_path = os.path.join(cache_dir, "results_ensemble_pca_mlp_svm.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {results_path}")
