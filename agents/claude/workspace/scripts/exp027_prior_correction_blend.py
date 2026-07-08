"""exp_027 (own): direct test of exp_022's hypothesis -- the LGBM+MLP blend
systematically smooths predictions toward the train class prior, away from
rare classes, which may explain sub_022's CV-up/LB-down drop if the real
test rare-class rate differs from train's. Fix candidate: prior-correct the
blend's predicted probabilities (divide by empirical train class prior, i.e.
scaled-likelihood / Bayes-flat correction) before taking argmax, and compare
OOF accuracy/balanced_accuracy vs the plain blend. Reuses exp_025's saved OOF
arrays (augmented blend) -- no retraining, near-free.
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
W_LGB = 0.4

oof_lgb = np.load(DATA_DIR + "oof_lgb_exp025.npy")
oof_mlp = np.load(DATA_DIR + "oof_mlp_exp025.npy")
y_all = np.load(DATA_DIR + "y_exp025.npy")
n_classes = oof_lgb.shape[1]
classes = np.arange(n_classes)

blend_proba = W_LGB * oof_lgb + (1 - W_LGB) * oof_mlp
plain_pred = classes[np.argmax(blend_proba, axis=1)]
plain_acc = accuracy_score(y_all, plain_pred)
plain_bacc = balanced_accuracy_score(y_all, plain_pred)
print(f"plain blend: acc={plain_acc:.4f} balanced_acc={plain_bacc:.4f}")

train_prior = np.bincount(y_all, minlength=n_classes) / len(y_all)
train_prior = np.clip(train_prior, 1e-6, None)

for alpha in (0.25, 0.5, 0.75, 1.0):
    corrected = blend_proba / (train_prior ** alpha)
    corrected_pred = classes[np.argmax(corrected, axis=1)]
    acc = accuracy_score(y_all, corrected_pred)
    bacc = balanced_accuracy_score(y_all, corrected_pred)
    print(f"prior-correction alpha={alpha}: acc={acc:.4f} (delta {acc-plain_acc:+.4f}) "
          f"balanced_acc={bacc:.4f} (delta {bacc-plain_bacc:+.4f})")

rare_classes_mask = train_prior < (10 / len(y_all))
rare_idx = np.isin(y_all, classes[rare_classes_mask])
print(f"\nrare-class rows in OOF: {rare_idx.sum()}/{len(y_all)}")
plain_rare_acc = accuracy_score(y_all[rare_idx], plain_pred[rare_idx])
print(f"plain blend rare-subset acc: {plain_rare_acc:.4f}")
for alpha in (0.25, 0.5, 0.75, 1.0):
    corrected = blend_proba / (train_prior ** alpha)
    corrected_pred = classes[np.argmax(corrected, axis=1)]
    rare_acc = accuracy_score(y_all[rare_idx], corrected_pred[rare_idx])
    print(f"prior-correction alpha={alpha}: rare-subset acc={rare_acc:.4f} "
          f"(delta {rare_acc-plain_rare_acc:+.4f})")
