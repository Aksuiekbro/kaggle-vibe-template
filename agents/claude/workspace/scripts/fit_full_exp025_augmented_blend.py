"""exp_025 follow-through: fit LGBM + MLP on the FULL train set, augmented
with fold-safe-style noise/time-stretch copies of rare (<10-sample) classes
(exp_018's transform, applied once to the whole train set since there's no
held-out fold to leak into at final-fit time), and predict test with the
w_lgb=0.4 soft-vote blend exp_025's 3-fold CV found best (OOF 0.9524 vs
un-augmented exp_017 blend 0.9476). Writes a submission.csv for the score gate.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import augment_rows, RARE_THRESHOLD, SEED

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
W_LGB = 0.4

t0 = time.time()
train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
test = pd.read_parquet(DATA_DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

rare_rows = train[train["Pitch_ID"].isin(rare_classes)]
rng_master = np.random.default_rng(SEED)
aug_df = augment_rows(rare_rows, rng_master)
print(f"augmented {len(rare_rows)} rare rows -> {len(aug_df)} extra rows ({time.time()-t0:.0f}s)", flush=True)

y_raw = np.concatenate([train["Pitch_ID"].values, aug_df["Pitch_ID"].values])
le = LabelEncoder()
ytr = le.fit_transform(y_raw)
n_classes = len(np.unique(ytr))

Xtr = np.vstack([train[feat_cols].values, aug_df[feat_cols].values])
Xte = test[feat_cols].values
print(f"train rows after augmentation: {len(Xtr)} (was {len(train)})", flush=True)

t1 = time.time()
clf_lgb = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
)
clf_lgb.fit(Xtr, ytr)
proba_lgb = clf_lgb.predict_proba(Xte)
full_proba_lgb = np.zeros((len(Xte), n_classes))
full_proba_lgb[:, clf_lgb.classes_] = proba_lgb
print(f"LGBM fit+predict done ({time.time()-t1:.0f}s)", flush=True)

t2 = time.time()
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
    early_stopping=True, random_state=42,
)
clf_mlp.fit(Xtr_s, ytr)
proba_mlp = clf_mlp.predict_proba(Xte_s)
full_proba_mlp = np.zeros((len(Xte), n_classes))
full_proba_mlp[:, clf_mlp.classes_] = proba_mlp
print(f"MLP fit+predict done ({time.time()-t2:.0f}s)", flush=True)

blend_proba = W_LGB * full_proba_lgb + (1 - W_LGB) * full_proba_mlp
blend_pred_idx = np.argmax(blend_proba, axis=1)
blend_pred = le.inverse_transform(blend_pred_idx)

sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": blend_pred})
out_path = DATA_DIR + "submission_exp025_augmented_blend.csv"
sub.to_csv(out_path, index=False)
print(f"wrote {out_path}, {len(sub)} rows ({time.time()-t0:.0f}s total)")
