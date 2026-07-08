"""exp_019 full CV: 5-seed MLP averaging blended with LGBM at w_lgb=0.4 on
grid features, proper 3-fold StratifiedKFold (not single holdout). Reuses
exp_017's saved oof_lgb_exp017.npy (identical StratifiedKFold(3, shuffle=True,
random_state=42) split) instead of refitting LGBM, since only the MLP
component changes -- saves ~1350s of redundant LGBM fitting.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
W_LGB = 0.4
SEEDS = [0, 1, 2, 3, 4]

train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
X = train[feat_cols].values
le = LabelEncoder()
y = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(np.unique(y))
classes = np.arange(n_classes)

oof_lgb = np.load(DATA_DIR + "oof_lgb_exp017.npy")
oof_mlp_single = np.load(DATA_DIR + "oof_mlp_exp017.npy")
y_saved = np.load(DATA_DIR + "y_exp017.npy")
assert np.array_equal(y, y_saved), "label encoding mismatch vs exp_017 saved arrays"

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_mlp_seeds = {s: np.zeros((len(y), n_classes)) for s in SEEDS}

t_start = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

    for seed in SEEDS:
        t0 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
            early_stopping=True, random_state=seed,
        )
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xva_s)
        full_proba = np.zeros((len(va_idx), n_classes))
        full_proba[:, clf.classes_] = proba
        oof_mlp_seeds[seed][va_idx] = full_proba
        acc = accuracy_score(yva, np.argmax(full_proba, axis=1))
        print(f"fold {fold} seed {seed}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

oof_mlp_avg = np.mean([oof_mlp_seeds[s] for s in SEEDS], axis=0)

acc_mlp_single = accuracy_score(y, np.argmax(oof_mlp_single, axis=1))
acc_mlp_avg = accuracy_score(y, np.argmax(oof_mlp_avg, axis=1))

blend_single = W_LGB * oof_lgb + (1 - W_LGB) * oof_mlp_single
blend_avg = W_LGB * oof_lgb + (1 - W_LGB) * oof_mlp_avg
acc_blend_single = accuracy_score(y, np.argmax(blend_single, axis=1))
acc_blend_avg = accuracy_score(y, np.argmax(blend_avg, axis=1))

print("\n=== summary ===")
print(f"MLP single-seed OOF acc: {acc_mlp_single:.4f}")
print(f"MLP 5-seed-avg OOF acc:  {acc_mlp_avg:.4f}")
print(f"blend (single-seed MLP, w_lgb={W_LGB}) OOF acc: {acc_blend_single:.4f}")
print(f"blend (5-seed-avg MLP, w_lgb={W_LGB}) OOF acc:  {acc_blend_avg:.4f}")
print(f"delta (5-seed blend - single-seed blend): {acc_blend_avg - acc_blend_single:+.4f}")
print(f"delta (5-seed blend - exp_017 best 0.9476): {acc_blend_avg - 0.9476:+.4f}")
print(f"total time: {time.time()-t_start:.0f}s")

np.save(DATA_DIR + "oof_mlp5seed_exp019.npy", oof_mlp_avg)
