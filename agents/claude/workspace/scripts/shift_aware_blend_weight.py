"""exp_011: shift-aware blend-weight search.

PLAN_DRAFT.md hypothesis (from cross-reviewing gemini's sub_021): a blend
optimizer that maximizes plain OOF accuracy will over-weight shift-prone
generic-spectral features even when a shift-robust feature family (grid-
harmonic, exp_005) is available, because it's blind to the measured
train/test shift (exp_003 AUC ~0.998 on generic features).

Method: build 3-fold OOF predict_proba for two LGBM models on the SAME split
(random_state=42) -- one on exp_005's grid-harmonic features (shift-robust,
exp_006 gap -0.0095), one on exp_004's RMS-normalized generic-librosa
features (shift-prone family, adversarial AUC ~0.998 per exp_003). Blend
proba as w*grid + (1-w)*generic for w in a grid [0, 1]. Compare the w chosen
by:
  (a) plain OOF accuracy (what a COBYLA-style optimizer blind to shift would do)
  (b) shift-aware accuracy: weighted by P(is_test) from exp_006's adversarial
      classifier (grid features), i.e. the same reweighting exp_006 used to
      audit exp_005 alone -- now applied to the blend.
If (a) and (b) pick different w, and (b)'s choice does better on the most-
test-like quartile, that's direct support for scoring blend search on
shift-aware accuracy rather than plain OOF accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
NORM_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp004/"

grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
norm_train = pd.read_parquet(NORM_DIR + "train_features_norm.parquet")

grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
norm_feat_cols = [c for c in norm_train.columns if c not in ("Path", "Pitch_ID")]

class_counts = grid_train["Pitch_ID"].value_counts()
N_SPLITS = 3
keep_classes = class_counts[class_counts >= N_SPLITS].index
grid_train = grid_train[grid_train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

# align norm_train to the same row order/subset as grid_train via Path
norm_train = norm_train.set_index("Path").loc[grid_train["Path"]].reset_index()
assert (norm_train["Pitch_ID"].values == grid_train["Pitch_ID"].values).all()

le = LabelEncoder()
y = le.fit_transform(grid_train["Pitch_ID"].values)
n_classes = len(np.unique(y))

Xg = grid_train[grid_feat_cols].values
Xn = norm_train[norm_feat_cols].values

N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(Xg, y))

oof_proba_g = np.zeros((len(y), n_classes))
oof_proba_n = np.zeros((len(y), n_classes))

for tr_idx, va_idx in splits:
    clf_g = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf_g.fit(Xg[tr_idx], y[tr_idx])
    oof_proba_g[va_idx] = clf_g.predict_proba(Xg[va_idx])

    clf_n = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
        subsample=0.8, colsample_bytree=0.8, objective="multiclass",
        num_class=n_classes, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf_n.fit(Xn[tr_idx], y[tr_idx])
    oof_proba_n[va_idx] = clf_n.predict_proba(Xn[va_idx])

acc_g = accuracy_score(y, oof_proba_g.argmax(axis=1))
acc_n = accuracy_score(y, oof_proba_n.argmax(axis=1))
print(f"grid-only OOF acc:    {acc_g:.4f}")
print(f"generic-only OOF acc: {acc_n:.4f}")

# --- adversarial classifier on grid features (same as exp_006) for shift-aware weights ---
X_adv = np.vstack([Xg, grid_test[grid_feat_cols].values])
y_adv = np.concatenate([np.zeros(len(Xg)), np.ones(len(grid_test))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tr_idx, va_idx in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tr_idx], y_adv[tr_idx])
    p_test_all[va_idx] = clf.predict_proba(X_adv[va_idx])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
p_test_train = p_test_all[: len(Xg)]
print(f"adversarial AUC (grid features, train vs test): {adv_auc:.4f}")

q75 = np.quantile(p_test_train, 0.75)
topq_mask = p_test_train >= q75

# --- blend weight grid search ---
print(f"\n{'w(grid)':>8} {'plain_acc':>10} {'shift_wtd_acc':>14} {'topq_acc':>10}")
results = []
for w in np.arange(0.0, 1.01, 0.1):
    proba_blend = w * oof_proba_g + (1 - w) * oof_proba_n
    pred = proba_blend.argmax(axis=1)
    correct = (pred == y).astype(float)
    plain_acc = correct.mean()
    shift_wtd_acc = np.average(correct, weights=p_test_train)
    topq_acc = correct[topq_mask].mean()
    results.append((w, plain_acc, shift_wtd_acc, topq_acc))
    print(f"{w:8.1f} {plain_acc:10.4f} {shift_wtd_acc:14.4f} {topq_acc:10.4f}")

best_plain = max(results, key=lambda r: r[1])
best_shift = max(results, key=lambda r: r[2])
print(f"\nbest w by plain OOF accuracy:        w={best_plain[0]:.1f} (plain_acc={best_plain[1]:.4f}, topq_acc={best_plain[3]:.4f})")
print(f"best w by shift-aware weighted acc:  w={best_shift[0]:.1f} (shift_wtd_acc={best_shift[2]:.4f}, topq_acc={best_shift[3]:.4f})")
print(f"topq_acc at plain-optimal w vs shift-optimal w: {best_plain[3]:.4f} vs {best_shift[3]:.4f}")
