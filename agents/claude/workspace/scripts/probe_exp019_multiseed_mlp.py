"""exp_019 probe: multi-seed MLP averaging blended with LGBM (w_lgb=0.4, per
exp_017) on grid features. MLP fit is fast (~5s), so averaging 5 seeds is
nearly free. Probe uses full-data single stratified 80/20 holdout (not a
row subsample) per exp_012's fidelity lesson: row-subsampling probes gave
reversed signals for LGBM capacity tuning on this dataset.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

DATA_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
W_LGB = 0.4
SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.2
SPLIT_SEED = 42

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def manual_stratified_split(pitch_ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(pitch_ids):
        idx = np.where(pitch_ids == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_frac)) if len(idx) > 1 else 0
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def main():
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR + "train_grid_features.parquet")
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    le = LabelEncoder()
    y_all = le.fit_transform(train["Pitch_ID"].values)
    n_classes = len(le.classes_)
    X_all = train[feat_cols].values

    tr_idx, va_idx = manual_stratified_split(train["Pitch_ID"].values, VAL_FRAC, SPLIT_SEED)
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx], y_all[va_idx]
    print(f"split: n_train={len(tr_idx)} n_val={len(va_idx)}", flush=True)

    clf_lgb = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_lgb.fit(Xtr, ytr)
    proba_lgb = np.zeros((len(va_idx), n_classes))
    proba_lgb[:, clf_lgb.classes_] = clf_lgb.predict_proba(Xva)
    acc_lgb = accuracy_score(yva, np.argmax(proba_lgb, axis=1))
    print(f"LGBM acc={acc_lgb:.4f} ({time.time()-t0:.0f}s)", flush=True)

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

    mlp_probas = []
    for seed in SEEDS:
        t1 = time.time()
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
            early_stopping=True, random_state=seed,
        )
        clf_mlp.fit(Xtr_s, ytr)
        proba = np.zeros((len(va_idx), n_classes))
        proba[:, clf_mlp.classes_] = clf_mlp.predict_proba(Xva_s)
        mlp_probas.append(proba)
        acc_seed = accuracy_score(yva, np.argmax(proba, axis=1))
        print(f"  MLP seed={seed} acc={acc_seed:.4f} ({time.time()-t1:.0f}s)", flush=True)

    proba_mlp_single = mlp_probas[0]
    proba_mlp_avg = np.mean(mlp_probas, axis=0)

    acc_mlp_single = accuracy_score(yva, np.argmax(proba_mlp_single, axis=1))
    acc_mlp_avg = accuracy_score(yva, np.argmax(proba_mlp_avg, axis=1))

    blend_single = W_LGB * proba_lgb + (1 - W_LGB) * proba_mlp_single
    blend_avg = W_LGB * proba_lgb + (1 - W_LGB) * proba_mlp_avg
    acc_blend_single = accuracy_score(yva, np.argmax(blend_single, axis=1))
    acc_blend_avg = accuracy_score(yva, np.argmax(blend_avg, axis=1))

    print("\n=== summary ===", flush=True)
    print(f"MLP single-seed acc={acc_mlp_single:.4f}, MLP 5-seed-avg acc={acc_mlp_avg:.4f}", flush=True)
    print(f"blend (single-seed MLP) acc={acc_blend_single:.4f}", flush=True)
    print(f"blend (5-seed-avg MLP) acc={acc_blend_avg:.4f}", flush=True)
    print(f"delta (5-seed-avg blend - single-seed blend): {acc_blend_avg - acc_blend_single:+.4f}", flush=True)
    print(f"total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
