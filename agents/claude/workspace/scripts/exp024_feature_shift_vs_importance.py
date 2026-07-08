"""exp_024: per-feature train/test shift diagnostic on grid features.

New angle on the still-unexplained sub_022 CV-up/LB-down drop: exp_006/020/023
all checked model- or blend-level shift diagnostics and found nothing. This
checks the FEATURE level instead -- for each of the 410 grid-harmonic feature
columns, compute a univariate adversarial AUC (train=0 vs test=1, ROC-AUC of
the raw column value as a score) and cross-reference against LGBM
feature_importances_ from a single full-train fit. Features that are BOTH
high-shift and high-importance are candidates to drop -- a concrete,
inspectable lever, unlike the exhausted model/blend-weight axes.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

CACHE_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def main():
    t0 = time.time()
    train = pd.read_parquet(CACHE_DIR + "train_grid_features.parquet")
    test = pd.read_parquet(CACHE_DIR + "test_grid_features.parquet")
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    print(f"n_features={len(feat_cols)} train={len(train)} test={len(test)}", flush=True)

    # 1. univariate adversarial AUC per feature (train=0, test=1)
    y_adv = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
    shift_auc = {}
    for c in feat_cols:
        vals = np.concatenate([train[c].values, test[c].values])
        try:
            auc = roc_auc_score(y_adv, vals)
        except ValueError:
            auc = 0.5
        shift_auc[c] = max(auc, 1 - auc)  # symmetric: how separable, either direction
    print(f"univariate AUC computed for all features ({time.time()-t0:.0f}s)", flush=True)

    # 2. single full LGBM fit for feature importances (full 2330-row train, no CV needed for this)
    le = LabelEncoder()
    y_all = le.fit_transform(train["Pitch_ID"].values)
    X_all = train[feat_cols].values
    t_fit = time.time()
    clf = lgb.LGBMClassifier(num_class=len(le.classes_), random_state=42, **LGBM_PARAMS)
    clf.fit(X_all, y_all)
    importances = dict(zip(feat_cols, clf.feature_importances_))
    print(f"full LGBM fit done ({time.time()-t_fit:.0f}s)", flush=True)

    df = pd.DataFrame({
        "feature": feat_cols,
        "shift_auc": [shift_auc[c] for c in feat_cols],
        "importance": [importances[c] for c in feat_cols],
    })
    df["importance_rank"] = df["importance"].rank(ascending=False)
    df["shift_rank"] = df["shift_auc"].rank(ascending=False)
    df["risk_score"] = df["importance_rank"] + df["shift_rank"]  # lower = more suspicious (both high)
    df = df.sort_values("risk_score")
    df.to_csv(CACHE_DIR + "exp024_feature_shift_vs_importance.csv", index=False)

    corr = df["shift_auc"].corr(df["importance"])
    print(f"\ncorrelation(shift_auc, importance) across all {len(df)} features: {corr:.4f}", flush=True)
    print("\ntop 15 most suspicious (high importance AND high shift):", flush=True)
    print(df.head(15).to_string(index=False), flush=True)

    high_shift = df[df["shift_auc"] > 0.7]
    high_shift_high_imp = high_shift[high_shift["importance_rank"] <= len(df) * 0.25]
    print(f"\n{len(high_shift)} features with shift_auc>0.7; "
          f"{len(high_shift_high_imp)} of those are also top-25% importance", flush=True)

    if len(high_shift_high_imp) > 0:
        drop_cols = high_shift_high_imp["feature"].tolist()
        keep_cols = [c for c in feat_cols if c not in drop_cols]
        print(f"\nprobe: 3-fold CV with {len(drop_cols)} suspicious features dropped "
              f"({len(keep_cols)} remain)", flush=True)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for label, cols in (("full", feat_cols), ("dropped", keep_cols)):
            X = train[cols].values
            preds = np.zeros(len(y_all), dtype=int)
            for tr_idx, va_idx in skf.split(X, y_all):
                c = lgb.LGBMClassifier(num_class=len(le.classes_), random_state=42, **LGBM_PARAMS)
                c.fit(X[tr_idx], y_all[tr_idx])
                preds[va_idx] = c.predict(X[va_idx])
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_all, preds)
            print(f"  {label}: OOF acc={acc:.4f}", flush=True)
    else:
        print("no features are both high-shift and high-importance -- no drop candidate found", flush=True)

    print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
