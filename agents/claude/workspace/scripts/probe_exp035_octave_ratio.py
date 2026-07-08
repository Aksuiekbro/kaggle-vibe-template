"""exp_035 probe: single new per-candidate feature targeting exp_034's confirmed
octave-confusion error mode (58% of OOF errors at 7/12/19/24-semitone harmonic-ratio
distances vs 27% null).

For each MIDI candidate d in the existing grid (d30-d110), add one ratio column:
    oct_ratio_d = harm_low[d] / (harm_low[d-12] + harm_low[d+12] + eps)
i.e. how much stronger the candidate's own low harmonics (1-3) are vs its
octave-neighbors' low harmonics. At the grid edges, only the neighbor(s) that
exist in d30-d110 are used.

CAUTION (C9 risk flag): three prior "add more grid-harmonic variants" attempts
this competition (exp_007, exp_014, exp_015) all reversed sign at probe scale.
Probed here at the same 308-row/65-class subsample as probe_lgbm_grid.py before
trusting it on full data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
EPS = 1e-8
MIDI_LO, MIDI_HI = 30, 110


def add_oct_ratio_features(df):
    df = df.copy()
    for d in range(MIDI_LO, MIDI_HI + 1):
        low_col = f"d{d}_harm_low"
        neighbors = []
        for nd in (d - 12, d + 12):
            if MIDI_LO <= nd <= MIDI_HI:
                neighbors.append(df[f"d{nd}_harm_low"])
        neighbor_sum = sum(neighbors) if neighbors else pd.Series(0.0, index=df.index)
        df[f"d{d}_oct_ratio"] = df[low_col] / (neighbor_sum + EPS)
    return df


def run(fname, label, transform=None):
    train = pd.read_parquet(OUT_DIR + fname)
    if transform is not None:
        train = transform(train)
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

    class_counts_all = train["Pitch_ID"].value_counts()
    N_SPLITS = 2
    keep_classes = class_counts_all[class_counts_all >= N_SPLITS].index
    train = train[train["Pitch_ID"].isin(keep_classes)].reset_index(drop=True)

    X = train[feat_cols].values
    y_raw = train["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(
            n_estimators=40, learning_rate=0.1, num_leaves=15, min_child_samples=2,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass",
            num_class=len(np.unique(y)), random_state=42, verbosity=-1, n_jobs=1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_pred[va_idx] = clf.predict(X[va_idx])

    acc = accuracy_score(y, oof_pred)
    print(f"{label}: n={len(train)} classes={len(np.unique(y))} n_feats={len(feat_cols)} OOF acc={acc:.4f}")
    return acc


acc_grid = run("train_grid_features_probe.parquet", "grid-harmonic only (reference)")
acc_oct = run("train_grid_features_probe.parquet", "grid + oct_ratio", transform=add_oct_ratio_features)
print(f"\ndelta (grid+oct_ratio - grid_only): {acc_oct - acc_grid:+.4f}")
