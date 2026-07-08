"""exp_087 probe: test whether folding octave-correction into the
nearest-centroid classification step (rather than exp_086's greedy
pre-correction of the F0 estimate) fixes exp_086's failure.

Reuses exp_086's already-computed raw F0 estimates (no re-extraction).
For each held-out sample, generate a small set of octave-hypothesis log2F0
values (divide/multiply the raw estimate by small integers, matching the
multiples exp_086 diagnosed: 1,2,3,6,7 plus a few extras for safety), then
classify by nearest class centroid over ALL (hypothesis, class) pairs
jointly -- centroids are fold-safe (train-fold only). This turns octave
correction into a byproduct of classification instead of a separate greedy
pre-step.

Usage: python exp087_octave_aware_centroid.py [--parquet path]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ratios exp_086 found real (missing-fundamental clusters at log2(2,3,6,7))
# plus small extras for robustness; both directions (candidate could have
# locked onto a multiple OR a submultiple of the true F0).
RATIOS = [1, 2, 3, 4, 5, 6, 7, 8, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]


def eval_single_centroid(log2f0, y, tr_idx, va_idx):
    centroids = {}
    for c in np.unique(y[tr_idx]):
        centroids[c] = np.median(log2f0[tr_idx][y[tr_idx] == c])
    classes = np.array(list(centroids.keys()))
    cvals = np.array(list(centroids.values()))
    preds = classes[np.argmin(np.abs(cvals[None, :] - log2f0[va_idx][:, None]), axis=1)]
    return (preds == y[va_idx]).mean()


def eval_octave_aware_centroid(log2f0, y, tr_idx, va_idx):
    centroids = {}
    for c in np.unique(y[tr_idx]):
        centroids[c] = np.median(log2f0[tr_idx][y[tr_idx] == c])
    classes = np.array(list(centroids.keys()))
    cvals = np.array(list(centroids.values()))
    log_ratios = np.log2(np.array(RATIOS))  # (H,)
    preds = []
    for v in log2f0[va_idx]:
        hyps = v + log_ratios  # (H,) candidate log2F0 after each octave hypothesis
        dists = np.abs(cvals[None, :] - hyps[:, None])  # (H, C)
        best = np.unravel_index(np.argmin(dists), dists.shape)
        preds.append(classes[best[1]])
    preds = np.array(preds)
    return (preds == y[va_idx]).mean()


def run(parquet_path):
    df = pd.read_parquet(parquet_path)
    y = df["Pitch_ID"].values
    log2f0 = df["log2f0"].values
    vc = pd.Series(y).value_counts()
    n_splits = 3 if vc.min() >= 3 else 2
    print(f"n={len(df)} classes={df['Pitch_ID'].nunique()} n_splits={n_splits} min_class_count={vc.min()}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    single_accs, oct_accs = [], []
    for tr_idx, va_idx in skf.split(df, y):
        single_accs.append(eval_single_centroid(log2f0, y, tr_idx, va_idx))
        oct_accs.append(eval_octave_aware_centroid(log2f0, y, tr_idx, va_idx))
    print(f"single-centroid   mean acc={np.mean(single_accs):.4f} std={np.std(single_accs):.4f} folds={single_accs}")
    print(f"octave-aware      mean acc={np.mean(oct_accs):.4f} std={np.std(oct_accs):.4f} folds={oct_accs}")
    print(f"delta (octave-aware - single) = {np.mean(oct_accs) - np.mean(single_accs):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="agents/claude/workspace/scripts/exp086_f0_estimates_probe.parquet")
    args = ap.parse_args()
    run(args.parquet)
