"""exp_086 probe: dense continuous-F0 harmonic-comb peak matcher + fold-safe
nearest-class-centroid classifier.

Motivation: real codex_* Kaggle submissions (verified directly via `kaggle
competitions submissions`) score public 0.994 / private up to 1.00000 on this
competition, described as "peak-grid harmonic ridge" matching -- far above our
GBDT-on-summary-features ceiling (topq ~0.97, LB 0.954). Sanity check first
(exp_005 fixed-grid argmax) got only 0.016-0.037 acc (~chance), confirming
Pitch_ID is NOT a direct index into MIDI notes 30-110 as exp_005 assumed.
This script instead: (1) estimates F0 per clip via a dense, continuous
(non-equal-tempered-assuming) harmonic-comb peak search, robust to octave
ambiguity via a coarse-then-fine two-stage search; (2) builds fold-safe
per-class centroid F0 from train labels; (3) classifies by nearest centroid
in log-frequency space. This never assumes Pitch_ID <-> MIDI note identity --
centroids are learned empirically per class.

Usage: python exp086_harmonic_comb_f0.py [--n N] [--full]
"""
import argparse
import time

import numpy as np
import pandas as pd
import soundfile as sf
from multiprocessing import Pool

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

TARGET_RMS = 0.1
N_HARMONICS = 15
F_LO, F_HI = 20.0, 2500.0
COARSE_N = 900   # ~0.1 semitone-ish resolution across ~7 octaves
FINE_HALFWIDTH_SEMITONES = 1.0
FINE_N = 61


def max_filter_1d(x, w):
    n = len(x)
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty(n)
    # simple O(n*w) is fine at this scale (n ~ 1.5e5, w ~7)
    for i in range(n):
        out[i] = xp[i:i + w].max()
    return out


def estimate_f0(path):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms * TARGET_RMS

    N = len(y)
    X_log = np.log1p(np.abs(np.fft.rfft(y)))
    X_max = max_filter_1d(X_log, 7)
    factor = N / sr
    n_bins = len(X_max)

    def score_grid(freqs):
        r = np.arange(1, N_HARMONICS + 1)
        targets = freqs[:, None] * r[None, :]  # (C, R)
        bins = np.clip(np.round(targets * factor).astype(int), 0, n_bins - 1)
        vals = X_max[bins]  # (C, R)
        weights = 1.0 / r  # de-emphasize high harmonics
        return (vals * weights[None, :]).sum(axis=1)

    coarse_freqs = np.geomspace(F_LO, F_HI, COARSE_N)
    coarse_scores = score_grid(coarse_freqs)
    best_coarse = coarse_freqs[np.argmax(coarse_scores)]

    lo = best_coarse * 2 ** (-FINE_HALFWIDTH_SEMITONES / 12.0)
    hi = best_coarse * 2 ** (FINE_HALFWIDTH_SEMITONES / 12.0)
    fine_freqs = np.linspace(lo, hi, FINE_N)
    fine_scores = score_grid(fine_freqs)
    best_fine = fine_freqs[np.argmax(fine_scores)]
    best_score = fine_scores.max()

    # Missing-fundamental octave/harmonic-multiple correction: a candidate at
    # k*F0_true scores well too (all its harmonics are a subset of F0_true's
    # real harmonics), so raw argmax is biased toward multiples of the true
    # (silent) fundamental. Test integer submultiples and prefer the SMALLEST
    # one whose score still clears a fraction of the best candidate's score
    # (standard octave-error correction heuristic).
    SUBMULT_THRESH = 0.5
    for k in range(8, 1, -1):
        cand = best_fine / k
        if cand < F_LO:
            continue
        s = score_grid(np.array([cand]))[0]
        if s >= SUBMULT_THRESH * best_score:
            best_fine = cand
            break

    return best_fine


def run(n, full):
    train = pd.read_csv(DATA_ROOT + "kaggle_dataset/train.csv")
    if not full:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
        train = train.iloc[idx].reset_index(drop=True)

    t0 = time.time()
    with Pool(2) as pool:
        f0s = []
        for i, f0 in enumerate(pool.imap(estimate_f0, train["Path"].tolist(), chunksize=4)):
            f0s.append(f0)
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(train)} ({time.time()-t0:.1f}s)")
    train["f0_est"] = f0s
    train["log2f0"] = np.log2(train["f0_est"])
    suffix = "_full" if full else "_probe"
    train.to_parquet(f"{OUT_DIR}exp086_f0_estimates{suffix}.parquet")
    print(f"wrote exp086_f0_estimates{suffix}.parquet in {time.time()-t0:.1f}s")

    # fold-safe nearest-centroid eval, 3-fold
    from sklearn.model_selection import StratifiedKFold
    y = train["Pitch_ID"].values
    log2f0 = train["log2f0"].values
    vc = pd.Series(y).value_counts()
    n_splits = 3 if vc.min() >= 3 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr_idx, va_idx in skf.split(train, y):
        centroids = {}
        for c in np.unique(y[tr_idx]):
            mask = y[tr_idx] == c
            centroids[c] = np.median(log2f0[tr_idx][mask])
        classes = np.array(list(centroids.keys()))
        cvals = np.array(list(centroids.values()))
        preds = []
        for v in log2f0[va_idx]:
            preds.append(classes[np.argmin(np.abs(cvals - v))])
        preds = np.array(preds)
        acc = (preds == y[va_idx]).mean()
        accs.append(acc)
        print(f"fold acc={acc:.4f}")
    print(f"mean acc={np.mean(accs):.4f} std={np.std(accs):.4f}")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(args.n, args.full)
