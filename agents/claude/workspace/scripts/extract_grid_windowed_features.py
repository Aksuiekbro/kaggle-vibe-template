"""exp_015 probe: temporal-windowed harmonic-evidence features, additive to
exp_005's grid features. Splits each clip into 3 equal, non-overlapping time
windows and computes each candidate's harmonic-energy score (same freq-domain
scoring as extract_grid_features.py) per window, then summarizes across
windows with max and std. Motivation: a 6.6s clip may carry stronger/weaker
missing-fundamental pitch evidence at different points (e.g. onset vs
sustain); the full-clip-only features (exp_005) can't see that variability.
This produces ADDITIONAL columns only (max_win, std_win per candidate) to be
concatenated with exp_005's existing 405 features, not a replacement.

Usage: ~/ml/bin/python extract_grid_windowed_features.py [--full]
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

TARGET_RMS = 0.1
MIDI_NOTES = np.arange(30, 111)
N_HARMONICS = 8
TOL = 0.02
N_WINDOWS = 3


def window_harm_scores(y_win, sr):
    N = len(y_win)
    if N < 64:
        return np.zeros(len(MIDI_NOTES))
    factor = N / sr
    X_log = np.log1p(np.abs(np.fft.rfft(y_win)))
    scores = np.zeros(len(MIDI_NOTES))
    for i, d in enumerate(MIDI_NOTES):
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        hs = []
        for r in range(1, N_HARMONICS + 1):
            target = r * f0
            lo = int(np.floor(target * (1 - TOL) * factor))
            hi = int(np.ceil(target * (1 + TOL) * factor)) + 1
            lo = max(0, lo)
            hi = min(len(X_log), hi)
            hs.append(X_log[lo:hi].max() if lo < hi else 0.0)
        scores[i] = np.sum(hs)
    return scores


def extract_one(path):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms * TARGET_RMS

    N = len(y)
    edges = np.linspace(0, N, N_WINDOWS + 1).astype(int)
    win_scores = np.zeros((N_WINDOWS, len(MIDI_NOTES)))
    for w in range(N_WINDOWS):
        win_scores[w] = window_harm_scores(y[edges[w]:edges[w + 1]], sr)

    max_over_win = win_scores.max(axis=0)
    std_over_win = win_scores.std(axis=0)

    feats = {}
    for i, d in enumerate(MIDI_NOTES):
        feats[f"d{d}_harm_win_max"] = max_over_win[i]
        feats[f"d{d}_harm_win_std"] = std_over_win[i]
    return feats


def run(parquet_in, out_name):
    ref = pd.read_parquet(OUT_DIR + parquet_in)
    paths = ref["Path"].tolist()

    t0 = time.time()
    with Pool(2) as pool:
        rows = []
        for i, feats in enumerate(pool.imap(extract_one, paths, chunksize=4)):
            rows.append(feats)
            if (i + 1) % 50 == 0:
                print(f"{out_name}: {i+1}/{len(paths)} ({time.time()-t0:.1f}s)")
    feat_df = pd.DataFrame(rows)
    feat_df["Path"] = ref["Path"].values
    if "Pitch_ID" in ref.columns:
        feat_df["Pitch_ID"] = ref["Pitch_ID"].values
    feat_df.to_parquet(OUT_DIR + out_name)
    print(f"wrote {OUT_DIR + out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    suffix = "" if args.full else "_probe"
    run(f"train_features{suffix}.parquet", f"train_grid_windowed_features{suffix}.parquet")
    run(f"test_features{suffix}.parquet", f"test_grid_windowed_features{suffix}.parquet")
