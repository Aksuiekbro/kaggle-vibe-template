"""exp_014 probe: grid-harmonic features v2 -- widen the ACF/cepstrum lag
lookup from a single rounded-sample lag to a small window (max over lag +/-2
samples), replacing exp_005's exact-rounding lookup.

Motivation: at high candidate f0 (e.g. MIDI 110 ~= 1046 Hz, sr=44100), the
ideal lag sr/f0 ~= 42.2 samples -- rounding to the nearest integer sample can
miss the true ACF/cepstrum peak by a meaningful fraction of a semitone,
especially at short lags where 1 sample = a bigger relative frequency step.
The harmonic frequency-domain score already uses a +/-2% tolerance window
(TOL=0.02); the ACF/cepstrum features had no such tolerance. This is a
targeted robustness fix to the existing feature family (not new candidates),
per STRATEGY.md priority 2.

Usage: ~/ml/bin/python extract_grid_features_v2.py [--full]
"""
import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

TARGET_RMS = 0.1
MIDI_NOTES = np.arange(30, 111)  # 81 candidate fundamentals, A440 equal temperament
N_HARMONICS = 8
TOL = 0.02  # +/-2% frequency tolerance window per harmonic
LAG_WINDOW = 2  # +/- samples around the rounded ideal lag for ACF/cepstrum


def extract_one(path):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms * TARGET_RMS

    N = len(y)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    X_log = np.log1p(np.abs(np.fft.rfft(y)))

    n_fft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    y_fft = np.fft.rfft(y, n_fft)
    acf = np.fft.irfft(y_fft * np.conj(y_fft))[:N]
    acf = acf / (acf[0] + 1e-8)

    cep = np.fft.irfft(np.log(np.abs(np.fft.rfft(y)) + 1e-6))[:N]

    feats = {}
    factor = N / sr
    for d in MIDI_NOTES:
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)

        harmonic_scores = []
        for r in range(1, N_HARMONICS + 1):
            target = r * f0
            lo = int(np.floor(target * (1 - TOL) * factor))
            hi = int(np.ceil(target * (1 + TOL) * factor)) + 1
            lo = max(0, lo)
            hi = min(len(X_log), hi)
            harmonic_scores.append(X_log[lo:hi].max() if lo < hi else 0.0)
        harmonic_scores = np.array(harmonic_scores)

        feats[f"d{d}_harm_sum"] = harmonic_scores.sum()
        feats[f"d{d}_harm_low"] = harmonic_scores[:3].sum()
        feats[f"d{d}_harm_high"] = harmonic_scores[3:].sum()

        lag = sr / f0
        lag_r = int(round(lag))
        lo_l = max(0, lag_r - LAG_WINDOW)
        hi_l = min(len(acf), lag_r + LAG_WINDOW + 1)
        feats[f"d{d}_acf"] = acf[lo_l:hi_l].max() if lo_l < hi_l else 0.0
        hi_c = min(len(cep), lag_r + LAG_WINDOW + 1)
        feats[f"d{d}_cep"] = cep[lo_l:hi_c].max() if lo_l < hi_c else 0.0

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
    run(f"train_features{suffix}.parquet", f"train_grid_v2_features{suffix}.parquet")
    run(f"test_features{suffix}.parquet", f"test_grid_v2_features{suffix}.parquet")
