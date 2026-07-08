"""Extract audio features for the missing-fundamental-puzzle competition.

Usage:
  ~/ml/bin/python extract_features.py                  # full train+test
  ~/ml/bin/python extract_features.py --probe 300 100   # probe: 300 train, 100 test rows
  ~/ml/bin/python extract_features.py --probe 300 100 --normalize  # exp_004: per-clip RMS-normalize amplitude first

Writes agents/claude/workspace/scripts/train_features.parquet and test_features.parquet
(or train_features_probe.parquet / test_features_probe.parquet in --probe mode;
--normalize appends a _norm suffix so it doesn't clobber the un-normalized probe).
This box has only 2 CPUs and each file costs ~7-8s (mfcc/harmonic/tonnetz dominate) —
always probe on a subsample first (C13), and only run the full extraction once a
probe shows signal (ship the full run to Kaggle compute via kkernel.py if it would
eat the whole work block locally).
"""
import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

N_MFCC = 20
SR_TARGET = None  # keep native 44100
NORMALIZE = False  # exp_004: per-clip RMS normalization (set by CLI flag)


def extract_one(path):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if NORMALIZE:
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        y = y / rms * 0.1  # target RMS 0.1, removes per-clip loudness/noise-floor differences

    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i}_mean"] = mfcc[i].mean()
        feats[f"mfcc_{i}_std"] = mfcc[i].std()

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_stft_{i}_mean"] = chroma_stft[i].mean()
        feats[f"chroma_stft_{i}_std"] = chroma_stft[i].std()

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_cqt_{i}_mean"] = chroma_cqt[i].mean()
        feats[f"chroma_cqt_{i}_std"] = chroma_cqt[i].std()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"contrast_{i}_mean"] = contrast[i].mean()
        feats[f"contrast_{i}_std"] = contrast[i].std()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = centroid.mean()
    feats["centroid_std"] = centroid.std()

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = bandwidth.mean()
    feats["bandwidth_std"] = bandwidth.std()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean"] = rolloff.mean()
    feats["rolloff_std"] = rolloff.std()

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    feats["flatness_mean"] = flatness.mean()
    feats["flatness_std"] = flatness.std()

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    feats["zcr_mean"] = zcr.mean()
    feats["zcr_std"] = zcr.std()

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        feats[f"tonnetz_{i}_mean"] = tonnetz[i].mean()
        feats[f"tonnetz_{i}_std"] = tonnetz[i].std()

    # Harmonic-spacing feature: autocorrelation of the log-magnitude spectrum
    # (cepstral peak picking) is more robust to a missing fundamental than
    # time-domain autocorrelation / YIN, since it looks at overtone spacing.
    S = np.abs(librosa.stft(y))
    log_S = np.log1p(S.mean(axis=1))
    spec_acf = np.correlate(log_S, log_S, mode="full")[len(log_S) - 1 :]
    spec_acf = spec_acf / (spec_acf[0] + 1e-8)
    for i, lag in enumerate([5, 10, 20, 30, 50, 80]):
        feats[f"spec_acf_lag{lag}"] = spec_acf[lag] if lag < len(spec_acf) else 0.0

    return feats


def run(csv_path, has_label, out_name, n_sample=None):
    df = pd.read_csv(csv_path)
    if n_sample is not None and n_sample < len(df):
        if has_label:
            frac = n_sample / len(df)
            parts = [g.sample(max(1, round(len(g) * frac)), random_state=42)
                     for _, g in df.groupby("Pitch_ID")]
            df = pd.concat(parts).reset_index(drop=True)
        else:
            df = df.sample(n_sample, random_state=42).reset_index(drop=True)

    t0 = time.time()
    with Pool(2) as pool:
        rows = []
        for i, feats in enumerate(pool.imap(extract_one, df["Path"], chunksize=4)):
            rows.append(feats)
            if (i + 1) % 50 == 0:
                print(f"{out_name}: {i+1}/{len(df)} ({time.time()-t0:.1f}s)")
    feat_df = pd.DataFrame(rows)
    feat_df["Path"] = df["Path"].values
    if has_label:
        feat_df["Pitch_ID"] = df["Pitch_ID"].values
    feat_df.to_parquet(OUT_DIR + out_name)
    print(f"wrote {OUT_DIR + out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs=2, type=int, metavar=("N_TRAIN", "N_TEST"), default=None)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()
    NORMALIZE = args.normalize
    norm_suffix = "_norm" if args.normalize else ""

    if args.probe:
        n_train, n_test = args.probe
        run(DATA_ROOT + "kaggle_dataset/train.csv", True, f"train_features{norm_suffix}_probe.parquet", n_train)
        run(DATA_ROOT + "kaggle_dataset/test.csv", False, f"test_features{norm_suffix}_probe.parquet", n_test)
    else:
        run(DATA_ROOT + "kaggle_dataset/train.csv", True, f"train_features{norm_suffix}.parquet")
        run(DATA_ROOT + "kaggle_dataset/test.csv", False, f"test_features{norm_suffix}.parquet")
