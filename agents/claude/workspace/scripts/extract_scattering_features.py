"""exp_073 (own, research-sourced): wavelet scattering transform (kymatio
Scattering1D) features, a mechanistically new family vs the grid-harmonic
features used everywhere so far. Motivated by kymatio's own docs citing
state-of-the-art results for musical-instrument classification under
limited annotated training data (small-N regime matches this competition's
82 classes / ~28 samples/class avg), and "Wavelet Scattering on the Pitch
Spiral" (arxiv 1601.00287, logged in READING_LOG) which argues scattering
coefficients linearize pitch dynamics separately from spectral envelope/
timbre -- a different invariance mechanism than harmonic-grid ACF/cepstrum
evidence scoring.

kymatio 0.3.0 has a scipy>=1.15 compatibility bug (scipy.special.sph_harm
was removed in favor of sph_harm_y) that breaks its top-level import even
though we only use Scattering1D, not the 3D module. Monkeypatched below --
narrow, load-bearing shim, not a general scipy downgrade.

Downsamples to 8kHz (voice/pitch content is well under the 4kHz Nyquist this
implies for the fundamentals/lower harmonics in this dataset's MIDI 30-110
range: MIDI 110 = ~3951 Hz, near the edge but higher harmonics beyond 4kHz
are dropped -- acceptable tradeoff for compute, flagged as a risk in
RESEARCH.md) and pads/truncates to 2**16 samples (~8.2s, covers all 6.6s
clips) for a fixed scattering shape. Summarizes each file's (234, 256)
scattering coefficient map by mean+std pooling over the time axis -> 468
features/file, comparable in scale to the 405-dim grid feature set.

Usage: ~/ml/bin/python extract_scattering_features.py [--full]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import scipy.special as _sp

if not hasattr(_sp, "sph_harm"):
    def _sph_harm_compat(m, n, theta, phi):
        return _sp.sph_harm_y(n, m, phi, theta)
    _sp.sph_harm = _sph_harm_compat

warnings.filterwarnings("ignore")
import librosa
from kymatio.numpy import Scattering1D

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
TRAIN_CSV = DATA_ROOT + "kaggle_dataset/train.csv"
TEST_CSV = DATA_ROOT + "kaggle_dataset/test.csv"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"

SR = 8000
N = 2 ** 16
J, Q = 8, 8

_scattering = Scattering1D(J=J, Q=Q, shape=(N,))


def extract_one(path):
    y, _ = librosa.load(DATA_ROOT + path, sr=SR)
    if len(y) < N:
        y = np.pad(y, (0, N - len(y)))
    else:
        y = y[:N]
    Sx = _scattering(y.astype(np.float64))  # (n_coeffs, n_time)
    mean_feats = Sx.mean(axis=1)
    std_feats = Sx.std(axis=1)
    feats = {f"sc_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"sc_std_{i}": v for i, v in enumerate(std_feats)})
    return feats


def run(csv_path, out_name, has_labels):
    df = pd.read_csv(csv_path)
    paths = df["Path"].tolist()

    t0 = time.time()
    rows = []
    for i, path in enumerate(paths):
        rows.append(extract_one(path))
        if (i + 1) % 50 == 0:
            print(f"{out_name}: {i+1}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)
    feat_df = pd.DataFrame(rows)
    feat_df["Path"] = df["Path"].values
    if has_labels:
        feat_df["Pitch_ID"] = df["Pitch_ID"].values
    feat_df.to_parquet(OUT_DIR + out_name)
    print(f"wrote {OUT_DIR + out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-subsample", type=int, default=0)
    args = ap.parse_args()

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if not args.full:
        n = args.n_subsample or 300
        train_df = train_df.sample(n=min(n, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(n // 3, len(test_df)), random_state=42)
        train_df.to_csv(OUT_DIR + "_scatter_probe_train.csv", index=False)
        test_df.to_csv(OUT_DIR + "_scatter_probe_test.csv", index=False)
        run(OUT_DIR + "_scatter_probe_train.csv", "train_scattering_features_probe.parquet", True)
        run(OUT_DIR + "_scatter_probe_test.csv", "test_scattering_features_probe.parquet", False)
    else:
        run(TRAIN_CSV, "train_scattering_features.parquet", True)
        run(TEST_CSV, "test_scattering_features.parquet", False)
