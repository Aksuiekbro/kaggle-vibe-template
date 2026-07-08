"""exp_075 full-fidelity promotion (scheduler id exp_075, idea exp_077):
full-dataset 16kHz scattering extraction. Probe (probe_exp077_scattering_higher_sr.py,
n=542 stratified subsample) found topq 0.7941 (16kHz) vs 0.7353 (8kHz), a +0.0588
probe delta -- but topq_high_freq_tertile (the exact subgroup the Nyquist-truncation
hypothesis predicts should move) was IDENTICAL (0.7317) between 8kHz and 16kHz,
so the aggregate lift may be probe noise (n=34) rather than the hypothesized
mechanism. Full CV is the only way to resolve this cheaply enough to trust.

Same extraction pattern as extract_scattering_features.py (exp_073), SR and N
doubled to keep the same ~8.2s time window at higher resolution.
"""
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

SR = 16000
N = 2 ** 17
J, Q = 8, 8

_scattering = Scattering1D(J=J, Q=Q, shape=(N,))


def extract_one(path):
    y, _ = librosa.load(DATA_ROOT + path, sr=SR)
    if len(y) < N:
        y = np.pad(y, (0, N - len(y)))
    else:
        y = y[:N]
    Sx = _scattering(y.astype(np.float64))
    mean_feats = Sx.mean(axis=1)
    std_feats = Sx.std(axis=1)
    feats = {f"sc16_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"sc16_std_{i}": v for i, v in enumerate(std_feats)})
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
    run(TRAIN_CSV, "train_scattering16k_features.parquet", True)
    run(TEST_CSV, "test_scattering16k_features.parquet", False)
