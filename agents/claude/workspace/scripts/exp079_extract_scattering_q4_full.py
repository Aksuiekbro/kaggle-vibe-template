"""exp_079 full-fidelity promotion (scheduler id exp_079, idea exp_080):
full-dataset Q=4 scattering extraction. Probe (probe_exp080_scattering_jq_sweep.py,
n=542/200 stratified subsample) found a monotonic gradient across Q in {4, 8, 16}:
Q=4 topq=0.7647 (+0.0294 vs Q=8 reference), Q=8 topq=0.7353 (reference, cached
full-dataset extraction), Q=16 topq=0.7059 (-0.0294) -- lower Q wins, opposite of
the arxiv-1601.00287-motivated hypothesis that pitch tasks need higher Q. Q=4 also
has fewer features (647 vs 873 at grid+scattering scale) so cheaper downstream.

Given this competition's 5+ prior probe-vs-full sign/magnitude reversals, this
full-dataset extraction + full CV is required before trusting the probe. Same SR/N
window as the Q=8 reference extraction (extract_scattering_features.py), only Q
changed.
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

SR = 8000
N = 2 ** 16
J, Q = 8, 4

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
    feats = {f"scq4_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"scq4_std_{i}": v for i, v in enumerate(std_feats)})
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
    run(TRAIN_CSV, "train_scattering_q4_features.parquet", True)
    run(TEST_CSV, "test_scattering_q4_features.parquet", False)
