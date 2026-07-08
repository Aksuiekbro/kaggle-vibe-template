"""exp_028 probe: frozen pretrained CREPE (torchcrepe, model='tiny') used as a
feature extractor -- NOT trained from scratch. Embeds each clip's 5th-maxpool
activations, mean+std pools over time to a fixed-size vector, to test whether
a pretrained pitch-CNN adds signal on top of/alongside the exp_005 grid-
harmonic features. Own implementation (torchcrepe is a third-party pretrained
checkpoint, used only as a frozen extractor per RULES.md "own your solution" --
no test labels touched, no fine-tuning).

Uses the existing probe subsample (same 308 rows as train_grid_features_probe.parquet)
for an apples-to-apples comparison, per C4/C13 (smallest test first, probe before commit).
"""
import time
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchcrepe
from scipy.signal import resample_poly

torch.set_num_threads(1)

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
CREPE_SR = 16000
HOP_LENGTH = 1024  # 64ms -- coarse time resolution is fine, we mean/std-pool anyway


def embed_one(path):
    y, sr = sf.read(DATA_ROOT + path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    g = gcd(sr, CREPE_SR)
    up, down = CREPE_SR // g, sr // g
    y16 = resample_poly(y, up, down).astype(np.float32)
    audio = torch.from_numpy(y16).unsqueeze(0)
    with torch.no_grad():
        emb = torchcrepe.embed(audio, CREPE_SR, hop_length=HOP_LENGTH, model="tiny", device="cpu")
    emb = emb.squeeze(0).numpy()  # (frames, 32, 8)
    mean = emb.mean(axis=0).ravel()
    std = emb.std(axis=0).ravel()
    return np.concatenate([mean, std])


def run(parquet_in, out_name):
    ref = pd.read_parquet(OUT_DIR + parquet_in)
    paths = ref["Path"].tolist()

    t0 = time.time()
    rows = []
    for i, path in enumerate(paths):
        rows.append(embed_one(path))
        if (i + 1) % 50 == 0:
            print(f"{out_name}: {i+1}/{len(paths)} ({time.time()-t0:.1f}s)")
    feat = np.vstack(rows)
    cols = [f"crepe_{i}" for i in range(feat.shape[1])]
    feat_df = pd.DataFrame(feat, columns=cols)
    feat_df["Path"] = ref["Path"].values
    if "Pitch_ID" in ref.columns:
        feat_df["Pitch_ID"] = ref["Pitch_ID"].values
    feat_df.to_parquet(OUT_DIR + out_name)
    print(f"wrote {OUT_DIR + out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run("train_grid_features_probe.parquet", "train_crepe_features_probe.parquet")
