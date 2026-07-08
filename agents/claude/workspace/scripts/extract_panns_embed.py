"""exp_029 probe: frozen pretrained PANNs (Cnn14, AudioSet-trained general
audio-event classifier) used as a FEATURE EXTRACTOR (not fine-tuned) --
different mechanism from exp_028's CREPE (pitch-specific, failed badly,
probe_delta -0.0206): Cnn14 is trained on broad timbral/spectral audio-event
discrimination (AudioSet, ~2M clips, 527 classes), which may capture the
harmonic/timbral cues this task's Pitch_ID classes depend on better than a
narrow single-f0 pitch-tracking model. Own implementation/idea (triggered by
/research pass finding general-purpose pretrained embeddings, e.g. VGGish/
OpenL3/PANNs, outperform task-specific embeddings on small-dataset transfer
in several audio-transfer-learning papers).

2048-dim global-pooled embedding per clip (Cnn14's penultimate layer), no
fine-tuning, no test labels touched.
"""
import time
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from panns_inference import AudioTagging

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
OUT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
PANNS_SR = 32000


def load_resampled(path):
    y, sr = sf.read(DATA_ROOT + path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    g = gcd(sr, PANNS_SR)
    up, down = PANNS_SR // g, sr // g
    return resample_poly(y, up, down).astype(np.float32)


def run(parquet_in, out_name, at):
    ref = pd.read_parquet(OUT_DIR + parquet_in)
    paths = ref["Path"].tolist()

    t0 = time.time()
    rows = []
    for i, path in enumerate(paths):
        audio = load_resampled(path)[None, :]
        _, emb = at.inference(audio)
        rows.append(emb[0])
        if (i + 1) % 50 == 0:
            print(f"{out_name}: {i+1}/{len(paths)} ({time.time()-t0:.1f}s)", flush=True)
    feat = np.vstack(rows)
    cols = [f"panns_{i}" for i in range(feat.shape[1])]
    feat_df = pd.DataFrame(feat, columns=cols)
    feat_df["Path"] = ref["Path"].values
    if "Pitch_ID" in ref.columns:
        feat_df["Pitch_ID"] = ref["Pitch_ID"].values
    feat_df.to_parquet(OUT_DIR + out_name)
    print(f"wrote {OUT_DIR + out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    at = AudioTagging(checkpoint_path=None, device="cpu")
    run("train_grid_features_probe.parquet", "train_panns_features_probe.parquet", at)
