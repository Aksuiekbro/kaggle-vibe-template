"""exp_040 (own, probe, no-ML): direct peak-ridge harmonic matcher.

Motivation: MEMORY_CANDIDATES.md row 21 flags 7 undocumented codex_* Kaggle
submissions (2026-07-05, confirmed live via `kaggle competitions submissions -v`,
zero footprint in shared/submissions/registry.json) scoring publicScore up to
0.99425 and privateScore up to 1.00000 -- far above this team's tracked best
(0.91954 LB). Descriptions ("peak-grid harmonic ridge", "dense log-spectrum
ridge blend", CV means of 0.987-0.9996) read as a DETERMINISTIC DSP decision
rule (argmax over a harmonic-ridge score), not a trained classifier. A naive
sanity check already ran (per that memory row): plain argmax over exp_005's
existing dense grid features (window-max energy at each harmonic, checked at
every one of 8 harmonics for every candidate regardless of whether that
harmonic is actually present) reached only 5.8% accuracy -- nowhere near
codex's number.

This script tests a materially different mechanism: PEAK-PICK the spectrum
first (find actual local maxima, not a blind window-max at fixed harmonic
frequencies), then score each candidate f0 by how well the DETECTED peaks
align with its harmonic series (tolerant of missing harmonics -- a peak-ridge
match only credits peaks that exist, rather than forcing a score at all 8
harmonic slots). This directly targets the "missing fundamental" structure:
gemini's pitch_yin_mapping.csv shows some classes have near-zero YIN std
(<0.01 semitones, e.g. Pitch_ID 2/67/57) while others have huge std (up to
18.4 semitones, e.g. Pitch_ID 70/1), consistent with per-clip variation in
WHICH harmonics are actually present -- a fixed all-harmonics-required score
would be the wrong invariance; a detected-peaks-only ridge match should not
be.

No training at all -- straight argmax, scored against true Pitch_ID on the
existing 308-row probe subsample. If this clears even 50%+ accuracy (vs the
5.8% naive-window-max baseline), it's evidence the missing piece was peak-
detection-based ridge matching, not window-max scoring, and worth building
out to full scale / blending across multiple peak-count and tolerance
configs (as codex's naming convention, e.g. "peak12"/"peak16", "d0.06"/"d0.1"
tolerance, "w0.005"/"w0.12" weight, suggests they did).
"""
import time
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TARGET_RMS = 0.1
MIDI_NOTES = np.arange(30, 111)  # same 81-candidate grid as exp_005
MAX_HARMONIC = 20  # allow matching up to the 20th harmonic (codex's "peak16"/"peak12" naming suggests >8)


def score_candidates(path, n_peaks=16, tol_rel=0.02):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms * TARGET_RMS

    N = len(y)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mag = np.abs(np.fft.rfft(y))
    log_mag = np.log1p(mag)

    peak_idx, props = find_peaks(log_mag, height=log_mag.max() * 0.01)
    if len(peak_idx) == 0:
        return np.zeros(len(MIDI_NOTES))
    # keep the n_peaks tallest detected peaks
    order = np.argsort(log_mag[peak_idx])[::-1][:n_peaks]
    peak_idx = peak_idx[order]
    peak_freqs = freqs[peak_idx]
    peak_mags = log_mag[peak_idx]

    scores = np.zeros(len(MIDI_NOTES))
    for i, d in enumerate(MIDI_NOTES):
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        ratios = peak_freqs / f0
        r_round = np.round(ratios)
        r_round = np.clip(r_round, 1, MAX_HARMONIC)
        dev = np.abs(ratios - r_round) / r_round  # relative deviation from nearest harmonic
        matched = dev <= tol_rel
        # credit = peak magnitude, downweighted by how far off the nearest harmonic it is,
        # only for peaks whose nearest harmonic index is within tolerance
        credit = peak_mags[matched] * (1.0 - dev[matched] / tol_rel)
        scores[i] = credit.sum()
    return scores


if __name__ == "__main__":
    t0 = time.time()
    ref = pd.read_parquet(DIR + "train_grid_features_probe.parquet")[["Path", "Pitch_ID"]]
    paths = ref["Path"].tolist()
    y_true = ref["Pitch_ID"].values

    for n_peaks in (12, 16, 24):
        for tol_rel in (0.02, 0.04, 0.06):
            preds = []
            for path in paths:
                scores = score_candidates(path, n_peaks=n_peaks, tol_rel=tol_rel)
                preds.append(int(np.argmax(scores)))
            preds = np.array(preds)
            acc = (preds == y_true).mean()
            print(f"n_peaks={n_peaks} tol_rel={tol_rel}: direct-argmax acc={acc:.4f} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"total time: {time.time()-t0:.0f}s", flush=True)
