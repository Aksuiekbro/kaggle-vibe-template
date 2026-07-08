"""exp_005 full-fidelity run: MIDI-grid harmonic/ACF/cepstrum pitch-candidate
features (own implementation, RMS-normalized), LGBM multiclass CV.

Probe (300/100 subsample) showed OOF 0.5876 vs exp_002's plain-librosa probe
0.1168 and exp_004's normalized-librosa probe 0.1512 -- a much larger signal
than generic spectral/chroma summary stats. Motivated by cross-reviewing
gemini's sub_018 (CV 0.968), whose MIDI-grid harmonic/ACF/cepstrum features
independently found the same lever; this is our own from-scratch
implementation to verify the signal ourselves (see
extract_grid_features.py and .ai/reviews/claude-reviews-gemini-sub_018.md).

Runs in one kernel session:
  1. Extract grid-harmonic/ACF/cepstrum features for ALL train+test WAVs
     (RMS-normalized, 81 MIDI-note candidates x 5 features each).
  2. StratifiedKFold(3) LightGBM multiclass CV (full_delta signal for exp_005).
  3. Fit on full train, predict test, write submission.csv + metrics.json.
"""
import glob
import json
import os
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def discover_data_root():
    matches = glob.glob("/kaggle/input/**/kaggle_dataset/train.csv", recursive=True)
    if not matches:
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"DIR {root}: {files[:10]}", flush=True)
        raise FileNotFoundError("no kaggle_dataset/train.csv found under /kaggle/input")
    root = matches[0].rsplit("kaggle_dataset/train.csv", 1)[0]
    print(f"discovered DATA_ROOT={root}", flush=True)
    return root


DATA_ROOT = discover_data_root()
N_WORKERS = max(1, os.cpu_count())
TARGET_RMS = 0.1
MIDI_NOTES = np.arange(30, 111)  # 81 candidate fundamentals, A440 equal temperament
N_HARMONICS = 8
TOL = 0.02  # +/-2% frequency tolerance window per harmonic


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
        feats[f"d{d}_acf"] = acf[lag_r] if 0 <= lag_r < len(acf) else 0.0
        feats[f"d{d}_cep"] = cep[lag_r] if 0 <= lag_r < len(cep) else 0.0

    return feats


def extract_all(csv_path, has_label, out_name):
    df = pd.read_csv(csv_path)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        rows = []
        for i, feats in enumerate(pool.imap(extract_one, df["Path"], chunksize=4)):
            rows.append(feats)
            if (i + 1) % 200 == 0:
                print(f"{out_name}: {i+1}/{len(df)} ({time.time()-t0:.1f}s)", flush=True)
    feat_df = pd.DataFrame(rows)
    feat_df["Path"] = df["Path"].values
    if has_label:
        feat_df["Pitch_ID"] = df["Pitch_ID"].values
    feat_df.to_parquet(out_name)
    print(f"wrote {out_name} shape={feat_df.shape} in {time.time()-t0:.1f}s", flush=True)
    return feat_df


def main():
    print(f"N_WORKERS={N_WORKERS}", flush=True)
    train = extract_all(DATA_ROOT + "kaggle_dataset/train.csv", True, "train_grid_features.parquet")
    test = extract_all(DATA_ROOT + "kaggle_dataset/test.csv", False, "test_grid_features.parquet")

    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    X_train = train[feat_cols].values
    X_test = test[feat_cols].values

    y_raw = train["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_counts = pd.Series(y_raw).value_counts()
    print(f"n={len(train)} classes={len(class_counts)} n_feats={len(feat_cols)} "
          f"min_per_class={class_counts.min()} max_per_class={class_counts.max()}", flush=True)

    N_SPLITS = 3
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_accs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
                                  objective="multiclass", num_class=len(np.unique(y)),
                                  random_state=42, verbosity=-1, n_jobs=N_WORKERS)
        clf.fit(X_train[tr_idx], y[tr_idx])
        pred = clf.predict(X_train[va_idx])
        oof_pred[va_idx] = pred
        acc = accuracy_score(y[va_idx], pred)
        fold_accs.append(acc)
        print(f"fold {fold}: acc={acc:.4f} (n_va={len(va_idx)})", flush=True)

    overall_acc = accuracy_score(y, oof_pred)
    print(f"exp_005 full OOF accuracy (vs exp_002 full 0.5682 unnormalized, exp_004 pending): {overall_acc:.4f}", flush=True)
    print(f"fold mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}", flush=True)

    clf_final = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
                                    objective="multiclass", num_class=len(np.unique(y)),
                                    random_state=42, verbosity=-1, n_jobs=N_WORKERS)
    clf_final.fit(X_train, y)
    test_pred = clf_final.predict(X_test)
    test_pitch_id = le.inverse_transform(test_pred)
    sub = pd.DataFrame({"Path": test["Path"].values, "Pitch_ID": test_pitch_id})
    sub.to_csv("submission.csv", index=False)

    metrics = {
        "exp_005_full_oof_acc": overall_acc,
        "exp_005_fold_accs": fold_accs,
        "exp_002_unnormalized_full_oof_acc": 0.5682,
        "exp_005_full_delta_vs_exp002": overall_acc - 0.5682,
        "min_per_class": int(class_counts.min()),
        "max_per_class": int(class_counts.max()),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
