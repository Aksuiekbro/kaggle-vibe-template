"""exp_018 probe: fold-safe additive-noise + time-stretch augmentation for
rare-tail classes (<10 samples/class, 14/82 classes, min 3), evaluated via a
single stratified 80/20 holdout (not full 3-fold CV -- probe fidelity fix
per exp_010/exp_012's row-subsampling lesson: keep full data, cut folds
instead of rows).

Cost stays probe-sized because only rare-class *training-split* audio is
reloaded and augmented (~50-90 files x 4 variants); everything else reuses
the already-cached full-scale grid features from kernel_output_exp005/
(same LGBM config as full_lgbm_tuned_grid.py's baseline for apples-to-apples
comparison against exp_005/exp_010's full OOF ~0.909).

No pitch-shift (would change the ground-truth label). Augmented copies are
added to the training split only and never touch the held-out val split, so
there is no leakage across the split.

Usage: ~/ml/bin/python probe_exp018_rare_augment.py
"""
import sys
import time
import warnings

import lightgbm as lgb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from extract_grid_features import DATA_ROOT, MIDI_NOTES, N_HARMONICS, TOL, TARGET_RMS

warnings.filterwarnings("ignore")

CACHE_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
RARE_THRESHOLD = 10
VAL_FRAC = 0.2
SEED = 42

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def extract_from_signal(y, sr):
    """Same feature set as extract_grid_features.extract_one, on an in-memory signal."""
    y = y.astype(np.float32)
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


def augment_variants(path, rng):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    variants = []
    sig_power = np.mean(y ** 2)
    for snr_db in (20.0, 12.0):
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
        variants.append(y + noise)
    for rate in (0.9, 1.1):
        variants.append(librosa.effects.time_stretch(y, rate=rate).astype(np.float32))
    return variants, sr


def manual_stratified_split(pitch_ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(pitch_ids):
        idx = np.where(pitch_ids == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_frac)) if len(idx) > 1 else 0
        n_val = min(n_val, len(idx) - 1)  # always leave >=1 for train
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def fit_eval(X_train, y_train, X_val, y_val, n_classes, label):
    clf = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    acc = accuracy_score(y_val, pred)
    bacc = balanced_accuracy_score(y_val, pred)
    print(f"{label}: n_train={len(y_train)} acc={acc:.4f} balanced_acc={bacc:.4f}")
    return acc, bacc, pred


def main():
    t0 = time.time()
    full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
    rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
    print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}")

    train = pd.read_parquet(CACHE_DIR + "train_grid_features.parquet")
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    le = LabelEncoder()
    y_all = le.fit_transform(train["Pitch_ID"].values)
    n_classes = len(le.classes_)

    tr_idx, va_idx = manual_stratified_split(train["Pitch_ID"].values, VAL_FRAC, SEED)
    print(f"split: n_train={len(tr_idx)} n_val={len(va_idx)}")

    X_val = train.loc[va_idx, feat_cols].values
    y_val = y_all[va_idx]
    val_is_rare = train.loc[va_idx, "Pitch_ID"].isin(rare_classes).values

    X_train_base = train.loc[tr_idx, feat_cols].values
    y_train_base = y_all[tr_idx]

    print(f"\n--- baseline (no augmentation), t={time.time()-t0:.0f}s ---")
    acc_base, bacc_base, pred_base = fit_eval(
        X_train_base, y_train_base, X_val, y_val, n_classes, "baseline"
    )
    rare_acc_base = accuracy_score(y_val[val_is_rare], pred_base[val_is_rare]) if val_is_rare.sum() else float("nan")
    print(f"  rare-class val subset (n={val_is_rare.sum()}): acc={rare_acc_base:.4f}")

    rare_train_rows = train.loc[tr_idx][train.loc[tr_idx, "Pitch_ID"].isin(rare_classes)]
    print(f"\n--- augmenting {len(rare_train_rows)} rare-class training rows x4 variants, t={time.time()-t0:.0f}s ---")
    aug_rows = []
    rng_master = np.random.default_rng(SEED)
    for path, pitch_id in zip(rare_train_rows["Path"], rare_train_rows["Pitch_ID"]):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        variants, sr = augment_variants(path, rng)
        for v in variants:
            feats = extract_from_signal(v, sr)
            feats["Pitch_ID"] = pitch_id
            aug_rows.append(feats)
    aug_df = pd.DataFrame(aug_rows)
    print(f"  extracted {len(aug_df)} augmented feature rows, t={time.time()-t0:.0f}s")

    X_train_aug = np.vstack([X_train_base, aug_df[feat_cols].values])
    y_train_aug = np.concatenate([y_train_base, le.transform(aug_df["Pitch_ID"].values)])

    print(f"\n--- augmented (+{len(aug_df)} rows), t={time.time()-t0:.0f}s ---")
    acc_aug, bacc_aug, pred_aug = fit_eval(
        X_train_aug, y_train_aug, X_val, y_val, n_classes, "augmented"
    )
    rare_acc_aug = accuracy_score(y_val[val_is_rare], pred_aug[val_is_rare]) if val_is_rare.sum() else float("nan")
    print(f"  rare-class val subset (n={val_is_rare.sum()}): acc={rare_acc_aug:.4f}")

    print(f"\n=== summary ===")
    print(f"overall accuracy delta (aug - base): {acc_aug - acc_base:+.4f}")
    print(f"overall balanced_accuracy delta (aug - base): {bacc_aug - bacc_base:+.4f}")
    print(f"rare-class-subset accuracy delta (aug - base): {rare_acc_aug - rare_acc_base:+.4f}")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
