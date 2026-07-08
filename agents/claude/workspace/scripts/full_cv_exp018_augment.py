"""exp_018 full CV: fold-safe rare-tail-class (<10 samples/class, 14/82
classes) additive-noise + time-stretch augmentation, evaluated via proper
3-fold StratifiedKFold CV (not the probe's single 80/20 holdout).

Augmentation happens strictly inside each fold's TRAIN split only -- augmented
copies of a rare clip never appear in that fold's validation split, so there
is no leakage across folds. No pitch-shift (would change the ground-truth
label).

Reuses the same feature extraction as probe_exp018_rare_augment.py and the
same LGBM config as exp_005/exp_010's full baseline (OOF ~0.9086/0.9090) for
an apples-to-apples comparison.
"""
import sys
import time
import warnings

import lightgbm as lgb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from extract_grid_features import DATA_ROOT, MIDI_NOTES, N_HARMONICS, TOL, TARGET_RMS

warnings.filterwarnings("ignore")

CACHE_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
RARE_THRESHOLD = 10
SEED = 42
N_SPLITS = 3

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)


def extract_from_signal(y, sr):
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


def augment_rows(rows_df, rng_master):
    aug_rows = []
    for path, pitch_id in zip(rows_df["Path"], rows_df["Pitch_ID"]):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        variants, sr = augment_variants(path, rng)
        for v in variants:
            feats = extract_from_signal(v, sr)
            feats["Pitch_ID"] = pitch_id
            aug_rows.append(feats)
    return pd.DataFrame(aug_rows)


def main():
    t0 = time.time()
    full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()
    rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
    print(f"rare classes (<{RARE_THRESHOLD} samples): {len(rare_classes)}/{len(full_counts)}", flush=True)

    train = pd.read_parquet(CACHE_DIR + "train_grid_features.parquet")
    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    le = LabelEncoder()
    y_all = le.fit_transform(train["Pitch_ID"].values)
    n_classes = len(le.classes_)
    X_all = train[feat_cols].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng_master = np.random.default_rng(SEED)

    pred_base = np.zeros(len(y_all), dtype=int)
    pred_aug = np.zeros(len(y_all), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        t_fold = time.time()
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        clf_base = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
        clf_base.fit(X_tr, y_tr)
        fold_classes = clf_base.classes_
        p = clf_base.predict(X_va)
        pred_base[va_idx] = p
        acc_base = accuracy_score(y_va, p)
        print(f"fold {fold} baseline: acc={acc_base:.4f} ({time.time()-t_fold:.0f}s)", flush=True)

        tr_rows = train.iloc[tr_idx]
        rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
        t_aug = time.time()
        aug_df = augment_rows(rare_tr_rows, rng_master)
        print(f"fold {fold}: augmented {len(rare_tr_rows)} rare rows -> "
              f"{len(aug_df)} extra rows ({time.time()-t_aug:.0f}s)", flush=True)

        X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
        y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

        t_fit = time.time()
        clf_aug = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
        clf_aug.fit(X_tr_aug, y_tr_aug)
        p_aug = clf_aug.predict(X_va)
        pred_aug[va_idx] = p_aug
        acc_aug_fold = accuracy_score(y_va, p_aug)
        print(f"fold {fold} augmented: acc={acc_aug_fold:.4f} ({time.time()-t_fit:.0f}s), "
              f"total fold time {time.time()-t_fold:.0f}s", flush=True)

    acc_base_oof = accuracy_score(y_all, pred_base)
    bacc_base_oof = balanced_accuracy_score(y_all, pred_base)
    acc_aug_oof = accuracy_score(y_all, pred_aug)
    bacc_aug_oof = balanced_accuracy_score(y_all, pred_aug)

    is_rare = train["Pitch_ID"].isin(rare_classes).values
    rare_acc_base = accuracy_score(y_all[is_rare], pred_base[is_rare])
    rare_acc_aug = accuracy_score(y_all[is_rare], pred_aug[is_rare])

    print("\n=== full 3-fold CV summary ===", flush=True)
    print(f"baseline OOF acc={acc_base_oof:.4f} balanced_acc={bacc_base_oof:.4f}", flush=True)
    print(f"augmented OOF acc={acc_aug_oof:.4f} balanced_acc={bacc_aug_oof:.4f}", flush=True)
    print(f"delta acc: {acc_aug_oof - acc_base_oof:+.4f}", flush=True)
    print(f"delta balanced_acc: {bacc_aug_oof - bacc_base_oof:+.4f}", flush=True)
    print(f"rare-class subset acc: base={rare_acc_base:.4f} aug={rare_acc_aug:.4f} "
          f"delta={rare_acc_aug - rare_acc_base:+.4f}", flush=True)
    print(f"total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
