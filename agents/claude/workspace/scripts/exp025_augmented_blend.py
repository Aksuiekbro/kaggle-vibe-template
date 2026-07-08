"""exp_025 (own): combine exp_018's fold-safe rare-tail-class augmentation
with exp_017's LGBM+MLP soft-vote blend, evaluated via proper 3-fold
StratifiedKFold CV.

Rationale: exp_018 showed augmentation is a real, grounded fix (more data for
under-represented classes) that improved LGBM-alone OOF acc 0.9090->0.9185
(+0.0094). The current local-CV-best (sub_022, 0.9476) is an LGBM+MLP blend
that scored WORSE on LB than the LGBM-only sub_020 (0.87931 vs 0.91954) --
per STRATEGY.md/PLAN.md, further blend-only CV gains are under suspicion.
This experiment tests whether augmentation (a feature/data-level fix, not an
ensemble-weight trick) stacks with the existing blend, which would be a more
trustworthy way to beat 0.9476 than further blend-weight tuning.

Augmentation happens strictly inside each fold's TRAIN split only (same
leakage-safe design as exp_018).
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from extract_grid_features import DATA_ROOT, MIDI_NOTES, N_HARMONICS, TOL, TARGET_RMS

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

    classes = np.arange(n_classes)
    oof_lgb = np.zeros((len(y_all), n_classes))
    oof_mlp = np.zeros((len(y_all), n_classes))
    pred_lgb = np.zeros(len(y_all), dtype=int)
    pred_mlp = np.zeros(len(y_all), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        t_fold = time.time()
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        tr_rows = train.iloc[tr_idx]
        rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
        t_aug = time.time()
        aug_df = augment_rows(rare_tr_rows, rng_master)
        print(f"fold {fold}: augmented {len(rare_tr_rows)} rare rows -> "
              f"{len(aug_df)} extra rows ({time.time()-t_aug:.0f}s)", flush=True)

        X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
        y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

        t_lgb = time.time()
        clf_lgb = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
        clf_lgb.fit(X_tr_aug, y_tr_aug)
        proba_lgb = clf_lgb.predict_proba(X_va)
        fold_classes_lgb = clf_lgb.classes_
        full_proba_lgb = np.zeros((len(va_idx), n_classes))
        full_proba_lgb[:, fold_classes_lgb] = proba_lgb
        oof_lgb[va_idx] = full_proba_lgb
        pred_lgb[va_idx] = fold_classes_lgb[np.argmax(proba_lgb, axis=1)]
        acc_lgb_fold = accuracy_score(y_va, pred_lgb[va_idx])
        print(f"fold {fold} LGBM(aug): acc={acc_lgb_fold:.4f} ({time.time()-t_lgb:.0f}s)", flush=True)

        t_mlp = time.time()
        sc = StandardScaler().fit(X_tr_aug)
        Xtr_s, Xva_s = sc.transform(X_tr_aug), sc.transform(X_va)
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
            early_stopping=True, random_state=42,
        )
        clf_mlp.fit(Xtr_s, y_tr_aug)
        proba_mlp = clf_mlp.predict_proba(Xva_s)
        fold_classes_mlp = clf_mlp.classes_
        full_proba_mlp = np.zeros((len(va_idx), n_classes))
        full_proba_mlp[:, fold_classes_mlp] = proba_mlp
        oof_mlp[va_idx] = full_proba_mlp
        pred_mlp[va_idx] = fold_classes_mlp[np.argmax(proba_mlp, axis=1)]
        acc_mlp_fold = accuracy_score(y_va, pred_mlp[va_idx])
        print(f"fold {fold} MLP(aug): acc={acc_mlp_fold:.4f} ({time.time()-t_mlp:.0f}s), "
              f"total fold time {time.time()-t_fold:.0f}s", flush=True)

    acc_lgb_oof = accuracy_score(y_all, pred_lgb)
    bacc_lgb_oof = balanced_accuracy_score(y_all, pred_lgb)
    acc_mlp_oof = accuracy_score(y_all, pred_mlp)
    bacc_mlp_oof = balanced_accuracy_score(y_all, pred_mlp)

    best_w, best_acc = None, -1
    for w in np.arange(0.0, 1.01, 0.1):
        blend_proba = w * oof_lgb + (1 - w) * oof_mlp
        blend_pred = classes[np.argmax(blend_proba, axis=1)]
        acc = accuracy_score(y_all, blend_pred)
        print(f"blend w_lgb={w:.1f}: acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_w = acc, w

    best_blend_pred = classes[np.argmax(best_w * oof_lgb + (1 - best_w) * oof_mlp, axis=1)]
    best_blend_bacc = balanced_accuracy_score(y_all, best_blend_pred)

    print("\n=== exp_025 summary (augmented LGBM+MLP blend, full 3-fold CV) ===", flush=True)
    print(f"LGBM(aug)-only OOF acc={acc_lgb_oof:.4f} balanced_acc={bacc_lgb_oof:.4f}", flush=True)
    print(f"MLP(aug)-only  OOF acc={acc_mlp_oof:.4f} balanced_acc={bacc_mlp_oof:.4f}", flush=True)
    print(f"Best blend: w_lgb={best_w:.1f}, acc={best_acc:.4f}, balanced_acc={best_blend_bacc:.4f}", flush=True)
    print(f"reference -- exp_017 (no augment) blend best acc was 0.9476 (w_lgb=0.4)", flush=True)
    print(f"delta vs exp_017 blend: {best_acc - 0.9476:+.4f}", flush=True)
    print(f"total time: {time.time()-t0:.0f}s", flush=True)

    np.save(CACHE_DIR + "oof_lgb_exp025.npy", oof_lgb)
    np.save(CACHE_DIR + "oof_mlp_exp025.npy", oof_mlp)
    np.save(CACHE_DIR + "y_exp025.npy", y_all)


if __name__ == "__main__":
    main()
