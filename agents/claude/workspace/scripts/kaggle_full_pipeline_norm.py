"""exp_004 full-fidelity run: same pipeline as kaggle_full_pipeline.py, but with
per-clip RMS normalization (target RMS 0.1) applied before feature extraction --
mitigation for the exp_003 full finding (train-vs-test adversarial AUC 0.9984,
driven by contrast/zcr/mfcc features that read as recording-level loudness).

Runs in one kernel session:
  1. Extract the same feature set as kaggle_full_pipeline.py, but on RMS-normalized audio.
  2. exp_003-norm: adversarial validation (train vs test AUC) on normalized features --
     check whether normalization actually closes the shift.
  3. exp_002-norm: StratifiedKFold(3) LightGBM multiclass CV (full_delta signal for exp_004).
  4. Fit on full train, predict test, write submission.csv + metrics.json.
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
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
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
N_MFCC = 20
N_WORKERS = max(1, os.cpu_count())
TARGET_RMS = 0.1


def extract_one(path):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    y = y / rms * TARGET_RMS

    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i}_mean"] = mfcc[i].mean()
        feats[f"mfcc_{i}_std"] = mfcc[i].std()

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_stft_{i}_mean"] = chroma_stft[i].mean()
        feats[f"chroma_stft_{i}_std"] = chroma_stft[i].std()

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_cqt_{i}_mean"] = chroma_cqt[i].mean()
        feats[f"chroma_cqt_{i}_std"] = chroma_cqt[i].std()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"contrast_{i}_mean"] = contrast[i].mean()
        feats[f"contrast_{i}_std"] = contrast[i].std()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = centroid.mean()
    feats["centroid_std"] = centroid.std()

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = bandwidth.mean()
    feats["bandwidth_std"] = bandwidth.std()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean"] = rolloff.mean()
    feats["rolloff_std"] = rolloff.std()

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    feats["flatness_mean"] = flatness.mean()
    feats["flatness_std"] = flatness.std()

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    feats["zcr_mean"] = zcr.mean()
    feats["zcr_std"] = zcr.std()

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        feats[f"tonnetz_{i}_mean"] = tonnetz[i].mean()
        feats[f"tonnetz_{i}_std"] = tonnetz[i].std()

    S = np.abs(librosa.stft(y))
    log_S = np.log1p(S.mean(axis=1))
    spec_acf = np.correlate(log_S, log_S, mode="full")[len(log_S) - 1:]
    spec_acf = spec_acf / (spec_acf[0] + 1e-8)
    for lag in [5, 10, 20, 30, 50, 80]:
        feats[f"spec_acf_lag{lag}"] = spec_acf[lag] if lag < len(spec_acf) else 0.0

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
    train = extract_all(DATA_ROOT + "kaggle_dataset/train.csv", True, "train_features_norm.parquet")
    test = extract_all(DATA_ROOT + "kaggle_dataset/test.csv", False, "test_features_norm.parquet")

    feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]
    X_train = train[feat_cols].values
    X_test = test[feat_cols].values

    # exp_003-norm: adversarial validation train vs test, on normalized features
    Xa = np.vstack([X_train, X_test])
    ya = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    skf_a = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_adv = np.zeros(len(ya))
    for tr_idx, va_idx in skf_a.split(Xa, ya):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=-1, n_jobs=N_WORKERS)
        clf.fit(Xa[tr_idx], ya[tr_idx])
        oof_adv[va_idx] = clf.predict_proba(Xa[va_idx])[:, 1]
    adv_auc = roc_auc_score(ya, oof_adv)
    clf_adv_full = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1,
                                       n_jobs=N_WORKERS).fit(Xa, ya)
    imp = pd.Series(clf_adv_full.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"exp_003-norm full adversarial AUC (vs exp_003 full 0.9984 unnormalized): {adv_auc:.4f}", flush=True)
    print(imp.head(15), flush=True)

    # exp_002-norm: LGBM multiclass CV on normalized features
    y_raw = train["Pitch_ID"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_counts = pd.Series(y_raw).value_counts()
    print(f"n={len(train)} classes={len(class_counts)} "
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
    print(f"exp_002-norm full OOF accuracy (vs exp_002 full 0.5682 unnormalized): {overall_acc:.4f}", flush=True)
    print(f"fold mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}", flush=True)

    # fit on all train, predict test, write submission
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
        "exp_002_norm_full_oof_acc": overall_acc,
        "exp_002_norm_fold_accs": fold_accs,
        "exp_002_unnormalized_full_oof_acc": 0.5682,
        "exp_004_full_delta_vs_unnormalized": overall_acc - 0.5682,
        "exp_003_norm_full_adv_auc": adv_auc,
        "exp_003_unnormalized_full_adv_auc": 0.9984,
        "exp_003_norm_top_features": imp.head(15).to_dict(),
        "min_per_class": int(class_counts.min()),
        "max_per_class": int(class_counts.max()),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
