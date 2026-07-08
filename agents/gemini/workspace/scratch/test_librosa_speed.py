import os
import time
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
import pandas as pd
df = pd.read_csv(train_csv)

def test_extract(path):
    full_path = os.path.join(data_dir, path)
    sr, wav = wavfile.read(full_path)
    y = wav.astype(np.float32) / 32768.0
    
    # Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    # Downsample by 4 for fast baseline
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N = len(y_ds)
    X = np.abs(np.fft.rfft(y_ds))
    
    # Log-frequency bins
    log_freq_bins = np.logspace(np.log10(60), np.log10(3000), 121)
    log_feats = np.zeros(120, dtype=np.float32)
    scale = N / sr_ds
    for i in range(120):
        low_f = log_freq_bins[i]
        high_f = log_freq_bins[i+1]
        idx_start = int(np.ceil(low_f * scale))
        idx_end = int(np.ceil(high_f * scale))
        if idx_start < idx_end and idx_start < len(X):
            log_feats[i] = np.sum(X[idx_start:idx_end])
            
    # HPS
    X_log = np.log(X + 1e-6)
    candidate_freqs = np.logspace(np.log10(60), np.log10(1000), 200)
    hps_feats = np.zeros(200, dtype=np.float32)
    for i, f in enumerate(candidate_freqs):
        val = 0.0
        for r in range(1, 6):
            target_f = r * f
            idx_start = int(np.ceil(target_f * 0.98 * scale))
            idx_end = int(np.floor(target_f * 1.02 * scale)) + 1
            idx_start = max(0, idx_start)
            idx_end = min(len(X_log), idx_end)
            if idx_start < idx_end:
                val += np.max(X_log[idx_start:idx_end])
            else:
                val += np.log(1e-6)
        hps_feats[i] = val
        
    # ACF
    min_lag = int(np.floor(sr_ds / 1000.0))
    max_lag = int(np.ceil(sr_ds / 60.0))
    acf_feats = np.array([np.dot(y_ds[:-lag], y_ds[lag:]) for lag in range(min_lag, max_lag + 1)], dtype=np.float32)
    target_acf_len = 174
    if len(acf_feats) < target_acf_len:
        acf_feats = np.pad(acf_feats, (0, target_acf_len - len(acf_feats)))
    else:
        acf_feats = acf_feats[:target_acf_len]
        
    # Now librosa features on full sample rate segment for frequency range
    t_start = time.time()
    
    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # 2. MFCCs
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    t_end = time.time()
    print(f"Librosa features extraction took {t_end - t_start:.4f}s")
    
    return np.concatenate([log_feats, hps_feats, acf_feats, chroma_mean, chroma_std, mfcc_mean, mfcc_std])

t0 = time.time()
for i in range(5):
    feats = test_extract(df.loc[i, 'Path'])
    print(f"File {i} total features: {feats.shape}")
print(f"Total time for 5 files: {time.time() - t0:.4f}s")
