import os
import time
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import correlate

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")

df = pd.read_csv(train_csv)

def extract_features_orig(path):
    full_path = os.path.join(data_dir, path)
    sr, wav = wavfile.read(full_path)
    
    # Convert to float32
    y = wav.astype(np.float32) / 32768.0
    
    # Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    # Downsample by 4
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N = len(y_ds)
    
    # 1. FFT Magnitude
    X = np.abs(np.fft.rfft(y_ds))
    freqs = np.fft.rfftfreq(N, 1/sr_ds)
    
    # 2. Log-frequency energy bins (60 Hz to 3000 Hz)
    log_freq_bins = np.logspace(np.log10(60), np.log10(3000), 121)
    log_feats = []
    for i in range(120):
        low_f = log_freq_bins[i]
        high_f = log_freq_bins[i+1]
        mask = (freqs >= low_f) & (freqs < high_f)
        if np.any(mask):
            log_feats.append(np.sum(X[mask]))
        else:
            log_feats.append(0.0)
    log_feats = np.array(log_feats)
    
    # 3. HPS features (60 Hz to 1000 Hz)
    X_log = np.log(X + 1e-6)
    candidate_freqs = np.logspace(np.log10(60), np.log10(1000), 200)
    hps_feats = []
    for f in candidate_freqs:
        val = 0.0
        for r in range(1, 6): # harmonics 1 to 5
            target_f = r * f
            mask = (freqs >= target_f * 0.98) & (freqs <= target_f * 1.02)
            if np.any(mask):
                val += np.max(X_log[mask])
            else:
                val += np.log(1e-6)
        hps_feats.append(val)
    hps_feats = np.array(hps_feats)
    
    # 4. ACF features
    acf = correlate(y_ds, y_ds, mode='full')
    center = len(y_ds) - 1
    min_lag = int(np.floor(sr_ds / 1000.0))
    max_lag = int(np.ceil(sr_ds / 60.0))
    acf_feats = acf[center + min_lag : center + max_lag + 1]
    
    target_acf_len = 174
    if len(acf_feats) < target_acf_len:
        acf_feats = np.pad(acf_feats, (0, target_acf_len - len(acf_feats)))
    else:
        acf_feats = acf_feats[:target_acf_len]
        
    return np.concatenate([log_feats, hps_feats, acf_feats])

def extract_features_opt(path):
    full_path = os.path.join(data_dir, path)
    sr, wav = wavfile.read(full_path)
    
    y = wav.astype(np.float32) / 32768.0
    
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N = len(y_ds)
    
    X = np.abs(np.fft.rfft(y_ds))
    
    # 2. Log-frequency energy bins
    log_freq_bins = np.logspace(np.log10(60), np.log10(3000), 121)
    log_feats = []
    factor = N / sr_ds
    for i in range(120):
        low_f = log_freq_bins[i]
        high_f = log_freq_bins[i+1]
        idx_start = int(np.ceil(low_f * factor))
        idx_end = int(np.ceil(high_f * factor))
        if idx_start < idx_end and idx_start < len(X):
            log_feats.append(np.sum(X[idx_start:min(idx_end, len(X))]))
        else:
            log_feats.append(0.0)
    log_feats = np.array(log_feats)
    
    # 3. HPS features
    X_log = np.log(X + 1e-6)
    candidate_freqs = np.logspace(np.log10(60), np.log10(1000), 200)
    hps_feats = []
    for f in candidate_freqs:
        val = 0.0
        for r in range(1, 6): # harmonics 1 to 5
            target_f = r * f
            idx_start = int(np.ceil(target_f * 0.98 * factor))
            idx_end = int(np.floor(target_f * 1.02 * factor)) + 1
            if idx_start < idx_end and idx_start < len(X_log):
                val += np.max(X_log[idx_start:min(idx_end, len(X_log))])
            else:
                val += np.log(1e-6)
        hps_feats.append(val)
    hps_feats = np.array(hps_feats)
    
    # 4. ACF features
    acf = correlate(y_ds, y_ds, mode='full')
    center = len(y_ds) - 1
    min_lag = int(np.floor(sr_ds / 1000.0))
    max_lag = int(np.ceil(sr_ds / 60.0))
    acf_feats = acf[center + min_lag : center + max_lag + 1]
    
    target_acf_len = 174
    if len(acf_feats) < target_acf_len:
        acf_feats = np.pad(acf_feats, (0, target_acf_len - len(acf_feats)))
    else:
        acf_feats = acf_feats[:target_acf_len]
        
    return np.concatenate([log_feats, hps_feats, acf_feats])

print("Testing equivalence...")
feats_orig = extract_features_orig(df.loc[0, 'Path'])
feats_opt = extract_features_opt(df.loc[0, 'Path'])
diff = np.max(np.abs(feats_orig - feats_opt))
print(f"Max absolute difference: {diff}")

print("Measuring original speed...")
t0 = time.time()
for i in range(5):
    extract_features_orig(df.loc[i, 'Path'])
print(f"Original: {time.time() - t0:.4f}s")

print("Measuring optimized speed...")
t0 = time.time()
for i in range(5):
    extract_features_opt(df.loc[i, 'Path'])
print(f"Optimized: {time.time() - t0:.4f}s")
