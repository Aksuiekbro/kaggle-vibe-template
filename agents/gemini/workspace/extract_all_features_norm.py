import os
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import correlate
from joblib import Parallel, delayed

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

def extract_all_normalized_features(path, data_dir):
    full_path = os.path.join(data_dir, path)
    try:
        sr, wav = wavfile.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return np.zeros(1628, dtype=np.float32)
    
    # 1. Convert to float32 and RMS normalize
    y = wav.astype(np.float32) / 32768.0
    
    # Target RMS normalization
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-8:
        y = y / rms * 0.1
    else:
        y = np.zeros_like(y)
        
    # 2. Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    # We will use y_segment for full resolution features (44100 Hz)
    # and y_ds (downsampled by 4) for downsampled features
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N_ds = len(y_ds)
    
    # Downsampled FFT
    X_ds = np.abs(np.fft.rfft(y_ds))
    
    # --- Feature family 1: Log-frequency energy bins (120 features, 60 to 3000 Hz) ---
    log_freq_bins = np.logspace(np.log10(60), np.log10(3000), 121)
    log_feats = []
    factor_ds = N_ds / sr_ds
    for i in range(120):
        low_f = log_freq_bins[i]
        high_f = log_freq_bins[i+1]
        idx_start = int(np.ceil(low_f * factor_ds))
        idx_end = int(np.ceil(high_f * factor_ds))
        if idx_start < idx_end and idx_start < len(X_ds):
            log_feats.append(np.sum(X_ds[idx_start:min(idx_end, len(X_ds))]))
        else:
            log_feats.append(0.0)
    log_feats = np.array(log_feats, dtype=np.float32)
    
    # --- Feature family 2: HPS features (200 features, 60 to 1000 Hz) ---
    X_ds_log = np.log(X_ds + 1e-6)
    candidate_freqs = np.logspace(np.log10(60), np.log10(1000), 200)
    hps_feats = []
    for f in candidate_freqs:
        val = 0.0
        for r in range(1, 6): # harmonics 1 to 5
            target_f = r * f
            idx_start = int(np.ceil(target_f * 0.98 * factor_ds))
            idx_end = int(np.floor(target_f * 1.02 * factor_ds)) + 1
            if idx_start < idx_end and idx_start < len(X_ds_log):
                val += np.max(X_ds_log[idx_start:min(idx_end, len(X_ds_log))])
            else:
                val += np.log(1e-6)
        hps_feats.append(val)
    hps_feats = np.array(hps_feats, dtype=np.float32)
    
    # --- Feature family 3: Downsampled ACF (174 features) ---
    acf_ds = correlate(y_ds, y_ds, mode='full')
    center_ds = len(y_ds) - 1
    min_lag_ds = int(np.floor(sr_ds / 1000.0))
    max_lag_ds = int(np.ceil(sr_ds / 60.0))
    acf_ds_feats = acf_ds[center_ds + min_lag_ds : center_ds + max_lag_ds + 1]
    
    # Normalize ACF by energy
    if acf_ds[center_ds] > 1e-8:
        acf_ds_feats = acf_ds_feats / acf_ds[center_ds]
    else:
        acf_ds_feats = np.zeros_like(acf_ds_feats)
        
    target_acf_len = 174
    if len(acf_ds_feats) < target_acf_len:
        acf_ds_feats = np.pad(acf_ds_feats, (0, target_acf_len - len(acf_ds_feats)))
    else:
        acf_ds_feats = acf_ds_feats[:target_acf_len]
        
    # --- Full resolution features (using y_segment at sr = 44100 Hz) ---
    N = len(y_segment)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    X = np.abs(np.fft.rfft(y_segment))
    X_log = np.log(X + 1e-6)
    
    # Autocorrelation (ACF) via FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    y_fft = np.fft.rfft(y_segment, n_fft)
    acf = np.fft.irfft(y_fft * np.conj(y_fft))[:N]
    if acf[0] > 1e-6:
        acf = acf / acf[0]
    else:
        acf = np.zeros_like(acf)
        
    # Real Cepstrum
    cep = np.fft.irfft(X_log)[:N]
    
    midi_grid = np.arange(30, 111) # 81 notes
    
    grid_harmonic_feats = []
    grid_acf_feats = []
    grid_cep_feats = []
    
    factor = N / sr
    
    for d in midi_grid:
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        
        # 1. FFT Harmonic scores (first 10 harmonics)
        harmonic_vals = []
        for r in range(1, 11):
            target_f = r * f0
            idx_start = int(np.ceil(target_f * 0.98 * factor))
            idx_end = int(np.floor(target_f * 1.02 * factor)) + 1
            idx_start = max(0, idx_start)
            idx_end = min(len(X_log), idx_end)
            if idx_start < idx_end:
                harmonic_vals.append(np.max(X_log[idx_start:idx_end]))
            else:
                harmonic_vals.append(np.log(1e-6))
        harmonic_vals = np.array(harmonic_vals)
        
        # Summarized harmonic features (6 features)
        grid_harmonic_feats.extend([
            np.sum(harmonic_vals),                      # Sum of all 10
            np.sum(harmonic_vals[0::2]),                 # Sum of odd (1, 3, 5, 7, 9)
            np.sum(harmonic_vals[1::2]),                 # Sum of even (2, 4, 6, 8, 10)
            np.sum(harmonic_vals[:3]),                    # Sum of low (1, 2, 3)
            np.sum(harmonic_vals[3:6]),                   # Sum of mid (4, 5, 6)
            np.sum(harmonic_vals[6:])                     # Sum of high (7, 8, 9, 10)
        ])
        
        # 2. ACF lag features (4 features)
        lag = sr / f0
        lag_floor = int(np.floor(lag))
        lag_ceil = int(np.ceil(lag))
        lag_round = int(np.round(lag))
        
        if 0 <= lag_floor < len(acf):
            grid_acf_feats.append(acf[lag_floor])
        else:
            grid_acf_feats.append(0.0)
            
        if 0 <= lag_ceil < len(acf):
            grid_acf_feats.append(acf[lag_ceil])
        else:
            grid_acf_feats.append(0.0)
            
        if 0 <= lag_round < len(acf):
            grid_acf_feats.append(acf[lag_round])
        else:
            grid_acf_feats.append(0.0)
            
        window_start = int(np.floor(lag - 1.5))
        window_end = int(np.ceil(lag + 1.5)) + 1
        window_start = max(0, window_start)
        window_end = min(len(acf), window_end)
        if window_start < window_end:
            grid_acf_feats.append(np.max(acf[window_start:window_end]))
        else:
            grid_acf_feats.append(0.0)
            
        # 3. Cepstrum lag features (4 features)
        if 0 <= lag_floor < len(cep):
            grid_cep_feats.append(cep[lag_floor])
        else:
            grid_cep_feats.append(0.0)
            
        if 0 <= lag_ceil < len(cep):
            grid_cep_feats.append(cep[lag_ceil])
        else:
            grid_cep_feats.append(0.0)
            
        if 0 <= lag_round < len(cep):
            grid_cep_feats.append(cep[lag_round])
        else:
            grid_cep_feats.append(0.0)
            
        window_start = int(np.floor(lag - 1.5))
        window_end = int(np.ceil(lag + 1.5)) + 1
        window_start = max(0, window_start)
        window_end = min(len(cep), window_end)
        if window_start < window_end:
            grid_cep_feats.append(np.max(cep[window_start:window_end]))
        else:
            grid_cep_feats.append(0.0)
            
    # Concatenate everything
    all_feats = np.concatenate([
        log_feats,                 # 120
        hps_feats,                 # 200
        acf_ds_feats,              # 174
        grid_harmonic_feats,       # 486
        grid_acf_feats,            # 324
        grid_cep_feats             # 324
    ])
    return all_feats

if __name__ == "__main__":
    cache_dir = "agents/gemini/workspace"
    
    print("Extracting normalized features for train...")
    t0 = time.time()
    X_train = Parallel(n_jobs=-1)(delayed(extract_all_normalized_features)(p, data_dir) for p in df_train['Path'])
    X_train = np.array(X_train, dtype=np.float32)
    print(f"Train features shape: {X_train.shape} in {time.time() - t0:.2f}s")
    
    print("Extracting normalized features for test...")
    t0 = time.time()
    X_test = Parallel(n_jobs=-1)(delayed(extract_all_normalized_features)(p, data_dir) for p in df_test['Path'])
    X_test = np.array(X_test, dtype=np.float32)
    print(f"Test features shape: {X_test.shape} in {time.time() - t0:.2f}s")
    
    np.save(os.path.join(cache_dir, "X_train_norm.npy"), X_train)
    np.save(os.path.join(cache_dir, "X_test_norm.npy"), X_test)
    print("Normalized features successfully saved to disk.")
