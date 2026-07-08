import os
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from joblib import Parallel, delayed

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Cover MIDI notes from 30 to 110 (81 notes)
midi_grid = np.arange(30, 111)

def extract_enhanced_grid_features(path):
    full_path = os.path.join(data_dir, path)
    try:
        sr, wav = wavfile.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return np.zeros(len(midi_grid) * 10, dtype=np.float32)
    
    y = wav.astype(np.float32) / 32768.0
    N = len(y)
    
    # FFT and log-magnitude
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    X = np.abs(np.fft.rfft(y))
    X_log = np.log(X + 1e-6)
    
    # Autocorrelation (ACF) via FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    y_fft = np.fft.rfft(y, n_fft)
    acf = np.fft.irfft(y_fft * np.conj(y_fft))[:N]
    if acf[0] > 1e-6:
        acf = acf / acf[0]
        
    features = []
    
    for d in midi_grid:
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        
        # 1. FFT Harmonic scores (first 10 harmonics)
        harmonic_vals = []
        for r in range(1, 11):
            target_f = r * f0
            idx_start = int(np.ceil(target_f * 0.98 * N / sr))
            idx_end = int(np.floor(target_f * 1.02 * N / sr)) + 1
            idx_start = max(0, idx_start)
            idx_end = min(len(X_log), idx_end)
            if idx_start < idx_end:
                harmonic_vals.append(np.max(X_log[idx_start:idx_end]))
            else:
                harmonic_vals.append(np.log(1e-6))
        harmonic_vals = np.array(harmonic_vals)
        
        # Add summarized harmonic features
        features.append(np.sum(harmonic_vals))                      # Sum of all 10
        features.append(np.sum(harmonic_vals[0::2]))                 # Sum of odd (1, 3, 5, 7, 9)
        features.append(np.sum(harmonic_vals[1::2]))                 # Sum of even (2, 4, 6, 8, 10)
        features.append(np.sum(harmonic_vals[:3]))                    # Sum of low (1, 2, 3)
        features.append(np.sum(harmonic_vals[3:6]))                   # Sum of mid (4, 5, 6)
        features.append(np.sum(harmonic_vals[6:]))                    # Sum of high (7, 8, 9, 10)
        
        # 2. ACF lag features
        lag = sr / f0
        lag_floor = int(np.floor(lag))
        lag_ceil = int(np.ceil(lag))
        lag_round = int(np.round(lag))
        
        # Floor lag value
        if 0 <= lag_floor < len(acf):
            features.append(acf[lag_floor])
        else:
            features.append(0.0)
            
        # Ceil lag value
        if 0 <= lag_ceil < len(acf):
            features.append(acf[lag_ceil])
        else:
            features.append(0.0)
            
        # Round lag value
        if 0 <= lag_round < len(acf):
            features.append(acf[lag_round])
        else:
            features.append(0.0)
            
        # Max value in a small lag window [lag - 1.5, lag + 1.5]
        window_start = int(np.floor(lag - 1.5))
        window_end = int(np.ceil(lag + 1.5)) + 1
        window_start = max(0, window_start)
        window_end = min(len(acf), window_end)
        if window_start < window_end:
            features.append(np.max(acf[window_start:window_end]))
        else:
            features.append(0.0)
            
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    cache_dir = "agents/gemini/workspace"
    
    print("Extracting enhanced grid features for train...")
    t0 = time.time()
    X_train = Parallel(n_jobs=-1)(delayed(extract_enhanced_grid_features)(p) for p in df_train['Path'])
    X_train = np.array(X_train)
    print(f"Train features shape: {X_train.shape} in {time.time() - t0:.2f}s")
    
    print("Extracting enhanced grid features for test...")
    t0 = time.time()
    X_test = Parallel(n_jobs=-1)(delayed(extract_enhanced_grid_features)(p) for p in df_test['Path'])
    X_test = np.array(X_test)
    print(f"Test features shape: {X_test.shape} in {time.time() - t0:.2f}s")
    
    np.save(os.path.join(cache_dir, "X_train_enhanced_grid.npy"), X_train)
    np.save(os.path.join(cache_dir, "X_test_enhanced_grid.npy"), X_test)
    print("Enhanced grid features saved to disk successfully.")
