import os
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import correlate
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# We will define a dense grid of MIDI notes
# Let's cover MIDI notes from 30 to 110 (81 notes)
midi_grid = np.arange(30, 111)

def extract_grid_features(path):
    full_path = os.path.join(data_dir, path)
    try:
        sr, wav = wavfile.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return np.zeros(len(midi_grid) * 2)
    
    y = wav.astype(np.float32) / 32768.0
    
    # We will use the full sample rate (44100 Hz) to keep maximum resolution
    N = len(y)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    X = np.abs(np.fft.rfft(y))
    X_log = np.log(X + 1e-6)
    
    # Compute full resolution autocorrelation using FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    y_fft = np.fft.rfft(y, n_fft)
    acf = np.fft.irfft(y_fft * np.conj(y_fft))[:N]
    # Normalize ACF
    if acf[0] > 1e-6:
        acf = acf / acf[0]
        
    features = []
    
    for d in midi_grid:
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        
        # 1. FFT Harmonic score (sum of log-magnitude at first 8 harmonics)
        harmonic_vals = []
        for r in range(1, 9):
            target_f = r * f0
            # Find the max in a small window around target_f
            idx_start = int(np.ceil(target_f * 0.98 * N / sr))
            idx_end = int(np.floor(target_f * 1.02 * N / sr)) + 1
            idx_start = max(0, idx_start)
            idx_end = min(len(X_log), idx_end)
            if idx_start < idx_end:
                harmonic_vals.append(np.max(X_log[idx_start:idx_end]))
            else:
                harmonic_vals.append(np.log(1e-6))
        features.append(np.sum(harmonic_vals))
        
        # 2. ACF lag score (value at the lag of period)
        lag = sr / f0
        lag_int = int(np.round(lag))
        if 0 <= lag_int < len(acf):
            features.append(acf[lag_int])
        else:
            features.append(0.0)
            
    return np.array(features, dtype=np.float32)

print("Extracting grid features for train...")
t0 = time.time()
X_train = Parallel(n_jobs=-1)(delayed(extract_grid_features)(p) for p in df_train['Path'])
X_train = np.array(X_train)
y_train = df_train['Pitch_ID'].values
print(f"Train features shape: {X_train.shape} in {time.time() - t0:.2f}s")

print("Extracting grid features for test...")
t0 = time.time()
X_test = Parallel(n_jobs=-1)(delayed(extract_grid_features)(p) for p in df_test['Path'])
X_test = np.array(X_test)
print(f"Test features shape: {X_test.shape} in {time.time() - t0:.2f}s")

# Let's save these features
cache_dir = "agents/gemini/workspace"
np.save(os.path.join(cache_dir, "X_train_grid.npy"), X_train)
np.save(os.path.join(cache_dir, "X_test_grid.npy"), X_test)
print("Features saved.")

# Train a simple model using Stratified 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
    X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]
    
    # Train a simple MLP classifier
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.002,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} MLP Accuracy: {acc:.5f}")

print(f"Mean CV Accuracy: {np.mean(cv_scores):.5f} (std: {np.std(cv_scores):.5f})")
