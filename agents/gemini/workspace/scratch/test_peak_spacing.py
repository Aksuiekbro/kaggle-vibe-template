import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

# Let's inspect 10 random samples
samples = df_train.sample(10, random_state=42)

for idx, row in samples.iterrows():
    path = os.path.join(data_dir, row['Path'])
    pitch_id = row['Pitch_ID']
    
    sr, wav = wavfile.read(path)
    y = wav.astype(np.float32) / 32768.0
    
    # Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    # FFT
    N = len(y_segment)
    X = np.abs(np.fft.rfft(y_segment))
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    
    # Find peaks
    X_norm = X / np.max(X)
    peaks, _ = find_peaks(X_norm, height=0.1, distance=20)
    peak_freqs = freqs[peaks]
    
    # Sort peak frequencies
    peak_freqs = np.sort(peak_freqs)
    
    # Compute differences
    diffs = np.diff(peak_freqs)
    
    print(f"\nPath: {row['Path']} | Pitch_ID: {pitch_id}")
    print(f"Top 8 peak frequencies: {peak_freqs[:8]}")
    print(f"Peak differences: {diffs[:7]}")
