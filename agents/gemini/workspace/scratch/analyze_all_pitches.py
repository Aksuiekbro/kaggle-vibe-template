import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import correlate
from collections import Counter

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

# We want to find for each Pitch_ID, what are the estimated MIDI notes or frequencies.
# We will use the full sample rate (44100 Hz) and no downsampling, or less downsampling, to get high resolution.
results = {}

for idx, row in df_train.iterrows():
    path = os.path.join(data_dir, row['Path'])
    pitch_id = row['Pitch_ID']
    try:
        sr, wav = wavfile.read(path)
    except Exception as e:
        continue
    y = wav.astype(np.float32) / 32768.0
    
    # Use full resolution autocorrelation
    # We only need the middle portion
    center = len(y) - 1
    # Check lags corresponding to 50 Hz to 2000 Hz
    min_lag = int(sr / 2000.0)
    max_lag = int(sr / 50.0)
    
    # Let's compute a fast autocorrelation using FFT
    n_fft = 2 ** int(np.ceil(np.log2(2 * len(y) - 1)))
    y_fft = np.fft.rfft(y, n_fft)
    acf = np.fft.irfft(y_fft * np.conj(y_fft))[:len(y)]
    
    # We want to find the first major peak (using YIN-like difference function or simple peak picking)
    # Let's just find the max in acf within [min_lag, max_lag]
    acf_part = acf[min_lag:max_lag+1]
    best_lag = min_lag + np.argmax(acf_part)
    freq = sr / best_lag
    midi = 12 * np.log2(freq / 440.0) + 69
    
    if pitch_id not in results:
        results[pitch_id] = []
    results[pitch_id].append(midi)

print("Pitch_ID | Median Est MIDI | Rounded | Count | Std")
print("-" * 50)
for pid in sorted(results.keys()):
    midis = results[pid]
    median_midi = np.median(midis)
    rounded = int(np.round(median_midi))
    std = np.std(midis)
    print(f"{pid:8d} | {median_midi:15.2f} | {rounded:7d} | {len(midis):5d} | {std:5.2f}")
