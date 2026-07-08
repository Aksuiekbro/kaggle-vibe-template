import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

# Let's inspect some samples
samples = df_train.sample(50, random_state=42)

for idx, row in samples.iterrows():
    path = os.path.join(data_dir, row['Path'])
    pitch_id = row['Pitch_ID']
    
    sr, wav = wavfile.read(path)
    y = wav.astype(np.float32) / 32768.0
    
    # Take FFT of the whole signal
    N = len(y)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    X = np.abs(np.fft.rfft(y))
    
    # Find peaks in the FFT spectrum
    # Normalize X
    X_norm = X / np.max(X)
    peaks, properties = find_peaks(X_norm, height=0.05, distance=10)
    peak_freqs = freqs[peaks]
    peak_amps = X_norm[peaks]
    
    # We want to find a fundamental frequency f0 such that the peak frequencies are close to multiples of f0
    # Let's search over a candidate MIDI note range
    best_midi = None
    best_score = -1e9
    
    for d in range(24, 108): # MIDI notes 24 to 107
        f0 = 440.0 * 2.0 ** ((d - 69.0) / 12.0)
        
        # Compute harmonic matching score
        # For each harmonic r, we check if there is a peak close to r * f0
        score = 0.0
        matched_harmonics = 0
        for r in range(1, 10):
            target_f = r * f0
            # Find closest peak frequency
            if len(peak_freqs) == 0:
                continue
            idx_closest = np.argmin(np.abs(peak_freqs - target_f))
            closest_f = peak_freqs[idx_closest]
            closest_amp = peak_amps[idx_closest]
            
            # Check if it is within a tolerance (e.g. 3%)
            if np.abs(closest_f - target_f) / target_f < 0.03:
                # Add peak amplitude weighted by harmonic order (e.g. 1/r or log-amplitude)
                score += closest_amp * (1.0 / r)
                matched_harmonics += 1
                
        if score > best_score:
            best_score = score
            best_midi = d
            
    print(f"Sample: {row['Path']} | Pitch_ID: {pitch_id:2d} | Best MIDI: {best_midi} (Freq: {440.0*2.0**((best_midi-69.0)/12.0):.1f} Hz) | Score: {best_score:.3f}")
