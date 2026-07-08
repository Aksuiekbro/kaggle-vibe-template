import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

def estimate_pitch_hps(y, sr):
    # Compute HPS
    N = len(y)
    X = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    
    # We only care about frequencies in [50, 2000]
    # Downsample the spectrum by factor 2, 3, 4, 5 and multiply
    # To do this safely, we can interpolate the spectrum
    hps = np.copy(X)
    for r in range(2, 6):
        # Downsample by r
        downsampled = np.zeros_like(X)
        for i in range(len(X)):
            idx = int(i * r)
            if idx < len(X):
                downsampled[i] = X[idx]
        hps *= downsampled
        
    # Find max in [50, 2000] range
    mask = (freqs >= 50.0) & (freqs <= 2000.0)
    if not np.any(mask):
        return np.nan
    best_idx = np.argmax(hps * mask)
    return freqs[best_idx]

print("Estimating pitch for all training samples...")
est_pitches = []
for idx, row in df_train.iterrows():
    path = os.path.join(data_dir, row['Path'])
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Extract middle 5s
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    y_segment = y[int(start_sec*sr):int((start_sec+5.0)*sr)]
    
    f_yin_track = librosa.yin(y_segment, fmin=50.0, fmax=2000.0, sr=sr)
    f_yin = np.median(f_yin_track[np.isfinite(f_yin_track) & (f_yin_track > 0)])
    
    f_hps = estimate_pitch_hps(y_segment, sr)
    
    est_pitches.append({
        'Pitch_ID': row['Pitch_ID'],
        'Path': row['Path'],
        'yin': f_yin,
        'hps': f_hps
    })
    
    if idx > 0 and idx % 200 == 0:
        print(f"Processed {idx}/{len(df_train)}...")

df_est = pd.DataFrame(est_pitches)

# Let's see if we can combine yin and hps
# Sometimes HPS is correct when YIN has octave error, and vice versa.
# Since both are estimating the fundamental frequency, the ratio of their estimates should be close to an integer or 1/integer.
# Let's print stats for each Pitch_ID
summary = []
for pid in sorted(df_est['Pitch_ID'].unique()):
    sub = df_est[df_est['Pitch_ID'] == pid]
    
    # Let's print the individual sample estimates to see if we can resolve them
    print(f"\nPitch_ID {pid}:")
    for i, r in sub.head(10).iterrows():
        print(f"  {r['Path']}: yin={r['yin']:.2f} Hz (MIDI {12*np.log2(r['yin']/440)+69:.1f}), hps={r['hps']:.2f} Hz (MIDI {12*np.log2(r['hps']/440)+69:.1f})")
