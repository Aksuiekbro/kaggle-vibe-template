import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import correlate

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

print(df_train.head(10))

# Let's inspect some samples for class 52, 64, 6
for pitch_id in [6, 20, 52, 64]:
    samples = df_train[df_train['Pitch_ID'] == pitch_id].head(3)
    print(f"\n--- Pitch_ID: {pitch_id} ---")
    for idx, row in samples.iterrows():
        path = os.path.join(data_dir, row['Path'])
        sr, wav = wavfile.read(path)
        y = wav.astype(np.float32) / 32768.0
        
        # Compute autocorrelation to find the main period
        # Downsample for speed
        y_ds = y[::4]
        sr_ds = sr / 4.0
        acf = correlate(y_ds, y_ds, mode='full')
        center = len(y_ds) - 1
        
        # Look for the peak in the range 50 Hz to 2000 Hz
        min_lag = int(sr_ds / 2000.0)
        max_lag = int(sr_ds / 50.0)
        acf_part = acf[center + min_lag : center + max_lag + 1]
        best_lag = min_lag + np.argmax(acf_part)
        freq = sr_ds / best_lag
        
        # Let's also estimate MIDI note: 12 * log2(freq / 440) + 69
        midi_est = 12 * np.log2(freq / 440.0) + 69
        print(f"Sample {row['Path']}: Est Freq: {freq:.2f} Hz, Est MIDI: {midi_est:.2f}")
