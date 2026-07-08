import os
import time
import numpy as np
import pandas as pd
import librosa
import scipy.io.wavfile as wavfile

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

for i in range(5):
    path = df_train.loc[i, "Path"]
    full_path = os.path.join(data_dir, path)
    t0 = time.time()
    # Read audio
    sr, wav = wavfile.read(full_path)
    y = wav.astype(np.float32) / 32768.0
    
    # RMS Normalize
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-8:
        y = y / rms * 0.1
    else:
        y = np.zeros_like(y)
        
    # Segment (middle 5 seconds)
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    t_load = time.time() - t0
    
    # CQT
    t_cqt_start = time.time()
    cqt = np.abs(librosa.cqt(y_segment, sr=sr, fmin=librosa.note_to_hz("C1"), n_bins=84, bins_per_octave=12))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_mean = np.mean(log_cqt, axis=1)
    cqt_std = np.std(log_cqt, axis=1)
    t_cqt = time.time() - t_cqt_start
    
    # MFCC
    t_mfcc_start = time.time()
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    t_mfcc = time.time() - t_mfcc_start
    
    # Spectral Contrast
    t_contrast_start = time.time()
    contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    t_contrast = time.time() - t_contrast_start
    
    # Tonnetz
    t_tonnetz_start = time.time()
    tonnetz = librosa.feature.tonnetz(y=y_segment, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)
    t_tonnetz = time.time() - t_tonnetz_start
    
    # Spectral Flatness
    t_flat_start = time.time()
    flatness = librosa.feature.spectral_flatness(y=y_segment)
    flatness_mean = np.mean(flatness, axis=1)
    flatness_std = np.std(flatness, axis=1)
    t_flat = time.time() - t_flat_start
    
    print(f"Sample {i}: Load={t_load:.3f}s | CQT={t_cqt:.3f}s | MFCC={t_mfcc:.3f}s | Contrast={t_contrast:.3f}s | Tonnetz={t_tonnetz:.3f}s | Flatness={t_flat:.3f}s")
