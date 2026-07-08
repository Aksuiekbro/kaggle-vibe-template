import os
import time
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

def estimate_pitch_hps(y, sr):
    N = len(y)
    X = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    
    hps = np.copy(X)
    for r in range(2, 6):
        decimated = X[::r]
        hps[:len(decimated)] *= decimated
        hps[len(decimated):] = 0.0
        
    mask = (freqs >= 50.0) & (freqs <= 2000.0)
    if not np.any(mask):
        return 0.0
    best_idx = np.argmax(hps * mask)
    return freqs[best_idx]

def extract_pitch_feats(path):
    full_path = os.path.join(data_dir, path)
    try:
        y, sr = librosa.load(full_path, sr=None, mono=True)
        # Extract middle 5s
        duration = len(y) / sr
        start_sec = max(0.0, (duration - 5.0) / 2.0)
        y_seg = y[int(start_sec*sr):int((start_sec+5.0)*sr)]
        
        # 1. YIN pitch
        pitch_track = librosa.yin(y_seg, fmin=50.0, fmax=2000.0, sr=sr)
        valid_track = pitch_track[np.isfinite(pitch_track) & (pitch_track > 0)]
        
        if len(valid_track) > 0:
            yin_freq = np.median(valid_track)
            yin_midi = 12 * np.log2(yin_freq / 440.0) + 69
            yin_std = np.std(valid_track)
        else:
            yin_freq = 0.0
            yin_midi = 0.0
            yin_std = -1.0
            
        # 2. HPS pitch
        hps_freq = estimate_pitch_hps(y_seg, sr)
        if hps_freq > 0:
            hps_midi = 12 * np.log2(hps_freq / 440.0) + 69
        else:
            hps_midi = 0.0
            
        # 3. Combined features
        diff_midi = abs(yin_midi - hps_midi) if (yin_midi > 0 and hps_midi > 0) else -1.0
        
    except Exception as e:
        yin_freq = 0.0
        yin_midi = 0.0
        yin_std = -1.0
        hps_freq = 0.0
        hps_midi = 0.0
        diff_midi = -1.0
        
    return np.array([yin_freq, yin_midi, yin_std, hps_freq, hps_midi, diff_midi], dtype=np.float32)

if __name__ == "__main__":
    cache_dir = "agents/gemini/workspace"
    
    print("Extracting pitch features for train...")
    t0 = time.time()
    train_feats = Parallel(n_jobs=-1)(delayed(extract_pitch_feats)(p) for p in df_train['Path'])
    train_feats = np.array(train_feats, dtype=np.float32)
    print(f"Train pitch features shape: {train_feats.shape} in {time.time() - t0:.2f}s")
    
    print("Extracting pitch features for test...")
    t0 = time.time()
    test_feats = Parallel(n_jobs=-1)(delayed(extract_pitch_feats)(p) for p in df_test['Path'])
    test_feats = np.array(test_feats, dtype=np.float32)
    print(f"Test pitch features shape: {test_feats.shape} in {time.time() - t0:.2f}s")
    
    np.save(os.path.join(cache_dir, "X_train_pitch_only.npy"), train_feats)
    np.save(os.path.join(cache_dir, "X_test_pitch_only.npy"), test_feats)
    print("Pitch features successfully saved to disk.")
