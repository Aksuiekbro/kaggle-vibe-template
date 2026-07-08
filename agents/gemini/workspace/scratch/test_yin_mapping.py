import os
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

mapping_df = pd.read_csv("agents/gemini/workspace/pitch_yin_mapping.csv")

def predict_sample(path):
    full_path = os.path.join(data_dir, path)
    try:
        y, sr = librosa.load(full_path, sr=None, mono=True)
        # Extract middle 5s
        duration = len(y) / sr
        start_sec = max(0.0, (duration - 5.0) / 2.0)
        y_seg = y[int(start_sec*sr):int((start_sec+5.0)*sr)]
        
        # Estimate pitch track
        pitch_track = librosa.yin(y_seg, fmin=50.0, fmax=2000.0, sr=sr)
        pitch_track = pitch_track[np.isfinite(pitch_track) & (pitch_track > 0)]
        
        if len(pitch_track) > 0:
            est_freq = np.median(pitch_track)
            est_midi = 12 * np.log2(est_freq / 440.0) + 69
        else:
            est_midi = np.nan
    except Exception as e:
        est_midi = np.nan
        
    if np.isnan(est_midi):
        # Default fallback
        return 0
        
    # Find closest Pitch_ID
    diffs = np.abs(mapping_df['median_midi'] - est_midi)
    closest_idx = np.argmin(diffs)
    return mapping_df.loc[closest_idx, 'Pitch_ID']

# Run in parallel on a subset or all of train.csv to see the accuracy
print("Running YIN rule-based classifier on train.csv...")
import time
t0 = time.time()
preds = Parallel(n_jobs=-1)(delayed(predict_sample)(p) for p in df_train['Path'])
acc = accuracy_score(df_train['Pitch_ID'], preds)
print(f"Accuracy: {acc:.5f} in {time.time() - t0:.2f}s")
