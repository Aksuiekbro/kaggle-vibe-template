import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
df_train = pd.read_csv(train_csv)

# Let's estimate the pitch for each training sample using librosa.yin
# We will use fmin=50 and fmax=2000 as bounds
print("Running librosa.yin on all train samples...")

pitches = []
for idx, row in df_train.iterrows():
    path = os.path.join(data_dir, row['Path'])
    pitch_id = row['Pitch_ID']
    
    # Load wav
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Run YIN pitch estimation on the whole file
    # We want a single pitch estimate for the file, so we can take the median of the estimated pitch track.
    fmin = 50.0
    fmax = 2000.0
    
    pitch_track = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    # Filter out nan or zero values (YIN returns estimates)
    pitch_track = pitch_track[np.isfinite(pitch_track) & (pitch_track > 0)]
    
    if len(pitch_track) > 0:
        est_freq = np.median(pitch_track)
        est_midi = 12 * np.log2(est_freq / 440.0) + 69
    else:
        est_freq = np.nan
        est_midi = np.nan
        
    pitches.append({
        'Pitch_ID': pitch_id,
        'Path': row['Path'],
        'Freq': est_freq,
        'MIDI': est_midi
    })
    
    if idx > 0 and idx % 200 == 0:
        print(f"Processed {idx}/{len(df_train)} samples...")

df_pitches = pd.DataFrame(pitches)

# Group by Pitch_ID and print stats
summary = df_pitches.groupby('Pitch_ID').agg(
    median_freq=('Freq', 'median'),
    median_midi=('MIDI', 'median'),
    std_midi=('MIDI', 'std'),
    count=('MIDI', 'count')
).reset_index()

print("\n=== YIN Pitch ID Mapping Summary ===")
print(summary.to_string(index=False))

# Let's save the summary to a CSV mapping file
summary.to_csv("agents/gemini/workspace/pitch_yin_mapping.csv", index=False)
print("Saved mapping to agents/gemini/workspace/pitch_yin_mapping.csv")
