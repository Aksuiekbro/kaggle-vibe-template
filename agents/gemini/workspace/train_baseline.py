import os
import time
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import correlate
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

def extract_features(path):
    full_path = os.path.join(data_dir, path)
    try:
        sr, wav = wavfile.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return np.zeros(494)
    
    # Convert to float32
    y = wav.astype(np.float32) / 32768.0
    
    # Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    # Downsample by 4
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N = len(y_ds)
    
    # 1. FFT Magnitude
    X = np.abs(np.fft.rfft(y_ds))
    
    # 2. Log-frequency energy bins (60 Hz to 3000 Hz)
    log_freq_bins = np.logspace(np.log10(60), np.log10(3000), 121)
    log_feats = []
    factor = N / sr_ds
    for i in range(120):
        low_f = log_freq_bins[i]
        high_f = log_freq_bins[i+1]
        idx_start = int(np.ceil(low_f * factor))
        idx_end = int(np.ceil(high_f * factor))
        if idx_start < idx_end and idx_start < len(X):
            log_feats.append(np.sum(X[idx_start:min(idx_end, len(X))]))
        else:
            log_feats.append(0.0)
    log_feats = np.array(log_feats)
    
    # 3. HPS features (60 Hz to 1000 Hz) in log-magnitude
    X_log = np.log(X + 1e-6)
    candidate_freqs = np.logspace(np.log10(60), np.log10(1000), 200)
    hps_feats = []
    for f in candidate_freqs:
        val = 0.0
        for r in range(1, 6): # harmonics 1 to 5
            target_f = r * f
            idx_start = int(np.ceil(target_f * 0.98 * factor))
            idx_end = int(np.floor(target_f * 1.02 * factor)) + 1
            if idx_start < idx_end and idx_start < len(X_log):
                val += np.max(X_log[idx_start:min(idx_end, len(X_log))])
            else:
                val += np.log(1e-6)
        hps_feats.append(val)
    hps_feats = np.array(hps_feats)
    
    # 4. ACF features
    acf = correlate(y_ds, y_ds, mode='full')
    center = len(y_ds) - 1
    # lags corresponding to 60Hz to 1000Hz: sr_ds/1000 to sr_ds/60
    min_lag = int(np.floor(sr_ds / 1000.0))
    max_lag = int(np.ceil(sr_ds / 60.0))
    acf_feats = acf[center + min_lag : center + max_lag + 1]
    
    target_acf_len = 174
    if len(acf_feats) < target_acf_len:
        acf_feats = np.pad(acf_feats, (0, target_acf_len - len(acf_feats)))
    else:
        acf_feats = acf_feats[:target_acf_len]
        
    return np.concatenate([log_feats, hps_feats, acf_feats])

print("Extracting train features...")
t0 = time.time()
X_train = Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in df_train['Path'])
X_train = np.array(X_train)
y_train = df_train['Pitch_ID'].values
print(f"Train features extracted: {X_train.shape} in {time.time() - t0:.2f}s")

print("Extracting test features...")
t0 = time.time()
X_test = Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in df_test['Path'])
X_test = np.array(X_test)
print(f"Test features extracted: {X_test.shape} in {time.time() - t0:.2f}s")

# Stratified 3-Fold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cv_scores = []
test_preds_probs = np.zeros((len(df_test), 82))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Training Fold {fold} ---")
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=82,
        metric='multi_logloss',
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    preds = model.predict(X_va)
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    
    # Accumulate test probabilities
    test_preds_probs += model.predict_proba(X_test) / 3.0

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\nCV Accuracy Mean: {cv_mean:.4f} Std: {cv_std:.4f}")

# Make directory and save submission
os.makedirs("agents/gemini/submissions", exist_ok=True)
sub_path = "agents/gemini/submissions/submission_baseline.csv"

# Final test predictions
final_preds = np.argmax(test_preds_probs, axis=1)
df_sub = pd.DataFrame({
    'Path': df_test['Path'],
    'Pitch_ID': final_preds
})
df_sub.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path}")

# Output results in standard JSON format for easy reporting
import json
results = {
    "cv_scores": cv_scores,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "submission_file": sub_path
}
with open("agents/gemini/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)
