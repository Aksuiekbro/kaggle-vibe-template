import os
import time
import json
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

data_dir = "shared/data/kaggle_dataset-20251026T143755Z-1-001/"
train_csv = os.path.join(data_dir, "kaggle_dataset/train.csv")
test_csv = os.path.join(data_dir, "kaggle_dataset/test.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

LOG_FREQ_BINS = np.logspace(np.log10(60), np.log10(3000), 121)
CANDIDATE_FREQS = np.logspace(np.log10(60), np.log10(1000), 200)

def extract_features_exp001(path):
    full_path = os.path.join(data_dir, path)
    try:
        sr, wav = wavfile.read(full_path)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return np.zeros(558)
    
    # Convert to float32
    y = wav.astype(np.float32) / 32768.0
    
    # Extract middle 5 seconds
    duration = len(y) / sr
    start_sec = max(0.0, (duration - 5.0) / 2.0)
    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + 5.0) * sr)
    y_segment = y[start_idx:end_idx]
    
    y_ds = y_segment[::4]
    sr_ds = sr / 4.0
    N = len(y_ds)
    X = np.abs(np.fft.rfft(y_ds))
    
    scale = N / sr_ds
    
    # 2. Log-frequency energy bins
    log_feats = np.zeros(120, dtype=np.float32)
    for i in range(120):
        low_f = LOG_FREQ_BINS[i]
        high_f = LOG_FREQ_BINS[i+1]
        idx_start = int(np.ceil(low_f * scale))
        idx_end = int(np.ceil(high_f * scale))
        if idx_start < idx_end and idx_start < len(X):
            log_feats[i] = np.sum(X[idx_start:idx_end])
            
    # 3. HPS features
    X_log = np.log(X + 1e-6)
    hps_feats = np.zeros(200, dtype=np.float32)
    for i, f in enumerate(CANDIDATE_FREQS):
        val = 0.0
        for r in range(1, 6):
            target_f = r * f
            idx_start = int(np.ceil(target_f * 0.98 * scale))
            idx_end = int(np.floor(target_f * 1.02 * scale)) + 1
            idx_start = max(0, idx_start)
            idx_end = min(len(X_log), idx_end)
            if idx_start < idx_end:
                val += np.max(X_log[idx_start:idx_end])
            else:
                val += np.log(1e-6)
        hps_feats[i] = val
        
    # 4. ACF features (direct)
    min_lag = int(np.floor(sr_ds / 1000.0))
    max_lag = int(np.ceil(sr_ds / 60.0))
    acf_feats = np.array([np.dot(y_ds[:-lag], y_ds[lag:]) for lag in range(min_lag, max_lag + 1)], dtype=np.float32)
    
    target_acf_len = 174
    if len(acf_feats) < target_acf_len:
        acf_feats = np.pad(acf_feats, (0, target_acf_len - len(acf_feats)))
    else:
        acf_feats = acf_feats[:target_acf_len]
        
    # 5. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # 6. MFCCs
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    return np.concatenate([log_feats, hps_feats, acf_feats, chroma_mean, chroma_std, mfcc_mean, mfcc_std])

# Feature caching
cache_dir = "agents/gemini/workspace"
X_train_path = os.path.join(cache_dir, "X_train_exp001.npy")
X_test_path = os.path.join(cache_dir, "X_test_exp001.npy")
y_train_path = os.path.join(cache_dir, "y_train_exp001.npy")

if os.path.exists(X_train_path) and os.path.exists(X_test_path) and os.path.exists(y_train_path):
    print("Loading cached features from disk...")
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    print(f"Loaded train features shape: {X_train.shape}")
    print(f"Loaded test features shape: {X_test.shape}")
else:
    print("Extracting train features...")
    t0 = time.time()
    X_train = Parallel(n_jobs=-1)(delayed(extract_features_exp001)(p) for p in df_train['Path'])
    X_train = np.array(X_train)
    y_train = df_train['Pitch_ID'].values
    print(f"Train features extracted: {X_train.shape} in {time.time() - t0:.2f}s")
    
    print("Extracting test features...")
    t0 = time.time()
    X_test = Parallel(n_jobs=-1)(delayed(extract_features_exp001)(p) for p in df_test['Path'])
    X_test = np.array(X_test)
    print(f"Test features extracted: {X_test.shape} in {time.time() - t0:.2f}s")
    
    # Save cache
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    print("Features cached to disk.")

# Stratified 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
test_preds_probs = np.zeros((len(df_test), 82))
oof_preds = np.zeros(len(df_train))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Training Fold {fold} ---")
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    # Same hyperparameters as baseline to compare features fairly
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=82,
        metric='multi_logloss',
        learning_rate=0.1,
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    t_start = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    preds = model.predict(X_va)
    oof_preds[val_idx] = preds
    acc = accuracy_score(y_va, preds)
    cv_scores.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f} in {time.time() - t_start:.2f}s")
    
    # Accumulate test probabilities
    test_preds_probs += model.predict_proba(X_test) / 5.0

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\nCV Accuracy Mean: {cv_mean:.4f} Std: {cv_std:.4f}")

# Save results for experiment logging
results = {
    "cv_scores": cv_scores,
    "cv_mean": cv_mean,
    "cv_std": cv_std
}
with open("agents/gemini/workspace/results_exp001.json", "w") as f:
    json.dump(results, f, indent=2)
