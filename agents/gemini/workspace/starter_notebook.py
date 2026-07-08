import os
import pandas as pd
import librosa
import numpy as np
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


# ========================================


root_dir = "/kaggle/input/missing-fundamental-puzzle/kaggle_dataset-20251026T143755Z-1-001/"

train = pd.read_csv(f"{root_dir}/kaggle_dataset/train.csv")
test = pd.read_csv(f"{root_dir}/kaggle_dataset/test.csv")
submission = pd.read_csv(f"{root_dir}/kaggle_dataset/sample_submission.csv")


# ========================================


train.head(3)


# ========================================


df_train, df_valid = train_test_split(train, test_size=0.2, random_state=42)
df_train.reset_index(inplace=True, drop=True)
df_valid.reset_index(inplace=True, drop=True)

print(df_train.shape)
print(df_valid.shape)


# ========================================


SR = 16000
DURATION = 0.01  # seconds
TARGET_LEN = int(SR * DURATION)

X_train = []
for idx in tqdm(range(len(df_train))):
    path_to_wav = os.path.join(root_dir, df_train.loc[idx, 'Path'])
    wav, sr = librosa.load(path_to_wav, sr=SR, mono=True)
    if len(wav) > TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    X_train.append(wav)

X_train = np.array(X_train)
y_train = df_train['Pitch_ID'].values


# ========================================


SR = 16000
DURATION = 0.01  # seconds
TARGET_LEN = int(SR * DURATION)

X_valid = []
for idx in tqdm(range(len(df_valid))):
    path_to_wav = os.path.join(root_dir, df_valid.loc[idx, 'Path'])
    wav, sr = librosa.load(path_to_wav, sr=SR, mono=True)
    if len(wav) > TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    X_valid.append(wav)

X_valid = np.array(X_valid)
y_valid = df_valid['Pitch_ID'].values


# ========================================


# --- Load test ---
X_test = []
for idx in tqdm(range(len(test))):
    path_to_wav = os.path.join(root_dir, test.loc[idx, 'Path'])
    wav, sr = librosa.load(path_to_wav, sr=SR, mono=True)
    if len(wav) > TARGET_LEN:
        wav = wav[:TARGET_LEN]
    else:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    X_test.append(wav)

X_test = np.array(X_test)


# ========================================


from sklearn.ensemble import RandomForestClassifier


# ========================================


model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


# ========================================


model.fit(X_train, y_train)


# ========================================


from sklearn.metrics import accuracy_score


# ========================================


y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Test Accuracy: {acc:.4f}")


# ========================================


y_pred = model.predict(X_test)


# ========================================


submission['Pitch_ID'] = y_pred


# ========================================


submission


# ========================================


submission.to_csv("submission.csv", index=False)


# ========================================

