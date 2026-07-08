"""exp_077 (own, research follow-up to exp_073's flagged-but-untested risk):
does raising the scattering sample rate from 8kHz (Nyquist 4kHz) to 16kHz
(Nyquist 8kHz) recover signal for high-fundamental classes?

exp_073's extraction note explicitly flagged (not tested): "8kHz downsampling
implies a 4kHz Nyquist -- MIDI 110 (~3951Hz) is near the edge, so higher
harmonics of the highest-pitch classes get truncated." Cross-checking
gemini's pitch_yin_mapping.csv (YIN-detected median_freq per Pitch_ID, noisy
but usable for a coarse high/low ranking): max median_freq is ~2005Hz
(Pitch_ID 18), 14/82 classes have median_freq > 1000Hz. A class whose
fundamental is ~1500-2000Hz has its 2nd-3rd harmonics at 3000-6000Hz --
exactly the range 8kHz-SR scattering truncates. This has never been tested
directly; it's a flagged risk, not a measured one.

This probe recomputes scattering at 16kHz (SC_N doubled to keep the same
~8.2s window) for a stratified SUBSAMPLE only (cheap check before committing
to a full 16kHz re-extraction, which would cost ~2x the original ~50min
extraction). Subsample is stratified by median_freq tertile so the
high-frequency-class subset is guaranteed representation.

Compares, on the identical subsample and identical 80/20-style split:
  grid + scattering-8kHz (already-extracted full-dataset parquet, filtered
    to this subsample) vs grid + scattering-16kHz (newly extracted here)
  -- overall topq AND topq restricted to the high-median_freq tertile only,
  since that's the subset the 8kHz Nyquist risk specifically predicts should
  improve.

No augmentation here (orthogonal to exp_076, which is testing that
composition separately) -- isolates the SR question alone.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.special as _sp

if not hasattr(_sp, "sph_harm"):
    def _sph_harm_compat(m, n, theta, phi):
        return _sp.sph_harm_y(n, m, phi, theta)
    _sp.sph_harm = _sph_harm_compat

warnings.filterwarnings("ignore")
import librosa
import xgboost as xgb
import lightgbm as lgb
from kymatio.numpy import Scattering1D
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import SEED

DATA_ROOT = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/"
GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = DATA_ROOT + "kaggle_dataset/train.csv"
TEST_CSV = DATA_ROOT + "kaggle_dataset/test.csv"
YIN_MAP_CSV = "/root/kaggle-vibe-template/agents/gemini/workspace/pitch_yin_mapping.csv"

SC_SR_HI = 16000
SC_N_HI = 2 ** 17
SC_J, SC_Q = 8, 8
_scattering_hi = Scattering1D(J=SC_J, Q=SC_Q, shape=(SC_N_HI,))

N_TRAIN_SUB = 500
N_TEST_SUB = 200


def extract_scattering_hi(path):
    y, _ = librosa.load(DATA_ROOT + path, sr=SC_SR_HI)
    if len(y) < SC_N_HI:
        y = np.pad(y, (0, SC_N_HI - len(y)))
    else:
        y = y[:SC_N_HI]
    Sx = _scattering_hi(y.astype(np.float64))
    mean_feats = Sx.mean(axis=1)
    std_feats = Sx.std(axis=1)
    feats = {f"sc16_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"sc16_std_{i}": v for i, v in enumerate(std_feats)})
    return feats


def make_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=n_classes, random_state=42, n_jobs=1, tree_method="hist",
        verbosity=0,
    )


t0 = time.time()
grid_train = pd.read_parquet(GRID_DIR + "train_grid_features.parquet")
grid_test = pd.read_parquet(GRID_DIR + "test_grid_features.parquet")
scat8_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat8_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")
yin_map = pd.read_csv(YIN_MAP_CSV)[["Pitch_ID", "median_freq"]]

grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
scat8_feat_cols = [c for c in scat8_train.columns if c.startswith("sc_")]

train_full = grid_train.merge(scat8_train[["Path"] + scat8_feat_cols], on="Path", how="inner")
train_full = train_full.merge(yin_map, on="Pitch_ID", how="left")
test_full = grid_test.merge(scat8_test[["Path"] + scat8_feat_cols], on="Path", how="inner")

freq_tertile = pd.qcut(train_full["median_freq"], 3, labels=["low", "mid", "high"])
train_full["freq_tertile"] = freq_tertile

# stratified subsample by (Pitch_ID) for representativeness, capped at N_TRAIN_SUB
rng = np.random.default_rng(SEED)
sub_idx = train_full.groupby("Pitch_ID", group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), max(1, int(np.ceil(N_TRAIN_SUB * len(g) / len(train_full))))),
                        random_state=SEED)
).index
train_sub = train_full.loc[sub_idx].reset_index(drop=True)
test_sub = test_full.sample(n=min(N_TEST_SUB, len(test_full)), random_state=SEED).reset_index(drop=True)
print(f"subsample: {len(train_sub)} train / {len(test_sub)} test "
      f"(high-tertile train rows: {(train_sub['freq_tertile']=='high').sum()}) ({time.time()-t0:.0f}s)", flush=True)

t_e = time.time()
train_sc16_path = SCAT_DIR + "exp077_train_scattering16k_subsample.parquet"
test_sc16_path = SCAT_DIR + "exp077_test_scattering16k_subsample.parquet"
train_sc16 = pd.DataFrame([extract_scattering_hi(p) for p in train_sub["Path"]])
test_sc16 = pd.DataFrame([extract_scattering_hi(p) for p in test_sub["Path"]])
train_sc16.to_parquet(train_sc16_path)
test_sc16.to_parquet(test_sc16_path)
print(f"16kHz scattering extracted for subsample ({time.time()-t_e:.0f}s), cached to {train_sc16_path}", flush=True)

sc16_feat_cols = [c for c in train_sc16.columns if c.startswith("sc16_")]

# drop singleton classes (can't be stratified across train/val, and XGB's
# num_class needs every training fold to see every class it was told exists)
class_counts = train_sub["Pitch_ID"].value_counts()
keep_mask = train_sub["Pitch_ID"].isin(class_counts[class_counts >= 2].index).values
n_dropped = (~keep_mask).sum()
if n_dropped:
    print(f"dropping {n_dropped} rows from {(class_counts < 2).sum()} singleton classes before split", flush=True)
train_sub = train_sub.loc[keep_mask].reset_index(drop=True)
train_sc16 = train_sc16.loc[keep_mask].reset_index(drop=True)

le = LabelEncoder()
y_all = le.fit_transform(train_sub["Pitch_ID"].values)
n_classes = len(le.classes_)

tr_idx, va_idx = train_test_split(
    np.arange(len(y_all)), test_size=0.25, random_state=SEED, stratify=y_all
)
high_va_mask = (train_sub["freq_tertile"].values[va_idx] == "high")

X_8 = np.hstack([train_sub[grid_feat_cols].values, train_sub[scat8_feat_cols].values])
X_16 = np.hstack([train_sub[grid_feat_cols].values, train_sc16[sc16_feat_cols].values])
X_test_8 = np.hstack([test_sub[grid_feat_cols].values, test_sub[scat8_feat_cols].values])
X_test_16 = np.hstack([test_sub[grid_feat_cols].values, test_sc16[sc16_feat_cols].values])

y_tr, y_va = y_all[tr_idx], y_all[va_idx]


def adv_weights(X_train_arr, X_test_arr, tr_idx, va_idx):
    X_adv = np.vstack([X_train_arr, X_test_arr])
    y_adv = np.concatenate([np.zeros(len(X_train_arr)), np.ones(len(X_test_arr))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    p = np.zeros(len(y_adv))
    for a, b in skf.split(X_adv, y_adv):
        clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=-1, n_jobs=1)
        clf.fit(X_adv[a], y_adv[a])
        p[b] = clf.predict_proba(X_adv[b])[:, 1]
    auc = roc_auc_score(y_adv, p)
    w_train = p[: len(X_train_arr)]
    return auc, w_train[va_idx]


def eval_variant(name, X_train_arr, X_test_arr):
    auc, w_va = adv_weights(X_train_arr, X_test_arr, tr_idx, va_idx)
    q75 = np.quantile(w_va, 0.75)
    top_mask = w_va >= q75
    clf = make_xgb(n_classes)
    clf.fit(X_train_arr[tr_idx], y_tr)
    pred = clf.predict(X_train_arr[va_idx])
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    topq = correct[top_mask].mean() if top_mask.sum() else float("nan")
    topq_high = correct[high_va_mask].mean() if high_va_mask.sum() else float("nan")
    print(f"{name}: adv_auc={auc:.4f} plain={plain:.4f} topq={topq:.4f} "
          f"(n={top_mask.sum()}) topq_high_freq_tertile={topq_high:.4f} (n={high_va_mask.sum()})",
          flush=True)
    return plain, topq, topq_high


t_v = time.time()
eval_variant("grid+scattering-8kHz ", X_8, X_test_8)
eval_variant("grid+scattering-16kHz", X_16, X_test_16)
print(f"eval done ({time.time()-t_v:.0f}s)", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
