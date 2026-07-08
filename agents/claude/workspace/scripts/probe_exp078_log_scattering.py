"""exp_078 (own, research follow-up to exp_073/074's scattering win): does
log-compressing the wavelet scattering coefficients before mean/std pooling
help, and does it reduce the near-perfect train/test adversarial separability
(0.9975) that exp_073/074/075 flagged as a real risk?

Motivation: kymatio's own scattering-for-audio-classification examples and
the broader scattering-transform literature (Andén & Mallat 2014, "Deep
Scattering Spectrum") standardly apply a log (or log1p) nonlinearity to
scattering coefficients before use in a classifier -- analogous to why
log-mel spectrograms, not raw-mel, are standard in audio ML. The current
extract_scattering_features.py pools raw (non-log) Sx values, which have a
heavy-tailed distribution dominated by a few large low-order/low-frequency
coefficients (mostly encoding overall signal energy/loudness) -- exactly the
kind of recording-level loudness difference exp_003/004 already diagnosed as
the shift mechanism. Log-compression compresses this dynamic range, which
could (a) reveal more of the relative/pitch-relevant structure the raw scale
currently drowns out, and (b) directly reduce how strongly scattering
encodes absolute loudness, addressing the adv-AUC risk without touching
pseudo-labeling/augmentation composition (both already tested against
raw scattering).

Genuinely new and cheap: no new audio decode needed beyond a single
Scattering1D call per file (same cost as the original extraction) -- log1p
pooling is computed from the exact same Sx array used for raw pooling, so a
SUBSAMPLE probe (not full 2913-file re-extraction) can directly A/B raw vs
log-pooled on identical rows before committing to a full re-extraction.

Compares, on an identical stratified subsample + 80/20-style holdout,
reusing the already-cached FULL grid+raw-scattering parquets filtered to the
same rows (no need to recompute raw features):
  1. grid + raw-scattering   (reference, matches exp_073/074's mechanism)
  2. grid + log-scattering   (log1p(Sx) mean/std pooled instead of raw)
  3. grid + BOTH concatenated (raw + log, in case they carry complementary
     information rather than one dominating the other)
Also reports adversarial train/test AUC for each variant (the exp_073/074/
075 risk flag was specifically about raw-scattering's 0.9975 AUC).
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

SC_SR = 8000
SC_N = 2 ** 16
SC_J, SC_Q = 8, 8
_scattering = Scattering1D(J=SC_J, Q=SC_Q, shape=(SC_N,))

N_TRAIN_SUB = 500
N_TEST_SUB = 200


def extract_raw_and_log(path):
    y, _ = librosa.load(DATA_ROOT + path, sr=SC_SR)
    if len(y) < SC_N:
        y = np.pad(y, (0, SC_N - len(y)))
    else:
        y = y[:SC_N]
    Sx = _scattering(y.astype(np.float64))  # (n_coeffs, n_time), Sx >= 0
    log_Sx = np.log1p(Sx)
    feats = {}
    for i, v in enumerate(Sx.mean(axis=1)):
        feats[f"sc_mean_{i}"] = v
    for i, v in enumerate(Sx.std(axis=1)):
        feats[f"sc_std_{i}"] = v
    for i, v in enumerate(log_Sx.mean(axis=1)):
        feats[f"logsc_mean_{i}"] = v
    for i, v in enumerate(log_Sx.std(axis=1)):
        feats[f"logsc_std_{i}"] = v
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
grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# stratified-by-class subsample, capped at N_TRAIN_SUB, mirroring exp_077's
# (now-fixed) approach
rng = np.random.default_rng(SEED)
sub_idx = train_df.groupby("Pitch_ID", group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), max(1, int(np.ceil(N_TRAIN_SUB * len(g) / len(train_df))))),
                        random_state=SEED)
).index
train_sub = train_df.loc[sub_idx].reset_index(drop=True)
test_sub = test_df.sample(n=min(N_TEST_SUB, len(test_df)), random_state=SEED).reset_index(drop=True)

# drop singleton classes (exp_077's bug fix) so stratified split never
# starves y_tr of a class XGBoost was told to expect
class_counts = train_sub["Pitch_ID"].value_counts()
keep_mask = train_sub["Pitch_ID"].isin(class_counts[class_counts >= 2].index).values
n_dropped = (~keep_mask).sum()
train_sub = train_sub.loc[keep_mask].reset_index(drop=True)
print(f"subsample: {len(train_sub)} train ({n_dropped} singleton-class rows dropped) / "
      f"{len(test_sub)} test ({time.time()-t0:.0f}s)", flush=True)

t_e = time.time()
train_feats = pd.DataFrame([extract_raw_and_log(p) for p in train_sub["Path"]])
test_feats = pd.DataFrame([extract_raw_and_log(p) for p in test_sub["Path"]])
train_feats.to_parquet(SCAT_DIR + "exp078_train_rawlog_scattering_subsample.parquet")
test_feats.to_parquet(SCAT_DIR + "exp078_test_rawlog_scattering_subsample.parquet")
print(f"raw+log scattering extracted for subsample ({time.time()-t_e:.0f}s)", flush=True)

raw_cols = [c for c in train_feats.columns if c.startswith("sc_")]
log_cols = [c for c in train_feats.columns if c.startswith("logsc_")]

train_merged = train_sub.merge(grid_train[["Path"] + grid_feat_cols], on="Path", how="inner")
train_merged = train_merged.merge(train_feats.assign(Path=train_sub["Path"].values), on="Path", how="inner")
test_merged = test_sub.merge(grid_test[["Path"] + grid_feat_cols], on="Path", how="inner")
test_merged = test_merged.merge(test_feats.assign(Path=test_sub["Path"].values), on="Path", how="inner")

le = LabelEncoder()
y_all = le.fit_transform(train_merged["Pitch_ID"].values)
n_classes = len(le.classes_)

tr_idx, va_idx = train_test_split(
    np.arange(len(y_all)), test_size=0.25, random_state=SEED, stratify=y_all
)
y_tr, y_va = y_all[tr_idx], y_all[va_idx]


def adv_weights(X_train_arr, X_test_arr):
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
    return auc, p[: len(X_train_arr)]


def eval_variant(name, feat_cols):
    X_train_arr = train_merged[grid_feat_cols + feat_cols].values
    X_test_arr = test_merged[grid_feat_cols + feat_cols].values
    auc, w_train = adv_weights(X_train_arr, X_test_arr)
    w_va = w_train[va_idx]
    q75 = np.quantile(w_va, 0.75)
    top_mask = w_va >= q75
    clf = make_xgb(n_classes)
    clf.fit(X_train_arr[tr_idx], y_tr)
    pred = clf.predict(X_train_arr[va_idx])
    correct = (pred == y_va).astype(float)
    plain = correct.mean()
    topq = correct[top_mask].mean() if top_mask.sum() else float("nan")
    print(f"{name}: adv_auc={auc:.4f} plain={plain:.4f} topq={topq:.4f} (n_topq={top_mask.sum()})",
          flush=True)
    return auc, plain, topq


t_v = time.time()
r_raw = eval_variant("grid+raw-scattering", raw_cols)
r_log = eval_variant("grid+log-scattering", log_cols)
r_both = eval_variant("grid+raw+log-scattering", raw_cols + log_cols)
print(f"delta topq (log vs raw): {r_log[2]-r_raw[2]:+.4f}  "
      f"delta adv_auc (log vs raw): {r_log[0]-r_raw[0]:+.4f}", flush=True)
print(f"eval done ({time.time()-t_v:.0f}s)", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
