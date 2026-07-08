"""exp_076 (own, scheduler id exp_074): make augment_rows scattering-compatible
so the two confirmed-independent grid-only levers -- augmentation (+0.0085
topq alone) and scattering (+0.0223 topq alone, exp_073/074 full CV) -- can
finally be tested together. Previously augment_rows only re-extracted grid
features from augmented audio (extract_from_signal), so scattering was
dropped from every augmented comparison (exp_073/074's grid+scattering runs
are all no-aug).

Adds extract_scattering_from_signal(y, sr): mirrors extract_scattering_features
.py's extract_one but takes an in-memory signal (already at native sr from
soundfile) instead of a file path, resampling to the same 8kHz/2**16-sample
shape via librosa.resample instead of librosa.load. Extends augment_rows to
emit both grid and scattering features per augmented variant.

exp_075 found scattering + pseudo-labeling composes NEGATIVELY (-0.0085),
flagged risk: grid+scattering's near-perfect adversarial train/test AUC
(0.9975) may make confident test-set pseudo-labels reinforce shift artifacts.
Augmentation adds only synthetic IN-DISTRIBUTION rows (no test-set labels
involved), a mechanistically different intervention -- this probe tests
directly rather than assuming it shares pseudo-labeling's failure mode.

Single stratified 80/20 holdout (probe fidelity), same SEED/split family as
probe_exp074_grid_noaug_control.py so the grid+scattering-no-aug topq is
directly comparable. Fold-safe: augmentation generated only from the TRAIN
split's rare-class rows; the held-out 20% is untouched, exactly as exp_037/071.
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
import soundfile as sf
import xgboost as xgb
import lightgbm as lgb
from kymatio.numpy import Scattering1D
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from exp025_augmented_blend import DATA_ROOT, SEED, RARE_THRESHOLD, extract_from_signal

GRID_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
SCAT_DIR = "/root/kaggle-vibe-template/agents/claude/workspace/scripts/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"

GRID_SCATTERING_TOPQ_NOAUG = 0.9658  # exp_073 full CV reference (different fidelity, context only)

SC_SR = 8000
SC_N = 2 ** 16
SC_J, SC_Q = 8, 8
_scattering = Scattering1D(J=SC_J, Q=SC_Q, shape=(SC_N,))

MORE_OVERSAMPLE_SNRS = (20.0, 15.0, 10.0)
MORE_OVERSAMPLE_RATES = (0.85, 0.9, 1.1, 1.15)


def extract_scattering_from_signal(y, sr):
    y8k = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SC_SR)
    if len(y8k) < SC_N:
        y8k = np.pad(y8k, (0, SC_N - len(y8k)))
    else:
        y8k = y8k[:SC_N]
    Sx = _scattering(y8k.astype(np.float64))
    mean_feats = Sx.mean(axis=1)
    std_feats = Sx.std(axis=1)
    feats = {f"sc_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"sc_std_{i}": v for i, v in enumerate(std_feats)})
    return feats


def augment_variants(path, rng, snrs, rates):
    y, sr = sf.read(DATA_ROOT + path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    variants = []
    sig_power = np.mean(y ** 2)
    for snr_db in snrs:
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
        variants.append(y + noise)
    for rate in rates:
        variants.append(librosa.effects.time_stretch(y, rate=rate).astype(np.float32))
    return variants, sr


def augment_rows_grid_scattering(rows_df, rng_master, snrs, rates):
    aug_rows = []
    for path, pitch_id in zip(rows_df["Path"], rows_df["Pitch_ID"]):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        variants, sr = augment_variants(path, rng, snrs, rates)
        for v in variants:
            feats = extract_from_signal(v, sr)
            feats.update(extract_scattering_from_signal(v, sr))
            feats["Pitch_ID"] = pitch_id
            aug_rows.append(feats)
    return pd.DataFrame(aug_rows)


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
scat_train = pd.read_parquet(SCAT_DIR + "train_scattering_features.parquet")
scat_test = pd.read_parquet(SCAT_DIR + "test_scattering_features.parquet")

grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
scat_feat_cols = [c for c in scat_train.columns if c.startswith("sc_")]
feat_cols = grid_feat_cols + scat_feat_cols

# same row-filter as exp_073/074: inner join to the scattering set
train = grid_train.merge(scat_train[["Path"] + scat_feat_cols], on="Path", how="inner")
test = grid_test.merge(scat_test[["Path"] + scat_feat_cols], on="Path", how="inner")

le = LabelEncoder()
y_all = le.fit_transform(train["Pitch_ID"].values)
n_classes = len(le.classes_)
X_all = train[feat_cols].values
full_counts = pd.read_csv(TRAIN_CSV)["Pitch_ID"].value_counts()

tr_idx, va_idx = train_test_split(
    np.arange(len(y_all)), test_size=0.2, stratify=y_all, random_state=SEED
)
X_tr, X_va = X_all[tr_idx], X_all[va_idx]
y_tr, y_va = y_all[tr_idx], y_all[va_idx]
tr_rows = train.iloc[tr_idx]

# adversarial classifier (grid+scattering feature space) for topq, same recipe as exp_073/074
X_test_real = test[feat_cols].values
X_adv = np.vstack([X_all, X_test_real])
y_adv = np.concatenate([np.zeros(len(X_all)), np.ones(len(X_test_real))])
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_test_all = np.zeros(len(y_adv))
for tri, vai in skf_adv.split(X_adv, y_adv):
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1, n_jobs=1,
    )
    clf.fit(X_adv[tri], y_adv[tri])
    p_test_all[vai] = clf.predict_proba(X_adv[vai])[:, 1]
adv_auc = roc_auc_score(y_adv, p_test_all)
w_full = p_test_all[: len(X_all)]
w_va = w_full[va_idx]
q75 = np.quantile(w_va, 0.75)
top_mask = w_va >= q75
print(f"grid+scattering adversarial AUC: {adv_auc:.4f} n_feats={len(feat_cols)} "
      f"n_topq={top_mask.sum()} ({time.time()-t0:.0f}s)", flush=True)

# baseline: grid+scattering, no augmentation
t_b = time.time()
clf_base = make_xgb(n_classes)
clf_base.fit(X_tr, y_tr)
pred_base = clf_base.predict(X_va)
correct_base = (pred_base == y_va).astype(float)
plain_b = correct_base.mean()
weighted_b = np.average(correct_base, weights=w_va)
topq_b = correct_base[top_mask].mean()
print(f"grid+scattering no-aug baseline ({time.time()-t_b:.0f}s): n_train={len(X_tr)} "
      f"plain={plain_b:.4f} weighted={weighted_b:.4f} topq={topq_b:.4f}", flush=True)

# treatment: grid+scattering + rare-class augmentation (more_oversample config, exp_071's best)
rare_classes = set(full_counts[full_counts < RARE_THRESHOLD].index.tolist())
rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
rng_master = np.random.default_rng(SEED)
t_a = time.time()
aug_df = augment_rows_grid_scattering(rare_tr_rows, rng_master, MORE_OVERSAMPLE_SNRS, MORE_OVERSAMPLE_RATES)
print(f"augmented {len(rare_tr_rows)} rare rows -> {len(aug_df)} extra rows ({time.time()-t_a:.0f}s)", flush=True)

X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

t_c = time.time()
clf_aug = make_xgb(n_classes)
clf_aug.fit(X_tr_aug, y_tr_aug)
pred_aug = clf_aug.predict(X_va)
correct_aug = (pred_aug == y_va).astype(float)
plain_a = correct_aug.mean()
weighted_a = np.average(correct_aug, weights=w_va)
topq_a = correct_aug[top_mask].mean()
print(f"grid+scattering + augmentation ({time.time()-t_c:.0f}s): n_train={len(X_tr_aug)} "
      f"plain={plain_a:.4f} (d{plain_a-plain_b:+.4f}) "
      f"weighted={weighted_a:.4f} (d{weighted_a-weighted_b:+.4f}) "
      f"topq={topq_a:.4f} (d{topq_a-topq_b:+.4f})  <- coordinator's gating metric", flush=True)

print(f"\n(context only, different fidelity) exp_073 full-CV grid+scattering-no-aug topq reference: "
      f"{GRID_SCATTERING_TOPQ_NOAUG}", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
