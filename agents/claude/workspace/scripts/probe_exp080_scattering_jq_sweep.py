"""exp_080 (own, new): sweep the wavelet scattering transform's Q parameter
(wavelets per octave / frequency resolution), never tuned this competition --
extract_scattering_features.py has hardcoded J=8, Q=8 since exp_073 first
introduced the family, chosen as a generic default, not searched.

Motivated directly by arxiv 1601.00287 ("Wavelet Scattering on the Pitch
Spiral"), the paper that originally motivated trying scattering here: it
argues Q controls how finely the first-order filterbank resolves pitch,
and that pitch-classification tasks benefit from Q values well above
generic audio-classification defaults (Q=1) -- exactly the axis nothing in
this competition has swept. This is a hyperparameter-tuning probe on an
already-confirmed-positive lever (exp_073/074, full CV topq +0.0223), same
class of experiment as the already-successful XGBoost/LGBM hyperparameter
sweeps (exp_053/069), not a new mechanism family.

Cheap stratified subsample (same construction as exp_077: 500 train / 200
test, stratified by Pitch_ID), same 80/20-style split/seed. Compares
grid+scattering at Q in {4, 8 (reference, already-cached full-dataset
J=8/Q=8 8kHz features), 16} on overall topq. If a wider Q beats the
reference by more than subsample noise, promote to a full re-extraction +
full CV (same pattern as exp_075/16kHz); if flat, this closes the
scattering-hyperparameter axis and the family is fully exhausted (7/7
non-positive-beyond-standalone: SR, log-compression, augmentation,
pseudo-labeling, proba-blend, Q-sweep all fail to add on top of the
standalone +0.0223).
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
SC_J = 8
SWEEP_Q = (4, 16)  # Q=8 reference reuses the already-cached full-dataset extraction

N_TRAIN_SUB = 500
N_TEST_SUB = 200

_scatterers = {q: Scattering1D(J=SC_J, Q=q, shape=(SC_N,)) for q in SWEEP_Q}


def extract_scattering(path, q):
    y, _ = librosa.load(DATA_ROOT + path, sr=SC_SR)
    if len(y) < SC_N:
        y = np.pad(y, (0, SC_N - len(y)))
    else:
        y = y[:SC_N]
    Sx = _scatterers[q](y.astype(np.float64))
    mean_feats = Sx.mean(axis=1)
    std_feats = Sx.std(axis=1)
    feats = {f"scq{q}_mean_{i}": v for i, v in enumerate(mean_feats)}
    feats.update({f"scq{q}_std_{i}": v for i, v in enumerate(std_feats)})
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

grid_feat_cols = [c for c in grid_train.columns if c not in ("Path", "Pitch_ID")]
scat8_feat_cols = [c for c in scat8_train.columns if c.startswith("sc_")]

train_full = grid_train.merge(scat8_train[["Path"] + scat8_feat_cols], on="Path", how="inner")
test_full = grid_test.merge(scat8_test[["Path"] + scat8_feat_cols], on="Path", how="inner")

# stratified subsample by Pitch_ID for representativeness -- identical
# construction to exp_077's subsample (same SEED), capped at N_TRAIN_SUB
rng = np.random.default_rng(SEED)
sub_idx = train_full.groupby("Pitch_ID", group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), max(1, int(np.ceil(N_TRAIN_SUB * len(g) / len(train_full))))),
                        random_state=SEED)
).index
train_sub = train_full.loc[sub_idx].reset_index(drop=True)
test_sub = test_full.sample(n=min(N_TEST_SUB, len(test_full)), random_state=SEED).reset_index(drop=True)
print(f"subsample: {len(train_sub)} train / {len(test_sub)} test ({time.time()-t0:.0f}s)", flush=True)

# drop singleton classes before split (XGB num_class needs every class seen)
class_counts = train_sub["Pitch_ID"].value_counts()
keep_mask = train_sub["Pitch_ID"].isin(class_counts[class_counts >= 2].index).values
n_dropped = (~keep_mask).sum()
if n_dropped:
    print(f"dropping {n_dropped} rows from {(class_counts < 2).sum()} singleton classes before split", flush=True)
train_sub = train_sub.loc[keep_mask].reset_index(drop=True)

le = LabelEncoder()
y_all = le.fit_transform(train_sub["Pitch_ID"].values)
n_classes = len(le.classes_)
tr_idx, va_idx = train_test_split(
    np.arange(len(y_all)), test_size=0.25, random_state=SEED, stratify=y_all
)
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
    print(f"{name}: n_feats={X_train_arr.shape[1]} adv_auc={auc:.4f} plain={plain:.4f} "
          f"topq={topq:.4f} (n={top_mask.sum()}) ({time.time()-t0:.0f}s)", flush=True)
    return plain, topq


X_grid = train_sub[grid_feat_cols].values
X_test_grid = test_sub[grid_feat_cols].values
X_8 = np.hstack([X_grid, train_sub[scat8_feat_cols].values])
X_test_8 = np.hstack([X_test_grid, test_sub[scat8_feat_cols].values])

results = {}
results["Q=8 (reference, cached)"] = eval_variant("grid+scattering Q=8 (reference)", X_8, X_test_8)

for q in SWEEP_Q:
    t_e = time.time()
    train_scq = pd.DataFrame([extract_scattering(p, q) for p in train_sub["Path"]])
    test_scq = pd.DataFrame([extract_scattering(p, q) for p in test_sub["Path"]])
    print(f"Q={q} scattering extracted for subsample ({time.time()-t_e:.0f}s)", flush=True)
    scq_feat_cols = [c for c in train_scq.columns if c.startswith(f"scq{q}_")]
    X_q = np.hstack([X_grid, train_scq[scq_feat_cols].values])
    X_test_q = np.hstack([X_test_grid, test_scq[scq_feat_cols].values])
    results[f"Q={q}"] = eval_variant(f"grid+scattering Q={q}", X_q, X_test_q)

print("\n=== summary (topq, vs Q=8 reference) ===", flush=True)
ref_topq = results["Q=8 (reference, cached)"][1]
for name, (plain, topq) in results.items():
    print(f"{name}: plain={plain:.4f} topq={topq:.4f} delta_vs_Q8={topq-ref_topq:+.4f}", flush=True)
print(f"\ntotal time: {time.time()-t0:.0f}s", flush=True)
