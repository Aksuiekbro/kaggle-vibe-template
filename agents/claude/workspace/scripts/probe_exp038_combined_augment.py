"""exp_038 (own, probe, scheduler-tracked): combine exp_037's two winning
single-factor augmentation changes -- wider_threshold (rare-class cutoff
<15 instead of <10) and more_oversample (3 SNR levels x 4 stretch rates,
richer than exp_018's original 2x2) -- which exp_037 only ever tested
independently. Same single 80/20 holdout + topq shift-aware metric as
exp_037, so results are directly comparable to that probe's table.

Runs concurrently with exp_037's full-CV promotion (exp038_fullcv_more_
oversample.py) on this box's 2nd core -- both single-threaded (n_jobs=1).
"""
import sys
import time
import warnings

import lightgbm as lgb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/kaggle-vibe-template/agents/claude/workspace/scripts")
from extract_grid_features import DATA_ROOT, MIDI_NOTES, N_HARMONICS, TOL, TARGET_RMS
from exp025_augmented_blend import extract_from_signal
from probe_exp037_augment_tuning import augment_rows

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
TRAIN_CSV = "/root/kaggle-vibe-template/shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv"
SEED = 42

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2,
    subsample=0.8, colsample_bytree=0.8, objective="multiclass",
    verbosity=-1, n_jobs=1,
)

CONFIGS = {
    "baseline_exp018": dict(threshold=10, snrs=(20.0, 12.0), rates=(0.9, 1.1)),
    "more_oversample": dict(threshold=10, snrs=(20.0, 15.0, 10.0), rates=(0.85, 0.9, 1.1, 1.15)),
    "combined": dict(threshold=15, snrs=(20.0, 15.0, 10.0), rates=(0.85, 0.9, 1.1, 1.15)),
}

t0 = time.time()
train = pd.read_parquet(DIR + "train_grid_features.parquet")
test = pd.read_parquet(DIR + "test_grid_features.parquet")
feat_cols = [c for c in train.columns if c not in ("Path", "Pitch_ID")]

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

X_adv = np.vstack([X_all, test[feat_cols].values])
y_adv = np.concatenate([np.zeros(len(X_all)), np.ones(len(test))])
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
q75 = np.quantile(w_full, 0.75)
top_mask = w_va >= q75
print(f"adversarial AUC: {adv_auc:.4f} ({time.time()-t0:.0f}s), n_topq={top_mask.sum()}", flush=True)

t_b = time.time()
clf_base = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
clf_base.fit(X_tr, y_tr)
pred_base = clf_base.predict(X_va)
correct_base = (pred_base == y_va).astype(float)
plain_b, weighted_b, topq_b = (
    correct_base.mean(),
    np.average(correct_base, weights=w_va),
    correct_base[top_mask].mean(),
)
print(f"no-augment baseline ({time.time()-t_b:.0f}s): plain={plain_b:.4f} weighted={weighted_b:.4f} topq={topq_b:.4f}", flush=True)

results = {"no_augment": (plain_b, weighted_b, topq_b)}
for name, cfg in CONFIGS.items():
    t_c = time.time()
    rare_classes = set(full_counts[full_counts < cfg["threshold"]].index.tolist())
    rare_tr_rows = tr_rows[tr_rows["Pitch_ID"].isin(rare_classes)]
    rng_master = np.random.default_rng(SEED)
    aug_df = augment_rows(rare_tr_rows, rng_master, cfg["snrs"], cfg["rates"])
    X_tr_aug = np.vstack([X_tr, aug_df[feat_cols].values])
    y_tr_aug = np.concatenate([y_tr, le.transform(aug_df["Pitch_ID"].values)])

    clf_aug = lgb.LGBMClassifier(num_class=n_classes, random_state=42, **LGBM_PARAMS)
    clf_aug.fit(X_tr_aug, y_tr_aug)
    pred_aug = clf_aug.predict(X_va)
    correct_aug = (pred_aug == y_va).astype(float)
    plain_a = correct_aug.mean()
    weighted_a = np.average(correct_aug, weights=w_va)
    topq_a = correct_aug[top_mask].mean()
    results[name] = (plain_a, weighted_a, topq_a)
    print(f"{name} ({len(rare_tr_rows)} rare rows -> {len(aug_df)} extra, "
          f"{time.time()-t_c:.0f}s): plain={plain_a:.4f} (d{plain_a-plain_b:+.4f}) "
          f"weighted={weighted_a:.4f} (d{weighted_a-weighted_b:+.4f}) "
          f"topq={topq_a:.4f} (d{topq_a-topq_b:+.4f})", flush=True)

print("\n=== summary vs no-augment baseline ===", flush=True)
for name, (p, wt, tq) in results.items():
    print(f"{name}: plain={p:.4f} weighted={wt:.4f} topq={tq:.4f}", flush=True)
print(f"total time: {time.time()-t0:.0f}s", flush=True)
