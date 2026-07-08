"""exp_034 (own, cheap, no retraining): confusion-matrix analysis on saved
OOF predictions to test a genuinely new hypothesis -- do LGBM's errors on
this task cluster at specific harmonic-ratio semitone distances (octave
~12 semitones, fifth ~7, octave+fifth ~19), consistent with the classic
"missing fundamental" failure mode (mistaking a harmonic multiple/submultiple
of the true f0 for the true f0 itself)?

Reuses exp_017's saved oof_lgb_exp017.npy / y_exp017.npy (no refit) and
gemini's published per-class median-MIDI mapping
(agents/gemini/workspace/pitch_yin_mapping.csv, built via librosa.yin on all
train clips) as a read-only reference for each Pitch_ID's approximate pitch
height -- this is reading a shared, already-cross-reviewed artifact, not
copying gemini's model.

This has not been tried this competition: exp_022/031 compared two
submissions' test predictions directly (no true labels), and exp_024
compared per-FEATURE shift vs importance -- neither looked at which classes
are confused for which, or whether errors are semitone-structured.
"""
import numpy as np
import pandas as pd

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
MAPPING = "/root/kaggle-vibe-template/agents/gemini/workspace/pitch_yin_mapping.csv"

oof_lgb = np.load(DIR + "oof_lgb_exp017.npy")
y = np.load(DIR + "y_exp017.npy")
pred = np.argmax(oof_lgb, axis=1)

mapping = pd.read_csv(MAPPING).set_index("Pitch_ID")
n_classes = oof_lgb.shape[1]

mis_mask = pred != y
n_mis = mis_mask.sum()
print(f"OOF rows: {len(y)}, misclassified: {n_mis} ({n_mis/len(y):.4f})")

rows = []
for true_c, pred_c in zip(y[mis_mask], pred[mis_mask]):
    if true_c not in mapping.index or pred_c not in mapping.index:
        continue
    d_midi = mapping.loc[pred_c, "median_midi"] - mapping.loc[true_c, "median_midi"]
    std_true = mapping.loc[true_c, "std_midi"]
    std_pred = mapping.loc[pred_c, "std_midi"]
    rows.append((true_c, pred_c, d_midi, std_true, std_pred))

df = pd.DataFrame(rows, columns=["true", "pred", "d_midi", "std_true", "std_pred"])
print(f"Matched {len(df)}/{n_mis} error rows to the YIN mapping (rest missing a class)")

# Bucket the absolute semitone distance into harmonic-ratio-relevant bins
def bucket(d):
    ad = abs(d)
    for name, lo, hi in [
        ("~0 (same-ish note, YIN noise)", 0, 1.5),
        ("~7 (fifth)", 5.5, 8.5),
        ("~12 (octave)", 10.5, 13.5),
        ("~19 (octave+fifth)", 17.5, 20.5),
        ("~24 (2 octaves)", 22.5, 25.5),
    ]:
        if lo <= ad < hi:
            return name
    return "other"

df["bucket"] = df["d_midi"].apply(bucket)
print("\nError distance bucket counts (matched errors only):")
print(df["bucket"].value_counts())
print(f"\nfraction in octave-or-fifth-family buckets (7/12/19/24): "
      f"{df['bucket'].isin(['~7 (fifth)', '~12 (octave)', '~19 (octave+fifth)', '~24 (2 octaves)']).mean():.4f}")

# Null comparison: what fraction of ALL class pairs (not just errors) fall in
# those buckets, to see if errors are enriched vs random chance
all_pairs = []
classes = mapping.index.to_numpy()
rng = np.random.default_rng(42)
sample_classes = rng.choice(classes, size=min(2000, len(classes) ** 2), replace=True)
sample_classes2 = rng.choice(classes, size=min(2000, len(classes) ** 2), replace=True)
d_null = mapping.loc[sample_classes, "median_midi"].to_numpy() - mapping.loc[sample_classes2, "median_midi"].to_numpy()
null_df = pd.DataFrame({"d_midi": d_null})
null_df["bucket"] = null_df["d_midi"].apply(bucket)
null_frac = null_df["bucket"].isin(["~7 (fifth)", "~12 (octave)", "~19 (octave+fifth)", "~24 (2 octaves)"]).mean()
print(f"null (random class pair) fraction in same buckets: {null_frac:.4f}")

# Does high within-class YIN std (ambiguous fundamental) predict higher error rate?
per_class_err = pd.DataFrame({"true": y, "err": mis_mask.astype(float)}).groupby("true")["err"].mean()
merged = per_class_err.to_frame().join(mapping[["std_midi", "count"]], how="inner")
corr = merged["err"].corr(merged["std_midi"])
print(f"\ncorr(per-class error rate, per-class YIN std_midi): {corr:.4f} (n={len(merged)} classes)")
print(merged.sort_values("std_midi", ascending=False).head(10))
