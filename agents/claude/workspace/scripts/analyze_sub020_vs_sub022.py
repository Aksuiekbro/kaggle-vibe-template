"""exp_022: cheap descriptive analysis (no retraining) of WHERE sub_020
(LGBM-only, LB 0.91954) and sub_022 (LGBM+MLP blend, LB 0.87931, CV 0.9476)
disagree on the test set, to dig into the CV-up/LB-down mechanism that
exp_020's adversarial-quartile diagnostic could not explain.
"""
import numpy as np
import pandas as pd

BASE = "/root/kaggle-vibe-template/"
sub020 = pd.read_csv(BASE + "agents/claude/submissions/submission_exp005_grid_features.csv")
sub022 = pd.read_csv(BASE + "agents/claude/workspace/kernel_output_exp005/submission_exp017_blend.csv")
train = pd.read_csv(BASE + "shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv")

m = sub020.merge(sub022, on="Path", suffixes=("_020", "_022"))
assert len(m) == len(sub020) == len(sub022)
disagree = m[m.Pitch_ID_020 != m.Pitch_ID_022]
print(f"test rows: {len(m)}, disagreements: {len(disagree)} ({len(disagree)/len(m):.1%})")

train_counts = train["Pitch_ID"].value_counts()
rare_classes = set(train_counts[train_counts < 10].index)
print(f"rare classes (<10 train samples): {len(rare_classes)}/{train_counts.size}")

def rare_frac(preds):
    return preds.isin(rare_classes).mean()

print(f"sub_020 test-pred rare-class rate: {rare_frac(m.Pitch_ID_020):.4f}")
print(f"sub_022 test-pred rare-class rate: {rare_frac(m.Pitch_ID_022):.4f}")
print(f"train rare-class rate (of rows):  {train['Pitch_ID'].isin(rare_classes).mean():.4f}")

disagree_rare_020 = disagree.Pitch_ID_020.isin(rare_classes).mean()
disagree_rare_022 = disagree.Pitch_ID_022.isin(rare_classes).mean()
print(f"of disagreements, sub_020 side is rare-class: {disagree_rare_020:.4f}")
print(f"of disagreements, sub_022 side is rare-class: {disagree_rare_022:.4f}")

# predicted-class-count distribution vs train prior (chi-sq-ish spread check)
train_dist = (train_counts / train_counts.sum()).sort_index()
pred020_dist = (m.Pitch_ID_020.value_counts() / len(m)).reindex(train_dist.index, fill_value=0)
pred022_dist = (m.Pitch_ID_022.value_counts() / len(m)).reindex(train_dist.index, fill_value=0)
l1_020 = (pred020_dist - train_dist).abs().sum()
l1_022 = (pred022_dist - train_dist).abs().sum()
print(f"L1 distance pred-class-dist vs train-class-dist: sub_020={l1_020:.4f} sub_022={l1_022:.4f}")

# how many distinct classes each submission predicts at all (mode collapse check)
print(f"distinct classes predicted: sub_020={m.Pitch_ID_020.nunique()} sub_022={m.Pitch_ID_022.nunique()} (train has {train_counts.size})")

# per-class prediction-count swing: which classes did the blend predict a lot MORE/LESS of
swing = (pred022_dist - pred020_dist).sort_values()
print("\ntop 8 classes blend predicts LESS often than LGBM-only:")
print(swing.head(8))
print("\ntop 8 classes blend predicts MORE often than LGBM-only:")
print(swing.tail(8))
