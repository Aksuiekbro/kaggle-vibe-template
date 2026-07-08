"""exp_031: descriptive, no-retraining, no-true-labels comparison of three test
prediction sets: sub_020 (LGBM-alone, LB-best), sub_024 (LGBM+MLP blend+aug,
LB worse), exp_030 (MLP-alone, not yet LB-tested). Extends exp_022's
disagreement/rare-class/class-prior-distance diagnostic to all three.
"""
import pandas as pd
import numpy as np

BASE = "/root/kaggle-vibe-template/"
lgbm_only = pd.read_csv(BASE + "agents/claude/submissions/submission_exp005_grid_features.csv")
blend_aug = pd.read_csv(BASE + "agents/claude/workspace/kernel_output_exp005/submission_exp025_augmented_blend.csv")
mlp_only = pd.read_csv(BASE + "agents/claude/workspace/kernel_output_exp005/submission_exp030_mlp_only.csv")
train = pd.read_csv(BASE + "shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/train.csv")

for df, name in [(lgbm_only, "lgbm_only"), (blend_aug, "blend_aug"), (mlp_only, "mlp_only")]:
    df.sort_values("Path", inplace=True)
    df.reset_index(drop=True, inplace=True)

assert (lgbm_only["Path"].values == blend_aug["Path"].values).all()
assert (lgbm_only["Path"].values == mlp_only["Path"].values).all()

train_counts = train["Pitch_ID"].value_counts()
rare_classes = set(train_counts[train_counts < 10].index)
train_prior = (train_counts / train_counts.sum()).to_dict()

def rare_rate(preds):
    return np.mean([p in rare_classes for p in preds])

def prior_l1(preds):
    pred_counts = pd.Series(preds).value_counts()
    pred_freq = (pred_counts / pred_counts.sum())
    all_classes = set(train_prior) | set(pred_freq.index)
    l1 = sum(abs(train_prior.get(c, 0.0) - pred_freq.get(c, 0.0)) for c in all_classes)
    return l1

print("train rare-class(<10) rate:", rare_rate(train["Pitch_ID"].values))
for name, df in [("lgbm_only(sub_020)", lgbm_only), ("blend_aug(sub_024)", blend_aug), ("mlp_only(exp_030)", mlp_only)]:
    print(f"{name}: rare_rate={rare_rate(df['Pitch_ID'].values):.4f} prior_l1={prior_l1(df['Pitch_ID'].values):.4f}")

n = len(lgbm_only)
lgbm_v_blend = (lgbm_only["Pitch_ID"] != blend_aug["Pitch_ID"])
lgbm_v_mlp = (lgbm_only["Pitch_ID"] != mlp_only["Pitch_ID"])
blend_v_mlp = (blend_aug["Pitch_ID"] != mlp_only["Pitch_ID"])
print(f"\ndisagreement rates (n={n}):")
print(f"  lgbm vs blend : {lgbm_v_blend.mean():.4f} ({lgbm_v_blend.sum()} rows)")
print(f"  lgbm vs mlp   : {lgbm_v_mlp.mean():.4f} ({lgbm_v_mlp.sum()} rows)")
print(f"  blend vs mlp  : {blend_v_mlp.mean():.4f} ({blend_v_mlp.sum()} rows)")

all3_agree = ((lgbm_only["Pitch_ID"] == blend_aug["Pitch_ID"]) & (lgbm_only["Pitch_ID"] == mlp_only["Pitch_ID"])).mean()
print(f"  all three agree: {all3_agree:.4f}")

# Where lgbm disagrees with blend, does mlp side with lgbm or with blend?
mask = lgbm_v_blend
mlp_sides_lgbm = (mlp_only.loc[mask, "Pitch_ID"] == lgbm_only.loc[mask, "Pitch_ID"]).mean()
mlp_sides_blend = (mlp_only.loc[mask, "Pitch_ID"] == blend_aug.loc[mask, "Pitch_ID"]).mean()
mlp_sides_neither = 1 - mlp_sides_lgbm - mlp_sides_blend
print(f"\nOn the {mask.sum()} rows where lgbm-alone disagrees with the blend:")
print(f"  mlp-alone agrees with lgbm  : {mlp_sides_lgbm:.4f}")
print(f"  mlp-alone agrees with blend : {mlp_sides_blend:.4f}")
print(f"  mlp-alone picks a third answer: {mlp_sides_neither:.4f}")
