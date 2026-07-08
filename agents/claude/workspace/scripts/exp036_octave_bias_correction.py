"""exp_036 (own, cheap, no retraining): decision-level octave-bias correction.

Follows directly from exp_034 (58% of OOF misclassifications land at
7/12/19/24-semitone harmonic-ratio distances vs 27% null). exp_035 attacks
this at the FEATURE level (adds an octave-ratio column). This attacks it at
the DECISION level instead: for rows where the top-2 predicted classes are
an octave-family distance apart and the margin between them is small, is
there a systematic direction bias (does LGBM over-predict the octave-up or
octave-down candidate relative to truth)? If so, a rule-based tiebreak
override could be a near-zero-cost fix orthogonal to exp_035's feature fix.

Fold-honesty: splits OOF rows in half, fits the bias direction on half A,
applies the resulting rule to half B, and reports the held-out delta -- not
tuned on the same rows it's scored on.

Reuses exp_017's saved oof_lgb_exp017.npy / y_exp017.npy (no refit).
"""
import numpy as np
import pandas as pd

DIR = "/root/kaggle-vibe-template/agents/claude/workspace/kernel_output_exp005/"
MAPPING = "/root/kaggle-vibe-template/agents/gemini/workspace/pitch_yin_mapping.csv"

oof = np.load(DIR + "oof_lgb_exp017.npy")
y = np.load(DIR + "y_exp017.npy")
n_rows, n_classes = oof.shape
mapping = pd.read_csv(MAPPING).set_index("Pitch_ID")

top2 = np.argsort(-oof, axis=1)[:, :2]
top1_c, top2_c = top2[:, 0], top2[:, 1]
top1_p = oof[np.arange(n_rows), top1_c]
top2_p = oof[np.arange(n_rows), top2_c]
margin = top1_p - top2_p

valid = np.isin(top1_c, mapping.index) & np.isin(top2_c, mapping.index)
d_midi = np.full(n_rows, np.nan)
d_midi[valid] = (
    mapping.loc[top1_c[valid], "median_midi"].to_numpy()
    - mapping.loc[top2_c[valid], "median_midi"].to_numpy()
)


def octave_family(ad):
    for lo, hi in [(5.5, 8.5), (10.5, 13.5), (17.5, 20.5), (22.5, 25.5)]:
        if lo <= ad < hi:
            return True
    return False


is_octave_pair = np.array(
    [valid[i] and octave_family(abs(d_midi[i])) for i in range(n_rows)]
)
print(f"rows: {n_rows}, octave-family top1/top2 pairs: {is_octave_pair.sum()} "
      f"({is_octave_pair.mean():.4f})")

MARGIN_THRESH = np.median(margin[is_octave_pair]) if is_octave_pair.any() else 0.1
print(f"using margin threshold (median over octave-pairs): {MARGIN_THRESH:.4f}")

close_octave = is_octave_pair & (margin < MARGIN_THRESH)
print(f"close-margin octave-family pairs: {close_octave.sum()} ({close_octave.mean():.4f} of all rows)")

base_acc_subset = (top1_c[close_octave] == y[close_octave]).mean()
print(f"base (top1) accuracy on this subset: {base_acc_subset:.4f} (n={close_octave.sum()})")

rng = np.random.default_rng(42)
idx = np.arange(n_rows)
rng.shuffle(idx)
half = n_rows // 2
half_a, half_b = idx[:half], idx[half:]

mask_a = np.zeros(n_rows, dtype=bool)
mask_a[half_a] = True
mask_b = ~mask_a

fit_mask = close_octave & mask_a
truth_is_top1 = (y[fit_mask] == top1_c[fit_mask]).astype(int)
truth_is_top2 = (y[fit_mask] == top2_c[fit_mask]).astype(int)
truth_is_higher = np.where(
    d_midi[fit_mask] > 0,
    truth_is_top2,  # top1 is higher (d = top1-top2 >0), so truth==top2 means truth is lower... invert carefully below
    truth_is_top1,
)
# Direction bias, defined unambiguously: for each close-octave-pair row,
# is the TRUE class the higher-pitched or lower-pitched of {top1, top2}?
higher_c = np.where(d_midi > 0, top1_c, top2_c)
lower_c = np.where(d_midi > 0, top2_c, top1_c)
truth_is_higher_all = (y == higher_c).astype(float)
truth_is_lower_all = (y == lower_c).astype(float)

fit_frac_higher = truth_is_higher_all[fit_mask].mean()
fit_frac_lower = truth_is_lower_all[fit_mask].mean()
fit_frac_other = 1 - fit_frac_higher - fit_frac_lower
print(f"\nfit half (A): P(truth=higher-of-pair)={fit_frac_higher:.4f}, "
      f"P(truth=lower-of-pair)={fit_frac_lower:.4f}, P(truth=neither)={fit_frac_other:.4f}")

rule_pick_higher = fit_frac_higher > fit_frac_lower
print(f"rule learned on half A: always predict the {'HIGHER' if rule_pick_higher else 'LOWER'}-pitched of the close-margin octave-family pair")

eval_mask = close_octave & mask_b
n_eval = eval_mask.sum()
base_acc_eval = (top1_c[eval_mask] == y[eval_mask]).mean()
rule_pred = np.where(rule_pick_higher, higher_c[eval_mask], lower_c[eval_mask])
rule_acc_eval = (rule_pred == y[eval_mask]).mean()
print(f"\nheld-out half (B), n={n_eval}: base(top1) acc={base_acc_eval:.4f}, "
      f"rule acc={rule_acc_eval:.4f}, delta={rule_acc_eval - base_acc_eval:+.4f}")

overall_base = (top1_c[mask_b] == y[mask_b]).mean()
full_pred_b = top1_c[mask_b].copy()
eval_idx_in_b = np.where(eval_mask[mask_b])[0]
full_pred_b[eval_idx_in_b] = rule_pred
overall_rule = (full_pred_b == y[mask_b]).mean()
print(f"\noverall half-B accuracy: base={overall_base:.4f}, with-rule={overall_rule:.4f}, "
      f"delta={overall_rule - overall_base:+.4f} (this is the number that matters for scheduler.py record)")
