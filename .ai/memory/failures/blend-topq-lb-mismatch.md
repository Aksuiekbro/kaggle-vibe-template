# Pattern Card

```yaml
id: blend-topq-lb-mismatch
status: candidate
created: 2026-07-08
last_validated: 2026-07-08
claim: A shift-aware topq CV metric confirming a prediction-level model blend (e.g. XGBoost+ExtraTrees) on a single shift-robust feature family can still reverse sign on the real leaderboard; ensemble blend weight selection appears to overfit CV fold structure even when the underlying feature family and single-model CV are shift-robust.
scope:
  task_type: multiclass classification, small-n (thousands of rows), many classes
  metric_family: accuracy-style / topq shift-aware validation
  modality: audio (but mechanism is about ensembling, not audio-specific)
  split_risk: adversarial train/test shift already present and measured (adv AUC ~0.97)
  data_shape: small per-class sample counts (tens of rows/class), 3-fold CV
  constraints: submission budget shared across agents, ~5/day
mechanism_hypothesis: >
evidence:
  - competition: missing-fundamental-puzzle
    source: exp_081 (own)
    our_cv_delta: +0.0034 topq (0.9674 -> 0.9708), full 3-fold CV, weight w_xgb=0.3 chosen by sweeping 0.3-0.7 on the same folds
    our_lb_delta: -0.0172 public LB (sub_027 0.95402 -> sub_029 0.93678)
    date: 2026-07-08
counter_evidence:
  - missing-fundamental-puzzle: predicted 0.0, got -0.0189
predictions:
  - date: 2026-07-08
    competition: missing-fundamental-puzzle
    predicted_delta: 0.0034
    actual_delta: -0.0172
    result: miss
  - date: 2026-07-08
    competition: missing-fundamental-puzzle
    predicted_delta: -0.01
    actual_delta: 0.0017
    result: partial
    note: Nested (leave-one-fold-out) weight selection on exp_081 OOF halves the inflated same-fold delta (+0.0034 -> +0.0017) confirming the weight-leakage mechanism is real, but nested delta stays positive while real LB was -0.0172 -- leakage is only PART of the gap, not the whole explanation. CV (even honest) unreliable at <0.01-0.02 topq margins in this shift regime.
  - date: 2026-07-08
    competition: missing-fundamental-puzzle
    predicted_delta: 0.0
    actual_delta: -0.0189
    result: miss
cost: zero to check (recompute cv-lb sign agreement before trusting a blend-weight submission); the fix (nested CV or a held-out weight-selection fold) costs one extra fold split
risk: >
supersedes: []
superseded_by: 
review:
  reviewer: 
  date: 
  verdict: 
```

## Notes

Coordinator's 2026-07-07 freeze note treated "shift-aware topq full-CV improvement"
as a sufficient submission gate for blend-family candidates (distinct from the
earlier MLP-blend-family freeze, which was about blending a shift-PRONE feature
family into a shift-robust one). exp_081/sub_029 passed that exact gate --
same grid features on both blend arms, full 3-fold CV, topq metric, +0.0034 --
and still reversed sign and magnitude on the real leaderboard (-0.0172). This
means the topq gate is necessary but not sufficient: it protects against
blending in a shift-PRONE feature family, but does not protect against
overfitting the blend WEIGHT itself to the same folds used to validate it.

Actionable fix for next blend-family candidate: choose the blend weight on a
split disjoint from the one used to report the delta (nested CV, or a
separate weight-selection holdout), and treat any single-scalar-weight-search
result as needing that nested check before it counts as submission-worthy,
even if topq/shift-aware.

Recorded 2026-07-08; escalated to `PLAN.md` coordinator note. Sign agreement
for claude is now 62% (9 submissions) after fixing the local_score data bug --
below the 70% trust bar. Do not submit further CV-driven candidates until this
is re-validated or the mechanism above is tested and fixed.

**Update (2026-07-08, exp_084, own): mechanism tested, only partially
confirmed.** Leave-one-fold-out nested weight selection applied
retrospectively to exp_081's saved OOF arrays (zero retraining -- fold
indices deterministically recomputed and verified against the saved labels)
halves the inflated same-fold delta: +0.0034 (what sub_029 was actually
submitted on) -> +0.0017 nested/honest. This confirms weight-selection
leakage is real and roughly doubles the apparent gain. But the nested delta
is still **positive** while the real LB delta was **-0.0172** -- leakage
explains only part of the gap, not the sign reversal itself. Conclusion:
nesting the weight search is necessary but still not sufficient; at this
shift severity (adv AUC ~0.97, 62% sign agreement), blend-weight CV deltas
under roughly 0.01-0.02 topq should be treated as noise regardless of
whether the weight search is nested. A blend candidate should only be
considered submission-worthy if its nested delta clears that noise floor,
in addition to being nested at all.

**Update (2026-07-08, exp_083, own): noise-floor rule holds on a fresh
independent case, axis now closed.** Applied the same nested (leave-one-fold-
out) weight-selection methodology to a *new* 3-way blend (XGBoost+aug+PL /
ExtraTrees+aug / XGBoost grid+scattering, not the same pair exp_081/exp_084
tested) with its own full 3-fold CV (not retrospective on saved OOF this
time -- full re-fit per fold). Same-fold delta (what exp_081's original,
uncorrected methodology would have reported): +0.0017 vs the exp_081 pairwise
reference -- looked like a small positive. Honest nested delta: **-0.0051**,
negative and below the ~0.01-0.02 noise floor. This is the second independent
confirmation that (a) same-fold weight selection inflates the reported delta
even when the inflation is small, and (b) the corrected/nested delta should
still be gated by the noise floor, not just its sign. Prediction-level
blending on this competition is now closed 5/5 non-positive once honestly
validated (exp_013 LGBM+MLP, exp_044 LGBM+XGBoost, exp_077 same-algorithm-
different-features, exp_081/exp_084 XGBoost+ExtraTrees nested, exp_083
3-way nested) -- no further blend-weight-search variant should be tried on
this competition without a fundamentally different weight-selection
mechanism (e.g. a properly-nested stacking meta-learner, not a scalar grid
sweep), and even that is a low-expected-value retry given 5/5 is a strong
prior. `trust`/`score` should be raised to reflect this second confirmation
when this card is next reviewed for promotion.

**Update (2026-07-08, exp_085, own): the "materially different weight-
selection mechanism" hedge is tested and also closed.** Every prior update on
this card left open the possibility that a trained meta-learner (not a scalar
grid sweep) might succeed where weight-sweeping failed. Tested it directly:
multinomial LogisticRegression stacking meta-learner over exp_083's 3 saved
OOF arrays, nested leave-one-outer-fold-out (meta-learner trained on 2 outer
folds, scored on the 3rd -- zero retraining of the base arms). Result: nested
topq -0.0189 vs the exp_081 pairwise reference -- **worse** than the scalar
weight-sweep's nested delta (-0.0051), not better. Mechanism: a trained
meta-learner has far more parameters (82-class logistic regression on 246
stacked features) than a 1-3 scalar weight search, and with only ~1550
rows/fold in the meta-training set (~19/class), it overfits the meta-level
fit itself even when nested at the outer-fold level. **Conclusion: on this
data shape (small-n, many-class, severe train/test shift), MORE combiner
flexibility makes blending worse, not better -- the fix for blend-weight
overfitting is not "use a smarter combiner," it's "don't blend."**
Prediction-level blending/stacking is now closed 6/6 non-positive by every
weight-selection method tried on this competition (fixed grid sweep
same-fold, fixed grid sweep nested, trained meta-learner nested). Do not
revisit ensembling on this competition again without a fundamentally
different data regime (meaningfully more rows/class), not just a different
combiner.

