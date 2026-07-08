# Idea Bank & Strategy Decisions

Shared across all agents. Track ideas, what has been tried, and what works.

During active multi-agent runs, do not let multiple agents edit this file concurrently. Agents should draft queue items in their own workspace, then a human or designated coordinator consolidates them here.

## Ideas to Try

| # | Idea | Priority | Status | Agent | Notes |
|---|------|----------|--------|-------|-------|
| 1 | [baseline approach] | HIGH | pending | - | Start here |

## What's Been Tried

| Idea | Agent | Result | Score | Why it worked/failed |
|------|-------|--------|-------|---------------------|

## Key Decisions
### Coordinator addendum — 2026-07-08

Blend submissions are 0-for-3 on LB (sub_022, sub_024, sub_029 all regressed vs
the single-pipeline champion) while single-pipeline submissions are 3-for-3
improvements (sub_020, sub_025, sub_026). The topq metric over-trusts blends.
RULE: no further blend submissions this competition. Blends may be BANKED as
decorrelated candidates for a final-two hedge only. Submission slots go to
single-pipeline improvements validated on topq.

### Coordinator note #2 — 2026-07-07 (submission freeze on untrusted proxy)

- sub_020 (grid-only): CV 0.9086 -> public 0.9195. sub_022/sub_024 (MLP blends):
  CV 0.9476/0.9523 -> public 0.879/0.878. Three data points now confirm:
  **MLP-blend-family CV is NOT a valid submission criterion** — it inflates
  under the train/test shift (exp_006 showed grid-only features are
  shift-robust; MLP components on the same features are not).
- SUBMISSION RULE until further notice: submit ONLY when the exp_006-style
  shift-aware holdout (most-test-like-quartile accuracy) improves over the
  sub_020 equivalent. Raw CV of blend models does not qualify, regardless of
  the score gate passing. Registry kaggle_score is synced — treat public LB as
  the arbiter it is.
- Blend research may continue (cheap probes), but route submission candidates
  through the shift-aware check first. Two slots were spent confirming this;
  do not spend a third.


### Coordinator note — 2026-07-06 (measured, act on this)

- CV-vs-public-LB gap is ~0.10 and **delta sign agreement is 54%** over 14
  submissions (`verifiers.py cv-lb --agent gemini`): incremental CV gains no
  longer predict LB gains. Cause is the train/test shift claude quantified
  (adversarial AUC 0.973, exp_003; loudness/noise-floor differences, exp_004).
- STOP ensemble-laddering on unchanged validation (gemini: 24-model blend gained
  nothing on LB). More blend != more score in this regime.
- PRIORITY 1: shift-aware validation — adversarially-weighted CV or a holdout
  reweighted toward test-like clips (use exp_003 classifier probabilities).
  Do not trust any CV delta until sign agreement > 70%.
- PRIORITY 2: features that survive the shift — per-clip normalization
  (exp_004, +0.034 probe) and harmonic-evidence pitch-grid features (exp_005,
  matches the pre-registered prediction).
- Submission cadence: max 2 per agent per day until the proxy is fixed.
  kaggle_score (public) is now synced into the registry — check it.

[Record strategic decisions made during the competition. Why we chose approach X over Y.]

### Coordinator note #3 — 2026-07-08 (measured, act on this: topq gate is not sufficient for blend weights)

- sub_029 (claude, exp_081: XGBoost+aug+PL blended with ExtraTrees+aug, both on
  the same shift-robust grid features) passed the note #2 shift-aware gate
  cleanly: full 3-fold CV topq 0.9674 -> 0.9708 (+0.0034), blend weight chosen
  by sweeping w_xgb on that same CV. **Public LB still reversed: 0.95402 ->
  0.93678 (-0.0172).** This is a bigger, more direct miss than anything note
  #2 was written about, and it happened on a blend of two shift-robust
  single-model configs, not a shift-robust/shift-prone mix.
- Root cause of the corrupted read: `sub_029`'s `local_score` was recorded as
  `0.0` in the registry (submit-tool bug, not a modeling bug) which briefly
  crashed `verifiers.py cv-lb` to pearson=0.235. **Patched to 0.9588 (the
  actual plain CV from the experiment log) this block.** With the fix, cv-lb
  for claude reads **pearson=0.976, delta sign agreement=62% over 9
  submissions — below the 70% trust bar.**
- New failure card: `.ai/memory/failures/blend-topq-lb-mismatch.md`.
  Hypothesis: sweeping a blend weight to maximize topq on the same folds used
  to report the delta overfits the weight itself, even when both blend arms
  individually have shift-robust CV. topq being shift-aware protects against
  blending in a shift-PRONE feature family (note #2's concern); it does not
  protect against this.
- SUBMISSION RULE update: **do not submit a prediction-level blend based on a
  CV-selected weight unless the weight was chosen on a split disjoint from
  the one the reported delta comes from** (nested CV / separate
  weight-selection holdout). Single-model (non-blend) full-CV topq
  improvements are not implicated by this finding and are not additionally
  restricted here.
- Until sign agreement is re-measured above 70%, treat every CV delta
  (blend or not) with more suspicion than usual; do not spend the shared
  daily budget confirming this same failure mode a second time.
- exp_083 (claude, in flight as of this note: 3-way blend adding
  grid+scattering-XGBoost on top of the now-falsified exp_081 2-way blend) is
  a cheap probe and was allowed to finish for information, but its result
  must not be treated as submission-worthy without first fixing the blend-
  weight-selection methodology above.

### Coordinator follow-up (claude, 2026-07-08, block 123) — nested weight selection tested; noise floor established

- Ran the fix note #3 called for: leave-one-fold-out nested weight selection,
  applied retrospectively to exp_081's already-saved OOF arrays (zero
  retraining, fold indices deterministically recomputed and verified against
  saved labels). Result (`exp_084`, `.ai/memory/failures/blend-topq-lb-mismatch.md`
  updated): the same-fold methodology's delta (+0.0034, what sub_029 was
  actually submitted on) **halves to +0.0017 nested**. This confirms the
  weight-leakage mechanism is real and roughly 2x-inflates same-fold blend
  deltas on this competition. **But the nested delta is still positive while
  the real LB delta was -0.0172 — leakage explains only part of the gap, not
  the sign reversal.**
- New working rule (applies to all agents, not just claude): **treat any
  blend-weight CV delta under ~0.01-0.02 topq as noise, whether or not the
  weight search is nested.** Nesting is necessary (do it) but not sufficient
  — a nested delta still has to clear this floor before it's a submission
  candidate. exp_083's probe (3-way blend, same-fold-style, single n=117
  holdout) found +0.0085, itself below this floor even before a nested full
  CV — full CV with nesting is running in background
  (`full_cv_exp083_threeway_blend_nested.py`) for completeness/information,
  but per this rule it is not expected to clear the bar.
- This does not change the SUBMISSION RULE from note #3 (still in effect:
  nested/disjoint weight selection required for any blend-weight candidate)
  — it adds a magnitude floor on top. Sign agreement re-measurement (currently
  62%/9 submissions, below the 70% trust bar) is still the other open
  precondition before resuming CV-driven submissions generally.

### Coordinator follow-up (claude, 2026-07-08, block 124) — exp_083 nested full CV closes prediction-level blending 5/5; feature/model-mechanism search re-confirmed exhausted

- exp_083's nested (leave-one-fold-out) full 3-fold CV landed: same-fold delta
  (what note #3's original methodology would have reported) was a small
  **+0.0017** vs the exp_081 pairwise reference; the honest **nested delta is
  -0.0051**, negative and below the ~0.01-0.02 noise floor from exp_084/note
  #3's follow-up. Recorded via `scheduler.py record --id exp_083 --stage full
  --delta -0.0051`. This is a second independent case (different blend, own
  full re-fit rather than retrospective on saved OOF) confirming the
  noise-floor rule: **prediction-level blending is now closed 5/5
  non-positive on this competition once honestly (nested) validated** —
  exp_013 (LGBM+MLP), exp_044 (LGBM+XGBoost), exp_077 (same-algo,
  different-features), exp_081/exp_084 (XGBoost+ExtraTrees, nested), exp_083
  (3-way, nested). `.ai/memory/failures/blend-topq-lb-mismatch.md` updated
  with this confirmation. No further blend-weight-search variant should be
  attempted here without a fundamentally different weight-selection
  mechanism, and given 5/5 that would still be low expected value.
- With this closure, the scheduler queue is empty and `memory_cli.py retrieve
  --anchor stuck` returns no untested candidate. This is now the **third**
  independent conclusion (blocks 113/114 research passes, block 118's
  STRATEGY.md closure note, and now this block's blend-axis closure) that the
  engineered-feature + GBDT/ExtraTrees + ensembling paradigm is exhausted for
  this competition: every mechanism family tried (grid-harmonic variants,
  wavelet scattering incl. SR/log-compression/Q/composition, domain
  adaptation/alignment, class reweighting/EM prior-shift, pseudo-labeling in
  all sub-variants, augmentation, centroid/metric-learning, label
  propagation, ordinal decomposition, GBDT/LGBM/XGBoost hyperparameters,
  frozen pretrained embeddings (CREPE/PANNs), a targeted octave-confusion
  feature (exp_034/035/036), and now prediction-level blending) is closed.
- **Recommend human/coordinator input on strategic direction** rather than
  continuing to search: current banked position is sub_027 (public 0.95402,
  CV-best) + sub_028 (public 0.94827, mechanistically-diverse hedge per the
  final-selection-by-public-lb card) — a reasonable stopping point for this
  paradigm. The one genuinely untried avenue is a from-scratch deep model
  trained on Kaggle's GPU compute via `tools/kkernel.py` (never invoked this
  competition, only referenced in a comment) — but PREDICTION.md pre-
  registered exactly this risk (too little per-class data, ~28 rows/class)
  and the closest analog tried (frozen pretrained CREPE/PANNs embeddings)
  failed outright, so expected value is uncertain and the cost (GPU quota,
  build-out time) is nontrivial. Not launched unilaterally this block; will
  hold pending guidance or occupy remaining blocks with maintenance
  (cross-review, sign-agreement tracking, calibration scoring) unless new
  gemini/codex activity or explicit direction arrives.

## Research Findings

[Shared research results — what other competitors are doing, what worked in similar competitions, relevant papers.]

## Kaggle Upsolve Queue

Use this table for winner writeups, tutorials, and similar-competition research. Each source must become an experiment or an explicit rejection.

Keep at most 3 active research-derived experiments in progress at once.

| Source | Similarity | Pattern claim | Mechanism hypothesis | Experiment | Predicted impact | Cost | Validation plan | Risk | Owner | Status | Result |
|--------|------------|---------------|----------------------|------------|------------------|------|-----------------|------|-------|--------|--------|

## Prospective Pattern Transfer Audits

Before reading current competition discussions or public notebooks, write a dated prediction. Score it after the competition closes and winner writeups appear.

| Competition | Date | Prediction file | Model cutoff | Naive default | Memory cards used | Raw hits | Informative hits | Misses | Scored date | Playbook update |
|-------------|------|-----------------|--------------|---------------|-------------------|----------|------------------|--------|-------------|-----------------|

## Postmortems

After private leaderboard reveal or reliable winner writeups, compare our final approach against top winners and write memory-card candidates.

| Competition | Agent | Final rank/score | Winner convergence | Main gap | Memory cards created | Follow-up |
|-------------|-------|------------------|--------------------|----------|----------------------|-----------|
