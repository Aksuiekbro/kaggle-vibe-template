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
