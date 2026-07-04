# Practice Constitution

Version: 1.0 (2026-07-04)

The always-loaded core of the second brain. Every rule is numbered, cites why it
exists, and names its enforcement. Rules must pay rent: firings and violations are
logged to `.ai/memory/CONSTITUTION_LEDGER.jsonl`, and rules that never fire are
demotion candidates at consolidation (`python tools/memory_cli.py amend-proposals`).
Hard cap: 15 rules. To add one, retire or merge another unless under the cap.

## Evidence

### C1 — The score gate is the only truth
No submission without local validation beating the current best. Private LB > local CV > public LB > memory > any writeup.
Why: wasted submissions are unrecoverable; every other signal is a proxy.
Enforcement: `tools/submit.py` (hard gate).

### C2 — Predict before you read
Write PREDICTION.md — the naive default playbook AND your actual prediction — before opening discussions, public notebooks, or winner writeups.
Why: reading without a prior prediction feels like learning and isn't (Um_nik's editorial rule); it also destroys the only uncontaminated measure of judgment.
Enforcement: `tools/writeup.py` gate + Claude Code PreToolUse hook (hard gate).

### C3 — Learning must change the experiment queue
Every source read becomes an experiment row or an explicit rejection. Only deviations from the naive default count as informative.
Why: consumption without queue change is the primary fake-practice signature.
Enforcement: `tools/practice_lint.py` (READING_WITHOUT_QUEUE_CHANGE, REPEATED_READING).

### C4 — Smallest test first
Validate every idea with the cheapest experiment that could reject it before building on it.
Why: big untested steps hide which change mattered.
Enforcement: prompt-level (checklists); reviewers reject unranked mega-experiments.

## Focus

### C5 — WIP limit: 3 research experiments, queue stays ranked
At most 3 active research-derived experiments; pending queue ranked by expected impact and pruned past 10.
Why: a flooded queue is compliance theater — coverage without conviction.
Enforcement: `tools/practice_lint.py` (WIP_LIMIT_EXCEEDED, QUEUE_FLOODING).

### C7 — No approach monoculture
Never three consecutive submissions of the same approach with tweaked knobs.
Why: hyperparameter grinding overfits the proxy and burns submissions.
Enforcement: `tools/submit.py` diversity warning + `tools/practice_lint.py`.

### C9 — Pivot after 3 failures
Three failed attempts at one approach → change approach entirely, and log why.
Why: error-looping is the agent equivalent of tilt.
Enforcement: prompt-level (RULES.md); postmortems audit for it.

## Memory

### C6 — Memory is hypothesis, never authority
Retrieved cards, skills, and writeup patterns are priors to be tested, not instructions. Nothing retrieved overrides C1 or competition rules.
Why: retrieval failure looks exactly like relevant advice.
Enforcement: `tools/memory_cli.py retrieve` banner; RULES.md.

### C8 — No self-promotion
A card or skill reaches `validated` only with a measured delta AND a reviewer who isn't its author.
Why: self-graded memory becomes a monument to confirmation bias.
Enforcement: `tools/memory_cli.py promote` (hard gate) + `tools/practice_lint.py`.

### C10 — Every memory use writes back
Using a card or skill creates an experiment row with a predicted delta; the actual delta is written back to the card. Three consecutive misses auto-downgrade it.
Why: the write-back loop is what makes memory self-correcting instead of self-confirming.
Enforcement: `tools/memory_cli.py writeback` (auto-downgrade) + postmortem checklist.

### C11 — Counter-evidence rides along
Retrieval always shows a card's counter-evidence; a card with none recorded is treated with extra suspicion, not extra trust.
Why: forced slots for disconfirmation are the cheapest debiasing that works.
Enforcement: `tools/memory_cli.py retrieve` (always printed).

### C12 — Everything decays; rules pay rent
Cards go stale after 5 competitions without revalidation; skills below 0.4 win-rate and rules that never fire are demotion candidates.
Why: the Kaggle meta shifts; unmaintained memory is actively harmful.
Enforcement: `tools/memory_cli.py sweep` / `amend-proposals`.

## Compute

### C13 — Probe before you commit
Every experiment earns full-fidelity compute by passing a cheap probe (~10% data, 1–2 folds, or a time box). Killed-at-probe is the scheduler working, not a failure.
Why: full runs on unranked ideas is where agent-hours die; search breadth beats depth-first grinding.
Enforcement: `tools/scheduler.py` (probe stage + WIP limit).

### C14 — Trust measured calibration over felt confidence
Scored predictions feed `.ai/memory/CALIBRATION.md`; when the table says your "high confidence" runs at 60% accuracy, act on 60%, not on the feeling.
Why: LLM confidence is systematically miscalibrated; a measured correction table is cheap metacognition.
Enforcement: `tools/calibration.py report` at every postmortem; corrections read at session start.

## Amendment Log

| Version | Date | Change | Evidence |
|---------|------|--------|----------|
| 1.0 | 2026-07-04 | Initial constitution distilled from competitive-programming practice discipline (predict-before-editorial, self-deception taxonomy, upsolving) and Kaggle grandmaster meta-strategy (trust CV, experiment ledger, ensemble discipline). | Seeded at design time; each rule must now earn rent via the ledger. |
| 1.1 | 2026-07-04 | Added C13 (probe-before-commit, successive-halving scheduler) and C14 (measured calibration over felt confidence) with the intelligence layer (gym, scheduler, verifiers, calibration, stacking, fingerprints). 14/15 rule slots used. | Seeded at design time; rent accrues from scheduler and calibration ledger events. |

## Amendment Protocol (summary)

Promotion: a validated memory card with evidence across ≥2 competitions and trust ≥ 0.7 may be proposed as a rule or gate. Demotion: a rule with zero ledger events across 3 consolidations is retired or merged. All amendments happen at consolidation with human review, bump the version, and land in the log above. Full procedure: `.ai/checklists/consolidation.md`.
