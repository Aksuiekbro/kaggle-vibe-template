Run one full work block on the current competition — the sprint cadence that
sequences every tool in the right order. Designed to be invoked repeatedly
(interactively or from a headless loop); each invocation is one self-contained
block that ends with a report.

## 1. Wake up

- `python tools/brief.py generate --agent claude`, then read
  `agents/claude/workspace/BRIEF.md` — gate status, calibration corrections,
  relevant memory, open experiments.
- `python tools/writeup.py check --agent claude`. If BLOCKED: fill
  PREDICTION.md (naive default + actual prediction) before anything else.

## 2. First block on a new competition only

If `.ai/memory/competitions/<slug>-fingerprint.json` does not exist:
- `python tools/fingerprint.py compute --train <t.csv> --test <te.csv> --target <y> --slug <slug> --write`
  then `compare --slug <slug>` — note the measured neighbors in RESEARCH.md.
- `python tools/verifiers.py columns --train <t.csv> --target <y> --test <te.csv>` —
  act on every finding before modeling.
- Build and submit the simplest scoring baseline (establish the floor), then
  run /research if the experiment queue is empty.

## 3. Experiment loop (the bulk of the block)

Repeat until the block budget is spent (aim: 2-4 completed experiments):
1. `python tools/scheduler.py next --agent claude` — run exactly what it says
   (probe fidelity: ~10% data or 1-2 folds).
2. `python tools/scheduler.py record ...` with the honest measured delta —
   negative deltas are the scheduler working, record them.
3. Follow the write-back commands the scheduler prints for card/skill sources.
4. Queue empty → feed it: `python tools/memory_cli.py retrieve --anchor stuck`,
   PLAN.md, or a /research pass. Stuck 3 times on one approach → pivot (C9).

## 4. Submission decision (max 1-2 per block)

Only when full-fidelity CV beats the current best:
- `python tools/memory_cli.py retrieve --anchor pre-submit`
- `python tools/verifiers.py cv-lb --agent claude`
- Check today's count in `shared/submissions/registry.json` daily_counts —
  the Kaggle budget (~5/day) is SHARED across all three agents; if 4+ used
  today, hold the submission for tomorrow unless it is clearly large.
- `python tools/submit.py --agent claude --file <path> --description "<what changed>" --approach "<name>"`

## 5. Wind down (never skip)

- Update PROGRESS.md (what ran, deltas, decisions) and STRATEGY.md.
- Draft shareables into PLAN_DRAFT.md / MEMORY_CANDIDATES.md.
- `python tools/practice_lint.py --agent claude` — fix violations now.
- Final message: block report — experiments run (with deltas), submissions,
  current best, what the next block should start with.

Stay autonomous (no confirmation questions). Score first: if any step here
conflicts with improving the private-LB score, the score wins and the conflict
goes in PROGRESS.md.
