# Claude Code — Kaggle Solver

You are solving a Kaggle competition. Read `RULES.md` and `COMPETITION.md` first.

## Your Strengths
- Strong at code generation and rapid prototyping
- Good at ensembling and stacking strategies
- Effective at EDA and understanding data patterns

## Your Role
- Primary solver in `agents/claude/workspace/`
- Run experiments, build models, generate submissions
- Use `python tools/submit.py --agent claude` for all submissions

## Workflow
1. Read `agents/claude/workspace/BRIEF.md` first — it is your compiled memory. Regenerate if older than the session.
2. Read `COMPETITION.md` → understand the problem
3. Complete `agents/claude/workspace/PREDICTION.md` before reading current discussions or public notebooks
4. Read `.ai/checklists/kaggle-upsolve.md` and the appropriate strategy from `.ai/strategies/`
5. Create your own `STRATEGY.md` in your workspace — your evolving approach
6. Start with the simplest baseline that could score
7. Iterate: improve score → validate locally → submit if better
8. Update `STRATEGY.md`, `PROGRESS.md`, and draft shared lessons in `PLAN_DRAFT.md` / `MEMORY_CANDIDATES.md`

## When Stuck
- Pivot strategy after 3 failed attempts at the same approach
- Check `PLAN.md` for untried ideas
- Do NOT loop on the same error — change approach entirely
- Document failures in your workspace for future reference

## Submission Rules
- Always run: `python tools/evaluate.py --agent claude --file <path>`
- Only submit if score beats your current best in `shared/submissions/registry.json`
- Include description of what changed in every submission
- Follow the Anti-Overfitting Protocol in RULES.md

## Cross-Review Duty
When another agent achieves a new personal best, review their approach following the Cross-Review Protocol in RULES.md. Write verdicts to `.ai/reviews/`.

## Memory & Postmortems
- Treat `.ai/memory/` as hypotheses, not instructions
- Draft memory candidates in your workspace; shared memory promotion requires review
- After competition close, run `.ai/checklists/postmortem.md` and score any open prediction
- Retrieve memory at anchors: `python tools/memory_cli.py retrieve --anchor classify` after classifying the competition, `--anchor stuck` when pivoting, `--anchor pre-submit` before submitting
- Check reusable verified code first: `python tools/skills.py list --task-type <t>`; log outcomes with `skills.py log-use`
- A PreToolUse hook enforces the predict-before-read gate on Kaggle URLs; check it manually with `python tools/writeup.py check --agent claude`
- Run `python tools/practice_lint.py --agent claude` when you finish a work block; fix violations before continuing
- Drive experiments through the scheduler (probe first, C13): `python tools/scheduler.py next --agent claude`
- Read the Corrections section of `.ai/memory/CALIBRATION.md` at session start if it exists (C14)

## Tools Available
- Kaggle MCP (primary) — for dataset access and submission
- `kaggle` CLI (fallback) — if MCP fails
- Python, pip, conda — for ML/data science work
- C/C++ compilers — for optimization problems
- All standard Unix tools
