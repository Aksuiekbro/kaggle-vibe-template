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
1. Read `COMPETITION.md` → understand the problem
2. Read the appropriate strategy from `.ai/strategies/`
3. Create your own `STRATEGY.md` in your workspace — your evolving approach
4. Start with the simplest baseline that could score
5. Iterate: improve score → validate locally → submit if better
6. Update `STRATEGY.md` and `PROGRESS.md` as you go

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

## Tools Available
- Kaggle MCP (primary) — for dataset access and submission
- `kaggle` CLI (fallback) — if MCP fails
- Python, pip, conda — for ML/data science work
- C/C++ compilers — for optimization problems
- All standard Unix tools
