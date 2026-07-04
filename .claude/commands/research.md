Research approaches for the current competition.

Follow the researcher role charter in .ai/prompts/researcher.md:

1. Read COMPETITION.md to understand the problem
2. Gate check: `python tools/writeup.py check --agent claude` — if blocked, complete
   agents/claude/workspace/PREDICTION.md first (naive default + actual prediction)
3. Retrieve prior knowledge as hypotheses:
   `python tools/memory_cli.py retrieve --anchor classify --task-type <t> --metric-family <m>`
   `python tools/skills.py list --task-type <t>`
4. Search for:
   - This competition's Kaggle discussion forum
   - Winning solutions from similar past competitions (top 3, note convergence)
   - Public notebooks with high scores
   - Academic papers on the problem type
   Log each read: `python tools/writeup.py log --agent claude --url <u> --kind <k>`
5. Document findings in agents/claude/workspace/RESEARCH.md using the charter's
   output format — every source becomes an experiment row or an explicit rejection
6. Update STRATEGY.md with the most promising approaches; queue rows go to PLAN_DRAFT.md
7. Finish with `python tools/practice_lint.py --agent claude` and fix any violations
