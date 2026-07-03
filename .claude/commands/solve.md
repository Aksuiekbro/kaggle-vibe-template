Start solving the current competition.

Read RULES.md, COMPETITION.md, and your STRATEGY.md in agents/claude/workspace/.

Follow this loop:
1. Check PROGRESS.md for what's been tried
2. Pick the next highest-priority experiment from your STRATEGY.md
3. Implement it in agents/claude/workspace/
4. Run local evaluation: `python tools/evaluate.py --agent claude --file <path>`
5. If score beats current best, submit: `python tools/submit.py --agent claude --file <path> --description "<what changed>" --approach "<approach name>"`
6. Update PROGRESS.md and STRATEGY.md
7. Repeat

Follow the Anti-Overfitting Protocol. Stay autonomous — do not ask for confirmation.
