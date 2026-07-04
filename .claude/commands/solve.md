Start solving the current competition.

Read RULES.md, .ai/constitution.md, COMPETITION.md, and your STRATEGY.md in
agents/claude/workspace/. If .ai/memory/CALIBRATION.md exists, read its
Corrections section and weight your own confidence accordingly (C14).

First EDA hour:
- Fingerprint the data and find measured neighbors:
  `python tools/fingerprint.py compute --train <t.csv> --test <te.csv> --target <y> --slug <comp> --write`
  `python tools/fingerprint.py compare --slug <comp>`
- Run mechanical verifiers before trusting anything:
  `python tools/verifiers.py columns --train <t.csv> --target <y> --test <te.csv>`

Experiment loop (scheduler-driven, C13):
1. Feed ideas into the queue: `python tools/scheduler.py add --agent claude --idea "..."
   --predicted-delta <x> --cost-hours <h> [--source card:<id>|skill:<id>|url:<u>]`
   (sources: STRATEGY.md, PLAN.md, `memory_cli.py retrieve --anchor classify|stuck`)
2. `python tools/scheduler.py next --agent claude` → run the PROBE it names
   (~10% data or 1-2 folds), record: `scheduler.py record --id <id> --stage probe --delta <x>`
3. Survivors get full CV; record with `--stage full`. Follow the write-back
   commands the scheduler prints for memory- and skill-derived experiments.
4. Check the skill library before writing risky plumbing: `python tools/skills.py list`
5. Evaluate: `python tools/evaluate.py --agent claude --file <path>`
6. Before submitting: `python tools/memory_cli.py retrieve --anchor pre-submit`
   and `python tools/verifiers.py cv-lb --agent claude`
7. If score beats current best, submit: `python tools/submit.py --agent claude
   --file <path> --description "<what changed>" --approach "<approach name>"`
8. Update PROGRESS.md and STRATEGY.md; repeat from step 2.

Follow the Anti-Overfitting Protocol. Stay autonomous — do not ask for confirmation.
Run `python tools/practice_lint.py --agent claude` at the end of each work block.
