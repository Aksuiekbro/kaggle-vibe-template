Run a consolidation session: turn accumulated agent drafts into curated memory,
with the human deciding and you doing all mechanical work. This implements
`.ai/checklists/consolidation.md` as a survey → menu → execute → report loop
(same shape as an improve-codebase-architecture skill: you find the candidates,
the human picks, you do the labor, the result is a reviewable diff).

## Hard scope limits (non-negotiable)

- You may modify: `.ai/memory/`, `PLAN.md`, `.ai/memory/predictions/INDEX.md`.
- You may NEVER modify: `.ai/constitution.md`, `tools/`, `.claude/`, `.codex/`,
  `RULES.md`, agent workspaces. Amendments and rule changes are PROPOSED in your
  report, never applied.
- Every promotion decision belongs to the human. Present options; do not choose.

## Phase 1 — Survey (no writes)

1. Locate the newest `.ai/runs/*/consolidation/` folder (created by
   `tools/share.py end`) and any `POSTMORTEM.md` / scored `PREDICTION.md` in it.
2. Run the mechanical scans and collect their output:
   - `python tools/memory_cli.py validate`
   - `python tools/memory_cli.py dedup`
   - `python tools/memory_cli.py sweep --dry-run`
   - `python tools/memory_cli.py revalidation-due --agent claude`
   - `python tools/memory_cli.py retrieve --anchor postmortem`
   - `python tools/memory_cli.py amend-proposals`
   - `python tools/skills.py test` and `python tools/skills.py stats`
   - `python tools/gym.py report` (if runs exist)
3. Read the drafts: each agent's `MEMORY_CANDIDATES.md`, `PLAN_DRAFT.md`.

## Phase 2 — Menu (one message, then wait)

Present ONE numbered menu of concrete decisions, grouped:

- **Memory candidates**: for each draft candidate worth keeping, show the
  proposed card (claim, scope, evidence) and recommend keep/reject with a reason.
- **Promotions**: cards with measured evidence awaiting `validated` — show the
  evidence and the reviewer who would be recorded.
- **Dedup merges**: flagged pairs with your recommended survivor.
- **Staleness/demotions**: what sweep and the skill win-rate floor would retire.
- **Amendment proposals**: constitution/gate changes suggested by
  `amend-proposals` — proposal text only, marked "requires human application".
- **Prediction scoring**: open predictions with your proposed per-category
  HIT/PARTIAL/MISS and the winner evidence for each.

Recommend a default for every item so the human can reply "do 1,3,4 as
recommended, skip 2".

## Phase 3 — Execute (only what was approved)

- Create cards: `python tools/memory_cli.py new --id <slug> --claim "..."`,
  then fill scope/mechanism/evidence in the file.
- Promote: `python tools/memory_cli.py promote --card <id> --reviewer <name>`.
- Merge duplicates: consolidate evidence into the survivor, then
  `python tools/memory_cli.py demote --card <loser> --to superseded --reason "dup of <survivor>"`.
- Apply writebacks, sweep (without --dry-run), update PLAN.md tables and
  `.ai/memory/predictions/INDEX.md`, copy scored predictions into
  `.ai/memory/predictions/`.
- Queue approved revalidation probes via `tools/scheduler.py add`.

## Phase 4 — Verify and report

1. `python tools/memory_cli.py validate` and `python tools/practice_lint.py`
   must pass; `python tools/selfcheck.py` must stay green.
2. `git status` — confirm the diff touches only allowed paths; list every
   changed file in the report.
3. Final report: what was created/promoted/retired, amendment proposals still
   awaiting human application, calibration refresh
   (`python tools/calibration.py report --write`), and the retrieval-health line.

If there is nothing to consolidate, say so in one line and stop — do not
manufacture work.
