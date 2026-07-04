# Kaggle Memory

This directory stores curated long-term memory for the solver template.

Memory cards are not raw notes. Each card is a falsifiable claim with scope, evidence, counter-evidence, status, and write-back history. Use `.ai/checklists/memory-governance.md` before adding or promoting cards.

Folders:

- `patterns/`: reusable strategy claims (cards)
- `competitions/`: postmortem-derived competition summaries — these files are also the
  staleness clock: cards go stale after 5 competitions without revalidation
- `failures/`: rejected or dangerous patterns worth remembering (always surfaced by
  `retrieve --anchor pre-submit`)
- `skills/`: executable, self-tested code with cross-competition win-rates (see its README)
- `predictions/`: open prospective predictions awaiting postmortem scoring
- `templates/`: card templates

Ledgers (append-only JSONL, written by tools — do not hand-edit):

- `CONSTITUTION_LEDGER.jsonl`: rule firings/violations — the constitution's rent record
- `skills/USAGE.jsonl`: per-skill outcomes across competitions

Work with cards through `tools/memory_cli.py` (retrieve/writeback/promote/sweep/stats),
not by hand-editing lifecycle fields. LightRAG or another retrieval system may index
this directory later, but these files remain the source of truth.

