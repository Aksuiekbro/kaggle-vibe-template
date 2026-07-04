# Consolidation & Amendment Protocol

Fastest path: run `/consolidate` in a Claude Code session — it executes this
checklist as survey → decision menu → execution → verified report, leaving only
the choices to the human. The steps below are the manual reference.

Run at the end of every sharing round (`python tools/share.py end` collects the
drafts) and after every postmortem. This is where episodic experience hardens
into semantic memory, and repeated semantic memory hardens into procedure —
the template's substitute for weight updates.

## 1. Consolidate drafts (human or designated coordinator)

- [ ] Open the latest `.ai/runs/<ts>/consolidation/` folder.
- [ ] Merge useful `PLAN_DRAFT.md` rows into `PLAN.md` (dedupe, keep ranking).
- [ ] For each `MEMORY_CANDIDATES.md` row worth keeping:
      `python tools/memory_cli.py new --id <slug> --claim "<claim>"`, fill scope,
      mechanism, evidence. Cards enter as `candidate` — never `validated`.
- [ ] Copy open `PREDICTION.md` files into `.ai/memory/predictions/` and index them
      in `INDEX.md`.
- [ ] Convert `POSTMORTEM.md` findings into cards under `patterns/` or `failures/`
      and a competition summary under `competitions/`.

## 2. Close the write-back loop

- [ ] `python tools/memory_cli.py retrieve --anchor postmortem` — list cards
      awaiting predicted-vs-actual write-back.
- [ ] `python tools/memory_cli.py writeback ...` for each. Misses append
      counter-evidence; three consecutive misses auto-downgrade.
- [ ] Score any closed predictions in `.ai/memory/predictions/INDEX.md`
      (per category; only deviations from the naive default count as informative hits).
- [ ] `python tools/calibration.py report --write` — refresh the confidence
      corrections agents read at session start (C14).
- [ ] If gym runs completed since last consolidation: `python tools/gym.py report` —
      check the memory-effect arm comparison and record it in PLAN.md.

## 3. Review & lifecycle

- [ ] Cross-review: another agent or the human reviews measured evidence, then
      `python tools/memory_cli.py promote --card <id> --reviewer <name>`.
- [ ] `python tools/memory_cli.py revalidation-due --agent <name>` — queue suggested probes for validated cards nearing stale.
- [ ] `python tools/memory_cli.py dedup` — flag near-duplicate claims; merge evidence manually, never auto-merge.
- [ ] `python tools/memory_cli.py sweep` — stale out unrevalidated cards.
- [ ] `python tools/memory_cli.py validate` — schema check.
- [ ] `python tools/skills.py test` — re-verify every skill still passes.

## 4. Amendments (memory hardening into procedure)

- [ ] `python tools/memory_cli.py amend-proposals` — review:
      - **Promotions**: validated cards with evidence across ≥2 competitions and
        trust ≥ 0.7 → propose a constitution rule, a lint check, or a gate.
        Prefer the strongest enforceable form: gate > lint check > constitution
        prose > checklist line.
      - **Demotions**: constitution rules with zero ledger events across 3
        consolidations → retire or merge. Skills below 0.4 win-rate → demote.
- [ ] Human approves every constitution change; bump the version and record it
      in the Amendment Log. Constitution stays ≤ 15 rules.

## Invariants

- Nothing writes directly to `.ai/memory/` during active competition work.
- No card skips `candidate`. No promotion without a reviewer (C8).
- Every amendment cites the evidence (cards, ledger counts) that justified it.
