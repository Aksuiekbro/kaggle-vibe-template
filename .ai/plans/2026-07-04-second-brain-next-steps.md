# Second Brain — Next Steps Implementation Plan

Audience: Codex (or any implementing agent). Date: 2026-07-04.
Tasks are ordered by leverage — implement in order, one task fully (including
its acceptance checks) before starting the next.

## Ground rules (read first)

- Python stdlib only. No new dependencies.
- Follow the existing tool patterns: helpers come from `tools/discipline.py`
  (`PROJECT_ROOT` respects `KAGGLE_TEMPLATE_ROOT` for sandbox testing,
  `ledger_event`, `read_jsonl`, `append_jsonl`, `workspace`).
- Reuse, don't reimplement: card parsing/trust scoring lives in
  `tools/memory_cli.py` (`all_cards`, `trust_score`, `retrieval_score`,
  `competitions_since_validation`) — import it (`sys.path.insert` pattern used
  by every tool).
- Do NOT change the semantics of existing gates (`writeup.py`, `submit.py`,
  `memory_cli.py promote`) or the card schema.
- Every new behavior gets a check in `tools/selfcheck.py` (same style as the
  existing sandbox checks). Definition of done for the whole plan:
  `python tools/selfcheck.py` exits 0 with your new checks included.
- Update `README.md`'s Tools table and `.ai/ARCHITECTURE.md` for anything you add.
- Test in a sandbox via `KAGGLE_TEMPLATE_ROOT=<tmpdir>`, never against live state.

---

## Task 1 — `tools/brief.py`: session-start context pack (highest leverage)

**Why:** agents start every session amnesiac; instruction-following decays.
One generated briefing replaces four "remember to read X" instructions.

**Command:** `python tools/brief.py generate --agent <claude|codex|gemini>`
writes `agents/<agent>/workspace/BRIEF.md` (overwrite each time; it is a
derived artifact, never hand-edited).

**BRIEF.md sections, in order:**
1. Header: competition slug (from `shared/submissions/registry.json`),
   generation timestamp, agent name.
2. **Gate status**: result of `prediction_gate_status(agent)` from
   `discipline.py` — one line: OPEN or BLOCKED with reasons.
3. **Constitution digest**: parse `.ai/constitution.md` for `### C<n> — <title>`
   headings; list id + title only (one line each). Do not inline full rule text.
4. **Your calibration corrections**: if `.ai/memory/CALIBRATION.md` exists, copy
   its "Corrections to inject into agent context" section verbatim; else one
   line: "No calibration data yet."
5. **Relevant memory (hypotheses, not instructions)**: top 5 cards by
   `retrieval_score` with an empty query (or scoped by the current competition's
   fingerprint file if `.ai/memory/competitions/<slug>-fingerprint.json` exists —
   map `target_kind` regression/classification → `task_type` query). Show id,
   status, trust, claim, counter-evidence. Honor `KAGGLE_MEMORY_OFF`: replace
   the section with "MEMORY OFF (ablation arm)".
6. **Skills**: id, status, win-rate from `tools/skills.py` usage log. Also
   honors `KAGGLE_MEMORY_OFF`.
7. **Experiment queue**: from `agents/<agent>/workspace/EXPERIMENTS.jsonl` —
   in-flight experiments first, then top-3 queued by the scheduler's
   `priority()`. If empty: "Queue empty — feed it (research / retrieve --anchor stuck)."
8. Footer: "Regenerate with `python tools/brief.py generate --agent <name>`.
   Memory is hypothesis (C6); every use needs an experiment row (C10)."

**Wiring:**
- `.claude/settings.json`: add a `SessionStart` hook running
  `python3 "$CLAUDE_PROJECT_DIR/tools/brief.py" generate --agent claude || true`.
- `.codex/hooks.json`: add a `SessionStart` hook (same event schema) running
  `python3 "$(git rev-parse --show-toplevel)/tools/brief.py" generate --agent codex || true`.
- `Makefile`: `brief:` target with `AGENT=` variable.
- `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`: replace the "read CALIBRATION.md at
  session start" bullet with "Read `workspace/BRIEF.md` first — it is your
  compiled memory. Regenerate if older than the session."
- `GEMINI.md` (no hooks): make generating the brief the explicit first step.

**Acceptance (add to selfcheck):**
- brief generates in the sandbox and contains "Constitution digest",
  "Gate status", and "Experiment queue" sections.
- with `KAGGLE_MEMORY_OFF=1`, the memory section says "MEMORY OFF" and lists no cards.
- brief generation exits 0 even when registry/calibration/queue files are absent
  (empty-state robustness).

---

## Task 2 — Retrieval-miss counter (the LightRAG trigger metric)

**Why:** "add a retrieval index" should be a decision made by evidence of
retrieval failure, not by enthusiasm. This creates the evidence stream.

**Changes:**
- `tools/memory_cli.py`: new subcommand
  `log-miss --card <id> --competition <slug> --reason "<why it should have surfaced>"`
  → appends `{ts, card, competition, reason}` to `.ai/memory/RETRIEVAL_MISSES.jsonl`.
  Validate the card exists first.
- `.ai/checklists/postmortem.md`: add a step after the memory scorecard:
  "For each card that SHOULD have influenced this competition but was never
  retrieved, run `memory_cli.py log-miss ...`."
- `tools/memory_cli.py amend-proposals`: new section "## Retrieval health" —
  total misses, misses in the last 2 consolidations, and the line:
  "≥3 misses across 2 consolidations, or corpus >100 cards → evaluate a derived
  retrieval index (see plan Task 5)."

**Acceptance:** `log-miss` writes the JSONL entry; `amend-proposals` output
contains "Retrieval health"; selfcheck exercises both in the sandbox.

---

## Task 3 — Revalidation scheduling (spaced repetition for memory)

**Why:** `sweep` retires stale cards passively; better to refresh valuable ones
before they expire.

**Changes:**
- `tools/memory_cli.py retrieve`: when a returned card has
  `competitions_since_validation(card) >= 3` and status `validated`, append a
  line to its display: "NEARING STALE (<n>/5) — schedule a revalidation probe."
- New subcommand `revalidation-due`: list validated cards with
  `competitions_since_validation >= 3`, each with a ready-to-paste command:
  `python tools/scheduler.py add --agent <name> --idea "revalidate: <claim
  (first 60 chars)>" --predicted-delta <last hit's actual_delta or 0.001>
  --cost-hours 0.5 --source card:<id>`.
- `.ai/checklists/consolidation.md`: add "run `memory_cli.py revalidation-due`
  and queue suggested probes" to section 3.

**Acceptance:** sandbox card with 3 competition files newer than
`last_validated` shows the warning in retrieve output and appears in
`revalidation-due`; selfcheck covers it.

---

## Task 4 — Semantic dedup at consolidation

**Why:** three agents drafting cards will produce near-duplicate claims;
duplicates dilute trust tracking (evidence splits across copies).

**Changes:**
- `tools/memory_cli.py`: new subcommand `dedup` — pairwise similarity over all
  cards' `claim` fields using token Jaccard (lowercase, strip punctuation, drop
  stopwords {the,a,an,of,to,in,on,for,and,or,is,are,with,by}); flag pairs with
  similarity ≥ 0.5. For each flagged pair print both ids, the score, and:
  "merge evidence into the older card, mark the newer `superseded`
  (`demote --to superseded`), set `superseded_by`."
  Do NOT auto-merge — human/coordinator decides.
- `.ai/checklists/consolidation.md`: add `dedup` to section 3.

**Acceptance:** sandbox with two cards sharing ≥ 0.5 claim overlap → `dedup`
flags the pair and exits 1; unrelated cards exit 0. Selfcheck covers both.

---

## Task 5 — LightRAG derived index (DO NOT BUILD YET — conditional)

**Trigger (from Task 2 data):** ≥3 logged retrieval misses across 2
consolidations, OR corpus exceeds ~100 cards. Until then this task is frozen.

**When triggered, the integration invariants (non-negotiable):**
1. Index ONLY `.ai/memory/` (cards, skills metadata, competition summaries) —
   never raw transcripts or chat logs.
2. Markdown stays the source of truth; the index is derived and rebuilt at
   consolidation, never written to directly.
3. Retrieval results must carry each card's `status`, trust score, and
   counter-evidence through to the agent — a raw text chunk without its
   epistemic metadata is a governance regression.
4. Retrieved content remains hypothesis (C6); the score gate and local
   validation stay the only authority.
5. Prefer the graph/entity mode (technique ↔ condition ↔ competition-type
   relations) over flat chunk embedding — flat lookup is already handled by
   scope filters and grep.

---

## Final step

Run `python tools/selfcheck.py` — must exit 0 with all new checks green.
Then update `.ai/ARCHITECTURE.md` (brief generator joins the Intelligence
Layer section; retrieval-miss counter joins Definition of Done) and the
README Tools table. Record what was implemented in the constitution's
amendment log ONLY if any rule text changed (none of these tasks should
require that).
