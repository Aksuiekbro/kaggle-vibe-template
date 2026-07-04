# Second-Brain Architecture

How this template changes agent behavior without weight updates. Version 1.0 (2026-07-04).

The design principle: **retrieval changes what an agent knows; harness structure
changes what it does.** Declarative knowledge lives in memory cards and gets
retrieved; procedural discipline lives in gates and hooks and gets *enforced*.
Adaptation is the consolidation pipeline that moves lessons between layers.

## The Three Layers

```
Layer 1  CONSTITUTION      .ai/constitution.md          always loaded, ≤15 rules,
         (procedural core)                              versioned, rules pay rent

Layer 2  GATES & SENSORS   tools/submit.py    (C1)      hard gates: violations are
         (harness)         tools/writeup.py   (C2)      impossible or immediately
                           tools/memory_cli.py promote (C8)   detected, not discouraged
                           tools/practice_lint.py       linter: self-deception
                           .claude/settings.json hooks  signatures → context injection

Layer 3  CURATED MEMORY    .ai/memory/patterns/         falsifiable claim cards with
         (semantic +       .ai/memory/failures/         scope, trust, counter-evidence
          episodic +       .ai/memory/competitions/     postmortems (episodic)
          procedural-as-   .ai/memory/skills/           executable, self-tested code
          code)            .ai/memory/predictions/      pre-registered judgments
```

### Data flows

1. **Anchor push (retrieval at decision points).** Agents don't have to remember to
   ask: workflow checklists call `memory_cli.py retrieve --anchor <classify|stuck|
   pre-submit|experiment-failed|postmortem>` at the moments where the corresponding
   knowledge matters. Free-form pull (grep, retrieve with scope flags) remains available.
2. **Write-back loop.** Every card/skill use carries a predicted delta; the actual
   delta is written back (`memory_cli.py writeback`, `skills.py log-use`). Trust is
   the Laplace-smoothed hit-rate of the card's own track record. Three consecutive
   misses auto-downgrade. Memory discredits itself without anyone deciding to.
3. **Rent ledger.** Gates and the linter log every rule firing/violation to
   `CONSTITUTION_LEDGER.jsonl`. Rules that never fire are demotion candidates.
4. **Consolidation (the adaptation step).** At sharing-round end and postmortems
   (`.ai/checklists/consolidation.md`): episodic drafts → candidate cards →
   (cross-review) → validated cards → (amend-proposals, human-approved) →
   constitution rules or new gates/lint checks. Demotion runs in the same pass.
   This is episodic → semantic → procedural consolidation — the human "practice
   until it's habit" loop, implemented in the only substrate an LLM agent has: the
   harness, versioned in git.

### Trust and staleness are measured, not opined

- `trust(card) = (1 + hits + 0.5·partials) / (2 + n_predictions)` — from the card's
  own predicted-vs-actual record, never from a model's judgment of importance.
- Staleness is counted in **competitions, not wall-clock**: postmortem files landing
  in `memory/competitions/` after a card's `last_validated` tick its clock; 5 ticks → stale.
- Retrieval score = scope-match × trust × recency × status-weight. Counter-evidence
  always prints alongside; failure cards always surface at `pre-submit`.

## Comparison With Prior Agent-Memory Systems

What each system's signature mechanism is, where this architecture adopts it, and
where it goes further. "Further" claims are structural, not benchmarked — see
Honest Limits.

### Voyager (skill library of verified code)
Adopted: skills as executable code, verified before storage, retrieved by relevance.
Beyond it:
- **Continuous re-verification.** Voyager verifies a skill once, in the environment
  state where it was born. Here `skills.py test` re-runs every `self_test()` at every
  consolidation — verification is a regression suite, not an event.
- **Cross-competition win-rates.** Voyager has no notion of a skill that *used to*
  work; here every use logs win/neutral/loss, and a <0.4 win-rate over ≥3 uses flags
  demotion. Skills can be un-learned.
- **Scope predicates.** Voyager retrieves by embedding similarity; here skills carry
  explicit applicability conditions (task type, metric family), so a tabular trick
  can't leak into an NLP competition on cosine vibes.
- **Curriculum, grounded.** Voyager's automatic curriculum is an LLM proposing the
  next task from progress; here the analog is the expected-impact-ranked experiment
  queue (WIP-limited) plus `retrieve --anchor stuck`, which surfaces untested
  candidate cards as next experiments — proposals come from evidence-carrying
  memory, not free generation.

### Reflexion (verbal self-feedback across episodes)
Adopted: episodic self-critique (postmortems, PROGRESS logs) injected into future
attempts.
Beyond it:
- **Reflections are validated before they persist.** Reflexion stores whatever the
  agent says about its own failure; here a lesson enters as `candidate` and needs a
  measured delta plus a *different* agent's review to become trusted. Self-serving
  reflections die in review.
- **Reflections are enforced, not advisory.** Reflexion's lessons compete for
  attention in the next episode's context; here the recurring ones become gates the
  next episode physically cannot bypass.
- **Cross-agent reflection.** Three agents review each other's postmortems — a
  failure one agent rationalizes, another flags.

### Generative Agents (memory stream + reflection + recency/importance/relevance retrieval)
Adopted: scheduled reflection (consolidation), scored retrieval, memory decay.
Beyond it:
- **Importance is earned, not rated.** Generative Agents ask the LLM to score its own
  memories' importance; here the analogous weight is the card's measured hit-rate —
  a track record, not an opinion.
- **Decay is domain-clocked.** Recency decays per *competition*, matching how the
  Kaggle meta actually shifts, instead of wall-clock time.
- **Counter-evidence is a first-class field.** The memory stream has no slot for
  "this belief failed"; here every retrieval shows the card's failures, and a card
  with no recorded counter-evidence is flagged as less trustworthy, not more.

### MemGPT / Letta (tiered memory, self-editing)
Adopted: tiers (constitution = core memory, workspace = working memory, cards/skills
= archival), agent-initiated edits.
Beyond it:
- **Review-gated self-editing.** MemGPT lets the agent rewrite its own core memory
  mid-conversation — drift by design. Here core-memory (constitution) edits happen
  only at consolidation, with evidence attached, human-approved, git-versioned, and
  reversible.
- **Eviction by rent, not recency.** MemGPT pages memory by staleness of access;
  here constitution rules are evicted for *not preventing anything* (zero ledger
  events) and cards for *predicting wrongly* — usefulness, not usage.

### Karpathy's system-prompt learning (learning accumulates in an evolving prompt)
Adopted: the constitution is exactly this — a versioned, evolving instruction core.
Beyond it:
- **A promotion pipeline with evidence thresholds** (validated card, ≥2 competitions,
  trust ≥0.7) instead of ad-hoc prompt editing.
- **An enforcement tier above prose.** The proposal stops at "the prompt learns";
  here the strongest lessons graduate past prose into gates and lint checks, because
  prompt text decays under long-session pressure and gates don't.
- **Demotion.** An evolving prompt monotonically grows; the rent ledger shrinks it.

### Agent Workflow Memory (induce reusable workflows from trajectories)
Adopted: postmortems induce reusable procedure (checklist steps, skills) from what
actually happened.
Beyond it:
- **Induction from failures too.** AWM learns workflows from successful trajectories;
  the `failures/` folder and gap-diagnosis postmortems mine the losses, which is
  where Kaggle's expensive lessons live (leakage, shake-ups, public-LB overfit).
- **Induced procedure is tested.** A workflow induced from one trajectory may be
  coincidence; here it ships with a predicted delta and gets written back against
  reality like any other card.

### Property no prior system has
**Adversarial multi-agent governance of shared memory.** Voyager, Reflexion,
Generative Agents, MemGPT, and AWM are all single-agent: one mind grading its own
homework. Here three heterogeneous agents (Claude/Codex/Gemini) share one curated
memory, and the write path runs through an agent that didn't author the claim.
Combined with the prediction gate, this also gives the system something none of them
attempt: an **uncontaminated measure of whether memory is working** — pre-registered
predictions, scored against private leaderboards, attributed to the cards that
influenced them.

## The Intelligence Layer

The layers above make the system *disciplined*; this layer makes it *smarter in
effect* without touching weights. Four sources, seven tools (all stdlib):

1. **Session-start context — the brief generator** (`tools/brief.py`). Agents
   begin with one compiled `BRIEF.md`: prediction-gate state, constitution digest,
   calibration corrections, relevant memory, skills, and the ranked experiment
   queue. It honors `KAGGLE_MEMORY_OFF`, so ablation runs do not leak curated
   memory through the context pack. This makes "read the right things first" a
   generated artifact rather than a fragile instruction pile.
2. **Ground truth on demand — the shadow gym** (`tools/gym.py`). Finished
   competitions with late submission still score against the real private LB.
   `gym.py start --arm memory-on|memory-off` replays them end-to-end (the
   `KAGGLE_MEMORY_OFF=1` env var disables retrieval and the skill library for the
   ablation arm), `score --fetch` pulls the private score, `end` restores live
   state and demands the postmortem loop, `report` accumulates the memory-effect
   A/B in percentile points. This converts the cold-start problem into a
   curriculum: trust scores, skills, calibration, and the rent ledger all burn in
   against reality in days instead of competition-months. Contamination inflates
   both arms equally; the delta stays meaningful.
3. **Search — the successive-halving scheduler** (`tools/scheduler.py`, C13).
   Experiments live in a persistent per-agent queue ranked by
   predicted-delta/cost with a novelty bonus; every idea runs a cheap probe
   first, dies at probe cost if the signal isn't there, and only survivors get
   full CV. `next` respects the C5 WIP limit; `record` routes memory-derived
   results straight into card write-backs and skill logs.
4. **Verification — mechanical probes** (`tools/verifiers.py`). Exploits the
   generation/verification asymmetry: ID-column/target correlation (ordering
   leaks), train/test drift per column, constant/duplicate columns, identical-fold
   detection, and CV-vs-LB agreement tracking from the registry. Each check turns
   a silent catastrophic failure into a loud flag before it costs a submission.
5. **Calibration — measured metacognition** (`tools/calibration.py`, C14). Joins
   confidence labels from scored PREDICTION.md files with their HIT/PARTIAL/MISS
   outcomes; reports accuracy per confidence level and per category, plus the
   informative-deviation rate (the honest measure of edge); writes correction
   lines agents read at session start.
6. **Diversity harvesting — agent-level stacking** (`tools/stack.py`). Pairwise
   prediction correlation across the three agents' submissions, CV-weighted
   blending, and decision-theoretic final selection: CV-best plus the
   least-correlated hedge within CV tolerance — never by public LB.
7. **Measured similarity — problem fingerprints** (`tools/fingerprint.py`).
   Dataset statistics (size, type mix, missingness, target shape, per-column
   drift) as a vector; retrieval and upsolve-source selection key on fingerprint
   distance to past competitions instead of self-reported similarity.
   Fingerprints accumulate in `memory/competitions/` from gym runs and postmortems.

## Definition of Done

"The harness works" is a falsifiable claim, not a feeling: `python tools/selfcheck.py`
runs the acceptance battery — static parsing of all tools, unit self-tests, card
schema validation, and functional sandbox tests of every gate and loop (C2 gate
block/hook/open, session brief generation and memory-off behavior, retrieval-miss
logging and retrieval-health reporting, C8 promote refusal and pass, C10
auto-downgrade, revalidation warnings and due probes, semantic dedup, C5 WIP
block, C13 probe kill, the ablation switch, calibration join, blend arithmetic,
fingerprint write, lint detection, hook wiring, rule-tool pairing, rule-count cap).
Exit 0 = verified. Run it after any change to `tools/` or `.ai/`, and at every
consolidation. What selfcheck cannot certify is *effectiveness* — that is the gym
A/B's job (n≥5 per arm), and the calibration report's.

## Honest Limits

- **Skill selection, not skill acquisition.** Context is not weights: the system
  changes which experiments run and what gets trusted; it cannot make the base model
  natively better at writing code.
- **Structural claims, not benchmarked ones — until gym runs accumulate.** "Beyond"
  above means the mechanism dominates on design (covers the same case plus a
  failure mode the original doesn't). The proving A/B — memory-on vs memory-off,
  private-LB percentile — is now runnable on demand via `tools/gym.py`; the claims
  stay structural until `gym.py report` shows n≥5 per arm.
- **Gate coverage varies by harness.** Claude Code (`.claude/settings.json`) and
  Codex CLI (`.codex/hooks.json`, hooks engine stable since v0.124.0) both enforce
  C2 mechanically via PreToolUse hooks — covering web fetches *and* gated URLs
  embedded in shell commands (curl/wget). Gemini CLI has no equivalent hook
  mechanism, so its gate is a mandatory charter step backstopped by the linter.
  Residual holes for all agents: content reached via search-result snippets, and
  any tool path a harness cannot intercept. Bypasses that use the sanctioned
  tools' `--force` flags are always ledger-logged; fully out-of-band submissions
  are caught after the fact by `verifiers.py reconcile`.
- **Cold start.** Trust scores start at 0.5 with empty ledgers; the system's edge
  over its own baseline appears only after the first few postmortems feed back.

## Operating It

| Moment | Command |
|--------|---------|
| New competition | `make setup COMP=<slug>` (scaffolds PREDICTION.md — the gate key) |
| Session start | `make brief AGENT=<name>` (hooks do this automatically for Claude/Codex) |
| Before reading anything | `make gate-check AGENT=<name>` |
| After classifying | `fingerprint.py compute --write` + `fingerprint.py compare` + `memory_cli.py retrieve --anchor classify ...` + `skills.py list` |
| First EDA hour | `verifiers.py columns --train ... --target ... --test ...` |
| Picking work | `scheduler.py next` → probe → `record` → full |
| Before submitting | `memory_cli.py retrieve --anchor pre-submit` + `verifiers.py cv-lb` |
| Sharing round | `stack.py correlation` / `blend` / `select-finals` |
| End of work block | `make lint` |
| Sharing round end | `tools/share.py end` → `.ai/checklists/consolidation.md` (`revalidation-due`, `dedup`, `amend-proposals`) |
| Competition close | `.ai/checklists/postmortem.md` → writebacks → `calibration.py report --write` → `make amend` |
| Between competitions | `gym.py start --arm ...` → replay → `gym.py report` |
