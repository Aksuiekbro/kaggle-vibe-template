# Competition Rules

Every agent reads this file before starting work. These rules are non-negotiable.

Read `.ai/constitution.md` next — the numbered practice rules (C1–C12) behind the
enforcement tools. Three of them are hard gates, not advice: the score gate
(`tools/submit.py`), the predict-before-read gate (`tools/writeup.py`), and the
memory promotion gate (`tools/memory_cli.py promote`). The fake-practice linter
(`tools/practice_lint.py`) detects the rest and logs violations to the rent ledger.

## Core Principles

1. **Score First** — Every action must aim to improve the competition score. No yak-shaving, no premature optimization of code quality, no refactoring for its own sake. Ship solutions that score higher.

2. **Prove Before Submit** — Never submit without local validation showing improvement over the current best. Wasted submissions are unrecoverable.

3. **Own Your Work** — Build original solutions. Ensembling public notebooks is allowed but must be declared in provenance. Credit the source.

4. **Stay Autonomous** — Do not ask for human confirmation before taking the next step. If you're stuck for >3 attempts on the same error, pivot strategy. If blocked on infrastructure, document and move on.

5. **Close Learning Loops** — Predictions, plan drafts, and memory candidates must be consolidated during sharing rounds or by a designated coordinator. Drafts that never get reviewed are not memory.

## Isolation Rules

- Work ONLY in your assigned workspace: `agents/<your-name>/workspace/`
- Stage submissions in: `agents/<your-name>/submissions/`
- Read (but never write) `shared/` unless during a sharing round
- Read (but never write) other agents' workspaces
- All code, data processing, and experiments happen in YOUR workspace
- Draft shared-plan and memory updates in your workspace first. A human or designated coordinator consolidates them into `PLAN.md` or `.ai/memory/`.

## Submission Protocol

1. Run local evaluation: `python tools/evaluate.py --agent <name> --file <path>`
2. Check against registry: score must beat current best for your agent
3. Submit via: `python tools/submit.py --agent <name> --file <path> --description "<what changed>"`
4. The tool handles MCP → CLI fallback automatically
5. Record provenance: what approach, what inputs, what changed from previous best

## Anti-Overfitting Protocol

The ONLY score that matters is the PRIVATE leaderboard score. Local CV and public LB scores are proxies — treat them with suspicion.

### Mandatory Cross-Validation
- Never evaluate on a single train/test split
- Use k-fold CV (k≥5) for tabular/ML competitions
- For optimization: test on multiple problem instances, not just the ones that improve your score
- Report CV variance alongside mean score. High variance = overfitting signal.

### Submission Diversity Requirement
- Do NOT submit minor variations of the same approach
- Each submission must represent a meaningfully different strategy or significant parameter change
- The registry tracks approach type — if your last 3 submissions are the same approach with tweaked hyperparameters, STOP and try something different

### Ensemble Discipline
- Never ensemble more than you can explain. If you can't describe why combining model A and model B should work, don't do it.
- Weight validation: ensemble weights must be derived from CV, not from public LB feedback
- Track ensemble components in provenance — if one component is overfit, the ensemble inherits the problem

### Red Flags (auto-detected by evaluate.py)
- Local score dramatically higher than public score → likely overfitting
- Score improves on every fold identically → data leakage suspected
- Ensemble of 10+ models with <0.001 improvement → diminishing returns, likely fitting to noise
- Identical submission file to a previous one → wasted submission

### For Optimization Competitions
- Test solutions on UNSEEN problem instances (held-out from your optimizer)
- A solution that only works on the specific instances you optimized for is overfit
- Validate on at least 3 different random seeds / initial configurations
- Score should be robust across seeds — high variance means fragile solution

## Sharing Rounds

Sharing rounds are triggered by the human operator via `/share-round`. During a sharing round:
- Each agent's current best submission is copied to `shared/best/<agent>/`
- Agents MAY read and improve upon other agents' submissions
- Any derivative work MUST declare provenance: "based on <agent>'s submission <id>"
- Improvements go through the same score gate before submission

Outside sharing rounds:
- Do NOT read other agents' submissions or workspace
- Do NOT access `shared/best/`

## Cross-Review Protocol

After any agent achieves a new personal best score, the other two agents review the approach. Reviews are stored in `.ai/reviews/`.

### Review Format

```
# Solution Review

Reviewer: <agent name>
Subject: <agent name>'s submission <id>
Score: X.X / 10
Status: IMPROVE or SOLID

## Approach Summary
<1-3 sentences describing what the solution does>

## Strengths
<what works well about this approach>

## Improvement Ideas
<specific, actionable ideas to improve the score — not vague suggestions>

## Risks
<potential issues: overfitting, data leakage, rule violations, fragility>

## Could Combine With
<how this approach could be combined with the reviewer's own work>
```

### Scoring Rubric
- 9.0-10.0 SOLID: Strong approach, minor tweaks only
- 7.0-8.9 IMPROVE: Good foundation, specific improvements identified
- 5.0-6.9 IMPROVE: Fundamental issues but salvageable ideas
- 0.0-4.9 IMPROVE: Approach is unlikely to be competitive

### Review Rules
- Reviews are READ-ONLY — reviewers don't modify the reviewed code
- Reviews must cite specific evidence (file:line or score numbers)
- "Improvement Ideas" must be actionable, not vague ("add regularization" → bad; "add L2 regularization with lambda=0.01 to the gradient boosting model in model.py:45" → good)
- Reviews feed into the next sharing round — the original agent gets the review and can act on it

## Strategy Development

1. Read the reference strategy from `.ai/strategies/<type>.md` (shared knowledge base)
2. Read `.ai/checklists/kaggle-upsolve.md` before doing serious research
3. Create `agents/<your-name>/workspace/STRATEGY.md` — your own evolving strategy
4. Update STRATEGY.md as experiments reveal what works and what doesn't
5. Log progress in `agents/<your-name>/workspace/PROGRESS.md`

The reference strategies are starting points. Develop your own approach based on experimental evidence.

## Kaggle Upsolve Protocol

Past Kaggle winning solutions are treated like Codeforces editorials: learn the reusable pattern, state why it might work as a hypothesis, and validate it locally before trusting it.

- Start research with similar competitions, not generic tutorials
- Match sources by task type, metric, modality, split style, data shape, and leakage risk
- Prefer top 3 winner writeups per similar competition when available; record whether top approaches converge or disagree
- Convert every useful source into an experiment row: source, similarity, pattern claim, mechanism hypothesis, transfer experiment, predicted impact, cost, validation plan, risk, result
- Use prospective pattern transfer audits for real evaluation: pre-register predictions before reading current discussions or public notebooks, then score them after the competition closes
- Retrospective audits are valid only when held-out competitions are after the model's knowledge cutoff, and the cutoff date is recorded
- Alternate learning and implementation: read a small batch, extract patterns, run the smallest experiment, update STRATEGY.md, then submit only through the score gate
- Promote a pattern into STRATEGY.md as validated only after it improves validation, reduces variance, exposes leakage, or is supported by postmortem evidence
- Keep at most 3 active research-derived experiments in progress at once

## Memory Governance

Long-term memory is a source of hypotheses, not authority. Read `.ai/checklists/memory-governance.md` before adding or promoting shared memory.

- Draft memory candidates in your own workspace during active competition work
- Promote shared memory under `.ai/memory/` only during review or postmortem consolidation
- Copy or index open prediction files under `.ai/memory/predictions/` during consolidation so they can be scored later
- Memory cards use statuses: candidate, validated, rejected, superseded, stale
- A card becomes validated only after another agent or human reviews measured evidence
- Every memory-derived experiment must write back predicted-vs-actual impact
- Retrieved memory must create an experiment row before it changes a solution
- Local validation, private leaderboard evidence, competition rules, and the score gate override memory

### Memory tooling (use these, don't hand-edit lifecycle fields)

- Retrieve at anchors: `python tools/memory_cli.py retrieve --anchor classify|stuck|pre-submit|experiment-failed|postmortem --task-type <t> ...`
- Write back predicted-vs-actual: `python tools/memory_cli.py writeback --card <id> --competition <slug> --result hit|partial|miss --actual-delta <x>`
- Promotion (cross-review enforced): `python tools/memory_cli.py promote --card <id> --reviewer <other-agent>`
- Executable skills: `python tools/skills.py list|test|log-use` — verified code lives in `.ai/memory/skills/`
- Before reading discussions/notebooks/writeups: `python tools/writeup.py check --agent <name>`; log reads with `python tools/writeup.py log ...`

### Intelligence tooling

- Experiment queue (probe-first, C13): `python tools/scheduler.py add|next|record|status --agent <name>`
- Data verifiers (leaks, drift, dead columns): `python tools/verifiers.py columns|cv-lb|folds`
- Measured similarity: `python tools/fingerprint.py compute|compare`
- Calibration (C14): `python tools/calibration.py report --write` at every postmortem; read the Corrections section at session start
- Agent stacking during sharing rounds: `python tools/stack.py correlation|blend|select-finals`
- Shadow gym between competitions: `python tools/gym.py start|score|end|report` — replay finished competitions; `--arm memory-off` runs the ablation (launch agents with `KAGGLE_MEMORY_OFF=1`)

## Postmortem Protocol

After a competition closes or reliable winner writeups appear, run `.ai/checklists/postmortem.md`.

- Compare final agent approaches against top winner writeups
- Score open prospective predictions from `.ai/memory/predictions/INDEX.md`
- Diagnose gaps in validation, features, model family, postprocessing, compute allocation, leakage handling, and public-LB overfit
- Convert durable lessons into memory-card candidates
- Record which prior memory cards helped or misled the agents

## What NOT To Do

- Do not submit without beating current best local score
- Do not modify files outside your workspace (except shared/ during sharing rounds)
- Do not ask "should I continue?" — just continue
- Do not spend more than 30 minutes on environment/tooling issues — document and move on
- Do not spend more than 60 minutes consuming tutorials or winner writeups during active competition work without running at least one experiment
- Do not ignore errors by retrying the same approach — pivot after 3 failures
- Do not delete or overwrite other agents' work
- Do not optimize for local/public score at the expense of generalizability
- Do not ensemble without explaining why the combination should work
- Do not copy winner solutions blindly — upsolve the pattern, document provenance, and validate transfer first
- Do not treat memory retrieval as proof
