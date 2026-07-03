# Competition Rules

Every agent reads this file before starting work. These rules are non-negotiable.

## Core Principles

1. **Score First** — Every action must aim to improve the competition score. No yak-shaving, no premature optimization of code quality, no refactoring for its own sake. Ship solutions that score higher.

2. **Prove Before Submit** — Never submit without local validation showing improvement over the current best. Wasted submissions are unrecoverable.

3. **Own Your Work** — Build original solutions. Ensembling public notebooks is allowed but must be declared in provenance. Credit the source.

4. **Stay Autonomous** — Do not ask for human confirmation before taking the next step. If you're stuck for >3 attempts on the same error, pivot strategy. If blocked on infrastructure, document and move on.

## Isolation Rules

- Work ONLY in your assigned workspace: `agents/<your-name>/workspace/`
- Stage submissions in: `agents/<your-name>/submissions/`
- Read (but never write) `shared/` unless during a sharing round
- Read (but never write) other agents' workspaces
- All code, data processing, and experiments happen in YOUR workspace

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
2. Create `agents/<your-name>/workspace/STRATEGY.md` — your own evolving strategy
3. Update STRATEGY.md as experiments reveal what works and what doesn't
4. Log progress in `agents/<your-name>/workspace/PROGRESS.md`

The reference strategies are starting points. Develop your own approach based on experimental evidence.

## What NOT To Do

- Do not submit without beating current best local score
- Do not modify files outside your workspace (except shared/ during sharing rounds)
- Do not ask "should I continue?" — just continue
- Do not spend more than 30 minutes on environment/tooling issues — document and move on
- Do not ignore errors by retrying the same approach — pivot after 3 failures
- Do not delete or overwrite other agents' work
- Do not optimize for local/public score at the expense of generalizability
- Do not ensemble without explaining why the combination should work
