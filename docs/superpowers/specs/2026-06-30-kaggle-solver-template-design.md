# Kaggle Solver Template — Design Spec

## Context

Based on analysis of a real experiment (chat.md) where 3 AI agents (Claude Code, Codex, Gemini) competed on Kaggle Santa 2025, validated by cross-agent research (Claude + Codex 5.5 consensus). The experiment revealed specific failure modes per agent, cross-contamination issues, and infrastructure problems. This template codifies the lessons into a reusable framework.

Key problems this solves:
- Agents asking for confirmation instead of working autonomously
- Agents getting stuck in self-correction loops
- No provenance tracking when agents build on each other's work
- Wasted submissions on unvalidated solutions
- Overfitting to local/public scores instead of optimizing for private LB
- No structured way to share work between agents

## Architecture: Flat Workspace with Convention

Markdown rules for agent behavior. Python tools for mechanical enforcement. Agents work in isolation by default with explicit sharing rounds for collaboration.

### Directory Structure

```
kaggle-solver-template/
├── RULES.md                          # Global competition rules (all agents)
├── COMPETITION.md                    # Filled per competition
├── PLAN.md                           # Shared idea bank
├── CLAUDE.md                         # Claude Code agent spec
├── AGENTS.md                         # Codex agent spec
├── GEMINI.md                         # Gemini agent spec
├── .ai/
│   ├── prompts/
│   │   ├── solver.md                 # Solver role charter
│   │   ├── reviewer.md               # Cross-review charter
│   │   └── researcher.md             # Research role charter
│   ├── checklists/
│   │   ├── submission-gate.md        # Pre-submission quality gate
│   │   ├── scoring-rubric.md         # Solution scoring rubric
│   │   └── sharing-round.md          # Sharing round rules
│   ├── strategies/
│   │   ├── optimization.md           # Reference: optimization competitions
│   │   ├── tabular.md                # Reference: tabular ML
│   │   ├── nlp.md                    # Reference: NLP competitions
│   │   ├── cv.md                     # Reference: computer vision
│   │   └── code.md                   # Reference: code competitions
│   └── reviews/
│       └── TEMPLATE.md               # Cross-review verdict template
├── agents/
│   ├── claude/
│   │   ├── workspace/                # Claude's isolated working directory
│   │   └── submissions/              # Claude's staged submissions
│   ├── codex/
│   │   ├── workspace/
│   │   └── submissions/
│   └── gemini/
│       ├── workspace/
│       └── submissions/
├── shared/
│   ├── submissions/
│   │   └── registry.json             # Central submission ledger
│   ├── best/                         # Sharing round artifacts
│   └── data/                         # Competition data (gitignored)
├── tools/
│   ├── submit.py                     # Score-gated submitter (MCP → CLI fallback)
│   ├── evaluate.py                   # Local evaluation with CV + overfitting flags
│   ├── share.py                      # Sharing round manager
│   ├── setup.py                      # Competition workspace initializer
│   └── registry.py                   # Submission registry manager
├── .claude/
│   ├── settings.local.json           # Claude Code permissions
│   └── commands/
│       ├── setup-competition.md
│       ├── solve.md
│       ├── submit.md
│       ├── share-round.md
│       ├── status.md
│       └── research.md
├── Makefile
├── README.md
└── .gitignore
```

## RULES.md — Global Competition Rules

All agents read this file. It encodes behavioral rules, isolation boundaries, and quality gates.

### Core Principles

1. **Score First** — Every action must aim to improve the competition score. No yak-shaving, no premature optimization of code quality, no refactoring for its own sake.
2. **Prove Before Submit** — Never submit without local validation showing improvement over the current best. Wasted submissions are unrecoverable.
3. **Own Your Work** — Build original solutions. Ensembling public notebooks is allowed but must be declared in provenance.
4. **Stay Autonomous** — Do not ask for human confirmation before taking the next step. If stuck for >3 attempts on the same error, pivot strategy. If blocked on infrastructure, document and move on.

### Isolation Rules

- Work ONLY in `agents/<your-name>/workspace/`
- Stage submissions in `agents/<your-name>/submissions/`
- Read (never write) `shared/` unless during a sharing round
- Read (never write) other agents' workspaces
- All code, data processing, experiments happen in YOUR workspace

### Submission Protocol

1. Run local evaluation: `python tools/evaluate.py --agent <name> --file <path>`
2. Check against registry: score must beat current best for your agent
3. Submit: `python tools/submit.py --agent <name> --file <path> --description "<what changed>"`
4. Tool handles MCP → CLI fallback automatically
5. Record provenance: approach, inputs, what changed from previous best

### Anti-Overfitting Protocol

The ONLY score that matters is the PRIVATE leaderboard score. Local CV and public LB scores are proxies — treat them with suspicion.

**Mandatory Cross-Validation:**
- Never evaluate on a single train/test split
- Use k-fold CV (k≥5) for tabular/ML competitions
- For optimization: test on multiple problem instances, not just the ones that improve your score
- Report CV variance alongside mean score — high variance = overfitting signal

**Submission Diversity Requirement:**
- Do NOT submit minor variations of the same approach
- Each submission must represent a meaningfully different strategy or significant parameter change
- Registry tracks approach type — if last 3 submissions are same approach with tweaked hyperparameters, STOP and try something different

**Ensemble Discipline:**
- Never ensemble more than you can explain
- Ensemble weights must come from CV, not from public LB feedback
- Track ensemble components in provenance

**Red Flags (auto-detected by evaluate.py):**
- Local score dramatically higher than public score → likely overfitting
- Score improves on every fold identically → data leakage suspected
- Ensemble of 10+ models with <0.001 improvement → diminishing returns
- Identical submission file to a previous one → wasted submission

**For Optimization Competitions:**
- Test solutions on UNSEEN problem instances
- Validate on at least 3 different random seeds / initial configurations
- Score should be robust across seeds — high variance means fragile solution

### Sharing Rounds

Triggered by human operator. During a sharing round:
- Each agent's current best is copied to `shared/best/<agent>/`
- Agents MAY read and improve upon other agents' submissions
- Derivative work MUST declare provenance
- Improvements go through the same score gate

Outside sharing rounds: do NOT access `shared/best/` or other agents' work.

### Cross-Review Protocol

After any agent achieves a new personal best, the other two agents review the approach. Reviews stored in `.ai/reviews/`.

**Review verdict format:**

```
# Solution Review

Reviewer: <agent name>
Subject: <agent name>'s submission <id>
Score: X.X / 10
Status: IMPROVE or SOLID

## Approach Summary
<1-3 sentences>

## Strengths
<what works well>

## Improvement Ideas
<specific, actionable — cite file:line or score numbers>

## Risks
<overfitting, data leakage, rule violations, fragility>

## Could Combine With
<how this could combine with reviewer's own work>
```

**Scoring rubric:**
- 9.0-10.0 SOLID: Strong approach, minor tweaks only
- 7.0-8.9 IMPROVE: Good foundation, specific improvements identified
- 5.0-6.9 IMPROVE: Fundamental issues but salvageable ideas
- 0.0-4.9 IMPROVE: Approach unlikely to be competitive

**Review rules:**
- Read-only — reviewers don't modify the reviewed code
- Must cite specific evidence
- "Improvement Ideas" must be actionable, not vague
- Reviews feed into the next sharing round

### Strategy Development

Each agent:
1. Reads the reference strategy from `.ai/strategies/<type>.md` (shared knowledge base)
2. Creates `agents/<name>/workspace/STRATEGY.md` — their own evolving strategy
3. Updates STRATEGY.md as experiments reveal what works and what doesn't

The reference strategies are starting points. Each agent develops its own approach based on experimental evidence.

## Agent Specs

### CLAUDE.md (Claude Code / Opus)

```markdown
# Claude Code — Kaggle Solver

Read RULES.md and COMPETITION.md first.

## Your Strengths
- Strong at code generation and rapid prototyping
- Good at ensembling and stacking strategies
- Effective at EDA and understanding data patterns

## Your Role
- Primary solver in `agents/claude/workspace/`
- Run experiments, build models, generate submissions
- Use `python tools/submit.py --agent claude` for all submissions

## Workflow
1. Read COMPETITION.md → understand problem
2. Read appropriate strategy from `.ai/strategies/`
3. Create your STRATEGY.md in your workspace
4. Start with the simplest baseline that could score
5. Iterate: improve score → validate locally → submit if better
6. Update STRATEGY.md and PROGRESS.md as you go

## When Stuck
- Pivot strategy after 3 failed attempts at the same approach
- Check PLAN.md for untried ideas
- Do NOT loop on the same error — change approach entirely

## Submission Rules
- Always run: `python tools/evaluate.py --agent claude --file <path>`
- Only submit if score beats your current best
- Include description of what changed in every submission

## Tools Available
- Kaggle MCP (primary), kaggle CLI (fallback)
- Python, pip, conda
- C/C++ compilers
- All standard Unix tools
```

### AGENTS.md (Codex)

```markdown
# Codex — Kaggle Solver

Read RULES.md and COMPETITION.md first.

## Your Strengths
- Strong at writing optimized C++ code
- Good at hyperparameter search and batch optimization
- Reliable at iterative improvement

## Your Role
- Primary solver in `agents/codex/workspace/`
- Focus on writing efficient solvers and optimizers
- Use `python tools/submit.py --agent codex` for all submissions

## Critical: Do NOT Ask for Confirmation
Work autonomously. Do not ask "should I continue?" or "should I run
the next batch?" Just run it. The human will interrupt if needed.
This is your most important behavioral rule.

## Workflow
1. Read COMPETITION.md → understand problem
2. Read appropriate strategy from `.ai/strategies/`
3. Create your STRATEGY.md in your workspace
4. Build a solver (prefer C++ for optimization, Python for ML)
5. Run hyperparameter sweeps in batches
6. Submit improvements via the score-gated tool
7. Update STRATEGY.md and PROGRESS.md as you go

## When Stuck
- Try a different algorithm or heuristic
- Check PLAN.md for untried ideas
- If MCP fails, use kaggle CLI:
  `kaggle competitions submit -c <comp> -f <file> -m "<msg>"`

## Sandbox
- Write in `agents/codex/workspace/`
- Read-only: RULES.md, COMPETITION.md, PLAN.md, `.ai/strategies/`
- During sharing rounds: read-only access to `shared/best/`
```

### GEMINI.md (Gemini CLI)

```markdown
# Gemini — Kaggle Solver

Read RULES.md and COMPETITION.md first.

## Your Strengths
- Built-in Google Search for research
- Good at finding and synthesizing public approaches
- Strong at implementing known algorithms from papers

## Your Role
- Primary solver in `agents/gemini/workspace/`
- Leverage search to find winning approaches from similar competitions
- Use `python tools/submit.py --agent gemini` for all submissions

## Critical: Avoid Self-Correction Loops
If you encounter an error:
1. First attempt: fix the error directly
2. Second attempt: try a different approach to the same goal
3. Third attempt: STOP entirely. Write what failed in
   `agents/gemini/workspace/FAILURES.md` and pivot to a completely
   different strategy.
DO NOT retry the same fix more than twice.

## Workflow
1. Read COMPETITION.md → understand problem
2. Use Google Search to find approaches from similar past competitions
3. Read appropriate strategy from `.ai/strategies/`
4. Create your STRATEGY.md in your workspace
5. Implement the most promising approach found via research
6. Validate locally before submitting
7. Update STRATEGY.md and PROGRESS.md as you go

## Research Protocol
- Search Kaggle discussions for this specific competition
- Search for similar past competition solutions
- Search academic papers for the problem type
- Document findings in `agents/gemini/workspace/RESEARCH.md`

## Error Handling
- Do NOT delete files to "start fresh" unless you've saved progress
- Do NOT retry the same command expecting different results
- If a library fails after 2 attempts, try an alternative
- If compilation fails after 2 attempts, switch to Python
```

## Python Tools

### tools/submit.py

```
Usage: python tools/submit.py --agent <name> --file <path> --description "<text>"

Flow:
1. Load shared/submissions/registry.json
2. Run evaluate.py → get local score + CV stats + flags
3. Compare against agent's current best in registry
4. If not better → REJECT (print score comparison)
5. If better → attempt Kaggle MCP submission
6. If MCP fails → fallback to kaggle CLI
7. Log result to registry.json (score, provenance, flags, method)
8. Update best_scores
9. Display any overfitting warnings from evaluate.py
```

### tools/evaluate.py

```
Usage: python tools/evaluate.py --agent <name> --file <path>

Returns JSON:
{
  "score": 0.8234,
  "metric": "normalized_area",
  "cv_scores": [0.8190, 0.8245, 0.8210, 0.8280, 0.8245],
  "cv_mean": 0.8234,
  "cv_std": 0.0033,
  "flags": [],
  "fold_count": 5
}

Flags: HIGH_VARIANCE, DIMINISHING_RETURNS, POSSIBLE_LEAKAGE,
       DUPLICATE_SUBMISSION, FORMAT_ERROR

For optimization competitions without ground truth: validates format,
checks constraints, runs heuristic scoring if possible.
```

### tools/share.py

```
Usage: python tools/share.py [start|status|end]

start: copies best submissions + approach code to shared/best/<agent>/
status: shows sharing round state and derivatives produced
end: locks shared/best/, logs derivatives, closes round
```

### tools/setup.py

```
Usage: python tools/setup.py --competition <kaggle-slug>

Creates directory structure, downloads data, generates COMPETITION.md
template, initializes registry.json.
```

### shared/submissions/registry.json

```json
{
  "competition": "<slug>",
  "submissions": [
    {
      "id": "sub_001",
      "agent": "<name>",
      "timestamp": "<ISO 8601>",
      "file": "<path>",
      "local_score": 0.0,
      "kaggle_score": null,
      "cv_std": 0.0,
      "flags": [],
      "method": "mcp|cli",
      "description": "<what changed>",
      "provenance": {
        "type": "original|derivative|ensemble",
        "based_on": null,
        "approach": "<approach name>"
      },
      "status": "submitted|rejected|failed"
    }
  ],
  "best_scores": {
    "claude": null,
    "codex": null,
    "gemini": null
  },
  "daily_counts": {}
}
```

## Strategy Reference Guides

Shared reference material in `.ai/strategies/`. Agents read these as starting points then develop their own `STRATEGY.md` in their workspace.

Each guide covers:
- **Approach order** — from simplest baseline to advanced techniques
- **Implementation guidelines** — language choice, parallelism, checkpointing
- **Common pitfalls** — typical mistakes for this competition type
- **What worked before** — known successful approaches from past competitions

Five guides: optimization.md, tabular.md, nlp.md, cv.md, code.md.

## COMPETITION.md Template

Filled per competition. Contains: overview (URL, type, deadline, limits), problem description, evaluation metric + CV variance threshold, data description, submission format, known constraints, initial research, baseline scores and medal thresholds.

## Operating Model

```
Human: setup.py → fill COMPETITION.md → launch agents in terminals
  │
  ├── Claude (reads CLAUDE.md + RULES.md + strategy → develops own STRATEGY.md)
  ├── Codex  (reads AGENTS.md + RULES.md + strategy → develops own STRATEGY.md)
  └── Gemini (reads GEMINI.md + RULES.md + strategy → develops own STRATEGY.md)
       │
       ▼
  tools/submit.py (score gate + evaluate.py + MCP→CLI fallback + registry)
       │
       ▼ (on new personal best)
  Cross-Review Protocol (other agents review, scored verdicts)
       │
       ▼ (human triggers)
  Sharing Round (share.py → agents improve each other's work → provenance tracked)
       │
       ▼ (repeat until deadline)
```

## Verification

After implementation, verify by:
1. Run `python tools/setup.py --competition playground-series-s6e1` on a test competition
2. Verify directory structure created correctly
3. Manually test submit.py with a dummy submission — confirm score gate rejects bad scores
4. Test evaluate.py returns correct JSON format with CV stats
5. Test share.py start/end lifecycle
6. Launch Claude Code with CLAUDE.md and verify it reads rules correctly
7. Confirm registry.json tracks submissions with provenance
8. Test MCP → CLI fallback by simulating MCP failure
