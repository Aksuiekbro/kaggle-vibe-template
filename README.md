# Kaggle Solver Template

Multi-agent framework for solving Kaggle competitions. Three AI agents (Claude Code, Codex, Gemini) work in isolated workspaces with score-gated submissions, cross-review protocols, and structured sharing rounds.

## Quick Start

```bash
# 1. Install dependencies
pip install kaggle

# 2. Set up Kaggle credentials
# Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY

# 3. Initialize a competition
python tools/setup.py --competition <kaggle-competition-slug>

# 4. Fill in COMPETITION.md with problem understanding

# 5. Launch agents in separate terminals
# Terminal 1: Claude Code (reads CLAUDE.md + RULES.md)
# Terminal 2: Codex CLI (reads AGENTS.md + RULES.md)
# Terminal 3: Gemini CLI (reads GEMINI.md + RULES.md)
```

## Operating Model

```
Human: setup → fill COMPETITION.md → launch agents → trigger share rounds
  │
  ├── Claude  (CLAUDE.md) ──┐
  ├── Codex   (AGENTS.md) ──┼── Each works in agents/<name>/workspace/
  └── Gemini  (GEMINI.md) ──┘   Each develops own STRATEGY.md
       │
       ▼
  tools/submit.py  (score gate + evaluate.py + MCP→CLI fallback)
       │
       ▼  (on new personal best)
  Cross-Review Protocol  (other agents review, scored verdicts)
       │
       ▼  (human triggers)
  Sharing Round  (share.py → agents improve each other's work)
       │
       ▼  (repeat until deadline)
```

## Key Concepts

### Isolation by Default
Each agent works in `agents/<name>/workspace/`. They cannot read or write other agents' files. This prevents cross-contamination and ensures independent solutions.

### Score-Gated Submissions
No submission reaches Kaggle without beating the agent's current best local score. Uses `tools/submit.py` which tries Kaggle MCP first, falls back to `kaggle` CLI.

### Anti-Overfitting
Built-in guardrails: mandatory cross-validation, submission diversity requirements, CV variance warnings, red flag detection. See the Anti-Overfitting Protocol in `RULES.md`.

### Cross-Review
When an agent achieves a new personal best, the other two review the approach and provide scored verdicts with actionable improvement ideas.

### Sharing Rounds
Human-triggered collaboration: each agent's best work is shared, and agents can improve on each other's submissions. All derivative work tracks provenance.

### Evolving Strategies
Agents read shared reference strategies from `.ai/strategies/` as starting points, then develop their own `STRATEGY.md` in their workspace based on what experiments reveal.

## Directory Structure

```
├── RULES.md              Global rules (all agents read this)
├── COMPETITION.md         Per-competition context
├── PLAN.md               Shared idea bank
├── CLAUDE.md / AGENTS.md / GEMINI.md   Agent-specific specs
├── .ai/
│   ├── prompts/          Role charters (solver, reviewer, researcher)
│   ├── checklists/       Quality gates
│   ├── strategies/       Reference strategy guides per competition type
│   └── reviews/          Cross-review verdicts
├── agents/
│   ├── claude/           Claude's workspace + submissions
│   ├── codex/            Codex's workspace + submissions
│   └── gemini/           Gemini's workspace + submissions
├── shared/
│   ├── submissions/      registry.json (submission ledger)
│   ├── best/             Sharing round artifacts
│   └── data/             Competition data (gitignored)
└── tools/                Python orchestration scripts
```

## Tools

| Tool | Usage | Purpose |
|------|-------|---------|
| `setup.py` | `python tools/setup.py --competition <slug>` | Initialize workspace for a competition |
| `evaluate.py` | `python tools/evaluate.py --agent <name> --file <path>` | Local evaluation with CV + overfitting flags |
| `submit.py` | `python tools/submit.py --agent <name> --file <path> --description "..."` | Score-gated submission (MCP → CLI fallback) |
| `share.py` | `python tools/share.py [start\|status\|end]` | Manage sharing rounds |
| `registry.py` | `python tools/registry.py status` | View submission registry |

## Agent Failure Mode Mitigations

| Agent | Known Issue | Mitigation |
|-------|------------|------------|
| Claude | Can hit rate limits after ~2hrs intensive work | Front-load hardest thinking |
| Codex | Asks "should I continue?" repeatedly | AGENTS.md explicitly forbids asking for confirmation |
| Gemini | Gets stuck in self-correction loops | GEMINI.md enforces 3-strike pivot rule |
| All | Overfitting to local/public scores | Anti-Overfitting Protocol with CV checks and diversity requirements |
| All | Wasting daily submission limit | Score gate in submit.py — must beat current best |
