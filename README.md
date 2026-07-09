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

### Kaggle Upsolve Protocol
Agents study similar past competitions the way competitive programmers upsolve editorials: learn recurring winner patterns, convert them into experiments, and validate transfer before promoting them into strategy. Transfer skill is measured with pre-registered predictions and post-competition postmortems, not by random retrospective splits over famous old competitions.

### Curated Memory
Durable lessons live as reviewed Markdown/YAML memory cards under `.ai/memory/`. Memory is treated as hypotheses with evidence and counter-evidence; local validation and the score gate remain the authority.

### Consolidation
Agents draft shared-plan updates, memory candidates, and predictions in their own workspaces. `tools/share.py end` collects those drafts into `.ai/runs/<timestamp>/consolidation/`. The `/consolidate` skill then runs the whole merge as survey → decision menu → execution → verified report — the human only picks from the menu; all mechanical work, validation, and the diff summary are automated. Constitution and gate changes are never applied by the skill, only proposed.

### Second Brain (behavior change without weight updates)
A three-layer architecture (`.ai/ARCHITECTURE.md`): an always-loaded **practice
constitution** of numbered rules that must pay rent (`.ai/constitution.md`);
**hard gates and a fake-practice linter** that enforce the load-bearing rules
mechanically — predict-before-read (`tools/writeup.py` + a Claude Code hook), the
score gate, cross-review-only memory promotion, and self-deception detection
(`tools/practice_lint.py`); and **curated memory with a measured trust loop** —
cards and executable skills whose trust is their own predicted-vs-actual track
record (`tools/memory_cli.py`, `tools/skills.py`). Consolidation promotes proven
lessons upward (card → rule → gate) and demotes what stops earning its keep.

## Directory Structure

```
├── RULES.md              Global rules (all agents read this)
├── COMPETITION.md         Per-competition context
├── PLAN.md               Shared idea bank
├── CLAUDE.md / AGENTS.md / GEMINI.md   Agent-specific specs
├── .ai/
│   ├── ARCHITECTURE.md   Second-brain design + comparison with prior systems
│   ├── constitution.md   Numbered practice rules (C1–C14), versioned, rent-tracked
│   ├── prompts/          Role charters (solver, reviewer, researcher)
│   ├── checklists/       Quality gates
│   ├── memory/           Curated memory: cards, skills, predictions, ledgers
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

## Operating It

`RUNBOOK.md` is the operator manual: setup → pre-registration → 24/7 agent
loops (tmux + headless `/day` blocks) → twice-daily 10-minute check-ins →
endgame protocol → post-close postmortem → gym runs between competitions.

## Tools

| Tool | Usage | Purpose |
|------|-------|---------|
| `setup.py` | `python tools/setup.py --competition <slug>` | Initialize workspace (archives the previous competition's workspaces) |
| `evaluate.py` | `python tools/evaluate.py --agent <name> --file <path>` | Local evaluation with CV + overfitting flags |
| `submit.py` | `python tools/submit.py --agent <name> --file <path> --description "..."` | Score-gated submission (MCP → CLI fallback) |
| `share.py` | `python tools/share.py [start\|status\|end]` | Manage sharing rounds and collect consolidation drafts |
| `ensemble_optimizer.py` | `python tools/ensemble_optimizer.py --predictions m1.csv m2.csv --cv-scores 0.85 0.83 --cv-stds 0.01 0.02` | Portfolio-optimized ensemble weights (Mean-CVaR) |
| `registry.py` | `python tools/registry.py status` | View submission registry |
| `brief.py` | `python tools/brief.py generate --agent <name>` | Session-start context pack: gate, constitution, calibration, memory, skills, queue |
| `writeup.py` | `python tools/writeup.py check\|log\|hook --agent <name>` | Predict-before-read gate + reading ledger (C2) |
| `practice_lint.py` | `python tools/practice_lint.py [--agent <name>]` | Fake-practice linter (self-deception signatures) |
| `memory_cli.py` | `python tools/memory_cli.py retrieve\|writeback\|promote\|log-miss\|revalidation-due\|dedup\|sweep\|stats\|amend-proposals` | Curated memory with measured trust, write-back, retrieval health, lifecycle checks |
| `skills.py` | `python tools/skills.py list\|test\|log-use\|stats` | Executable skill library with win-rates |
| `scheduler.py` | `python tools/scheduler.py add\|next\|record\|status --agent <name>` | Successive-halving experiment queue (C13) |
| `verifiers.py` | `python tools/verifiers.py columns\|cv-lb\|folds` | Mechanical leak/drift/agreement probes |
| `calibration.py` | `python tools/calibration.py report --write` | Confidence-vs-accuracy corrections (C14) |
| `stack.py` | `python tools/stack.py correlation\|blend\|select-finals` | Agent-level ensembling + final selection |
| `fingerprint.py` | `python tools/fingerprint.py compute\|compare` | Measured competition similarity |
| `gym.py` | `python tools/gym.py start\|score\|end\|report` | Shadow gym on finished competitions (memory A/B) |
| `selfcheck.py` | `python tools/selfcheck.py` | Acceptance battery — verifies every gate and loop |

## NVIDIA GPU Acceleration (Optional)

The framework integrates NVIDIA GPU-accelerated skills that provide 10-100x speedups for specific workflows. These are optional -- everything works without a GPU, but agents will automatically use GPU tools when available.

### By Competition Type

**Optimization:**
- [cuOpt Numerical Optimization](https://docs.nvidia.com/cuopt/) -- formulate problems as LP/MILP/QP and solve on GPU. First-class approach before simulated annealing.
- [cuOpt Routing](https://docs.nvidia.com/cuopt/) -- dedicated VRP/TSP/PDP solver with real-world constraints.
- [TileGym](https://github.com/NVIDIA/TileGym) -- custom GPU kernel autotuning for compute-bound scoring functions.

**Tabular:**
- [cuDF](https://docs.rapids.ai/api/cudf/stable/) -- drop-in pandas replacement (`import cudf as pd`), 10-100x faster.
- [Data Designer](https://docs.nvidia.com/nemo/) -- synthetic tabular data generation for augmenting small/imbalanced datasets.
- [TAO AutoML](https://docs.nvidia.com/tao/) -- GPU-accelerated hyperparameter search with WandB tracking.

**Computer Vision:**
- [TAO Train](https://docs.nvidia.com/tao/) -- pre-built training pipelines for 20+ architectures (EfficientNet, DINO, RT-DETR, SegFormer, Mask2Former, etc.).
- [DALI](https://docs.nvidia.com/deeplearning/dali/) -- GPU data loading and augmentation pipeline.
- [Data Designer](https://docs.nvidia.com/nemo/) -- synthetic image generation.

**NLP:**
- [Data Designer](https://docs.nvidia.com/nemo/) -- synthetic text generation for data augmentation.

**Ensembling (all competition types):**
- `tools/ensemble_optimizer.py` -- portfolio-optimized blend weights using Mean-CVaR optimization (concept from [cufolio](https://docs.nvidia.com/cufolio/)).
- [Multi-Objective Exploration](https://docs.nvidia.com/cuopt/) -- Pareto frontier tracing for multi-metric optimization.

### Installation

Skills are installed per-session and provide API reference documentation to the agents:

```bash
# Install a specific skill
npx skills add nvidia/skills/<skill-name>

# Common installs for tabular competitions
npx skills add nvidia/skills/data-designer
npx skills add nvidia/skills/tao-run-automl
npx skills add nvidia/skills/cufolio

# Common installs for optimization competitions
npx skills add nvidia/skills/cuopt-numerical-optimization-formulation
npx skills add nvidia/skills/cuopt-routing-formulation

# Common installs for CV competitions
npx skills add nvidia/skills/tao-train-efficientnet
npx skills add nvidia/skills/dali-dynamic-mode
```

cuDF is installed via pip: `pip install cudf-cu12` (requires NVIDIA GPU with CUDA 12+).

## Agent Failure Mode Mitigations

| Agent | Known Issue | Mitigation |
|-------|------------|------------|
| Claude | Can hit rate limits after ~2hrs intensive work | Front-load hardest thinking |
| Codex | Asks "should I continue?" repeatedly | AGENTS.md explicitly forbids asking for confirmation |
| Gemini | Gets stuck in self-correction loops | GEMINI.md enforces 3-strike pivot rule |
| All | Overfitting to local/public scores | Anti-Overfitting Protocol with CV checks and diversity requirements |
| All | Wasting daily submission limit | Score gate in submit.py — must beat current best |
