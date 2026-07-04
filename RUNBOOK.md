# Operator Runbook — from "found a competition" to 24/7

You are the operator. Your total hands-on time: ~30 min at start, ~10 min twice
a day, ~1 h at the end. Everything else runs unattended.

## Phase 0 — Setup (once, ~5 min of your attention)

Your only truly-manual step: **accept the competition rules on kaggle.com** in
your browser (Kaggle requires a logged-in click; the API/CLI cannot do it, and
data downloads fail until it's done).

Then open a Claude Code session and type:

```
/setup-competition <slug>
```

The agent does the rest: downloads data (Kaggle MCP/CLI), scaffolds workspaces
(archiving the previous competition's), fills COMPETITION.md including the
Metric/Direction fields the score gate needs, and runs selfcheck. It will tell
you if anything needs you.

(Every `make xyz` in this runbook is just a shortcut for a `python tools/...`
command — see the Makefile. You never need make; you can always ask an agent
to run the underlying tool, or run it yourself.)

## Phase 1 — Pre-registration (before ANY reading)

Each agent must fill its own `agents/<name>/workspace/PREDICTION.md` (naive
default playbook + actual prediction). The C2 hooks enforce this for Claude and
Codex — they cannot read discussions/notebooks until it's done, so make it each
agent's literal first task. Don't write predictions for them; the prediction is
the measurement of *their* judgment.

## Phase 2 — Launch 24/7 (the Ralph loops)

One tmux session, one window per agent. Each window runs a headless loop that
invokes one work block, then restarts (survives crashes and rate-limit pauses):

```bash
tmux new -s kaggle

# window 1 — Claude
while true; do
  claude -p "/day" --permission-mode acceptEdits
  echo "block done $(date)"; sleep 300
done

# window 2 — Codex (adjust flags to your installed version)
while true; do
  codex exec "Read AGENTS.md and run one full work block on the current competition, ending with a block report."
  sleep 300
done

# window 3 — Gemini (same pattern via its headless/prompt mode)
while true; do
  gemini -p "Read GEMINI.md and run one full work block on the current competition."
  sleep 300
done
```

Notes:
- `/day` is the sprint skill — it sequences brief → gate → scheduler →
  verify → submit → lint per block, so the loop body stays one line.
- Rate limits are expected: a block that dies on limits just restarts after the
  sleep. Raise the sleep if you keep hitting caps.
- **Submission budget**: Kaggle allows ~5/day for the ONE account all agents
  share. `/day` tells agents to check `daily_counts` and hold at 4+. If two
  agents fight over the last slot, that's your call at the next check-in.
- Permission modes: `acceptEdits` keeps a safety floor. Fully unattended runs
  need broader permissions — prefer an allowlist in `.claude/settings.json`
  over blanket skips, and only on a machine you'd let an agent loose on.

## Phase 3 — Your ritual (2 × 10 min per day)

Morning and evening:
1. `make status` — scores, submissions, flags.
2. `make lint` — self-deception check across all agents.
3. Skim each `PROGRESS.md` tail — are the blocks producing deltas or churn?
4. Every 1–2 days: `python tools/share.py start` → let agents run one block
   (they can now read each other's best) → `python tools/share.py end` →
   run `/consolidate` in a Claude session and answer its menu.
   That last step is the entire human bottleneck now: one menu.

Intervene only when: lint keeps flagging the same agent (read its transcript),
scores flatline for a day (trigger a /research pass or a pivot instruction), or
submission slots are being wasted on <0.001 deltas (tell them to hold).

## Phase 4 — Endgame (last 48 h) — this is where competitions are won/lost

Switch the system from exploration to exploitation:
1. Tell each agent: no new probe experiments; finish in-flight fulls only.
2. Sharing round + `python tools/stack.py correlation --files <agent bests>` —
   blend if off-diagonal correlation < 0.95: `stack.py blend` (weights from CV).
3. `python tools/stack.py select-finals --files ... --cv-scores ...` —
   CV-best + decorrelated hedge.
4. On kaggle.com, mark those two as your final submissions. NEVER pick finals
   by public LB (failure card: final-selection-by-public-lb).

## Phase 5 — After close (~1 h, highest-value hour in the system)

When the private LB and winner writeups appear:
1. `.ai/checklists/postmortem.md` per agent (freeze state first).
2. Score every PREDICTION.md; `python tools/memory_cli.py writeback ...` for
   every card used; `skills.py log-use` for every skill; `log-miss` for cards
   that should have surfaced but didn't.
3. `python tools/calibration.py report --write`.
4. `/consolidate` — this is where the competition becomes permanent memory.

## Phase 6 — Between competitions (optional but compounding)

`python tools/gym.py start --competition <recent finished slug> --arm memory-off`
→ run the same loops for a day or two → `score --fetch` → `end` → repeat with
`--arm memory-on` → `make gym-report`. Each pair is one data point on whether
the second brain is earning its keep.

## Monitoring quick reference

| Question | Command |
|----------|---------|
| Who's winning, submissions left today | `make status` |
| Anyone fooling themselves | `make lint` |
| What is agent X working on | `python tools/scheduler.py status --agent <x>` |
| Is memory healthy | `make memory-stats`, `make amend` |
| Is the harness intact | `make selfcheck` |
| Is memory helping at all | `make gym-report` |

## Appendix — tmux in 5 minutes (first-timers)

tmux keeps terminals alive on the machine after you disconnect — without it,
closing your laptop or dropping SSH kills the agent loops.

Install: `brew install tmux` (Mac) or `sudo apt install -y tmux` (Ubuntu droplet).

Every tmux command = press `Ctrl+b`, release, then one more key.

| You want to | Do this |
|---|---|
| Start a session named kaggle | `tmux new -s kaggle` |
| New window inside it | `Ctrl+b` then `c` |
| Switch windows | `Ctrl+b` then `0` / `1` / `2` |
| Rename current window | `Ctrl+b` then `,` |
| Detach (everything keeps running) | `Ctrl+b` then `d` |
| Come back | `tmux attach -t kaggle` |
| Scroll output | `Ctrl+b` then `[` (q to stop) |
| List sessions | `tmux ls` |

First-time walkthrough: SSH in → `tmux new -s kaggle` → rename window 0 to
"claude", paste its loop → `Ctrl+b c`, rename "codex", paste its loop →
`Ctrl+b c`, rename "gemini", paste its loop → `Ctrl+b d` → log out. Next day:
`tmux attach -t kaggle` and flip through the windows.
