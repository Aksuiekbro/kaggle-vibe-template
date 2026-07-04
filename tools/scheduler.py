#!/usr/bin/env python3
"""Successive-halving experiment scheduler.

Experiments earn full-fidelity compute by passing a cheap probe first
(constitution C13). Selection is value-per-cost with a novelty bonus; the queue
is a JSONL file in the agent's workspace so it survives sessions.

Stages: queued -> probe -> full -> done | killed

Usage:
  python tools/scheduler.py add    --agent a --idea "..." --predicted-delta 0.003 --cost-hours 2 [--source card:<id>]
  python tools/scheduler.py next   --agent a          # what to run now (respects WIP limit)
  python tools/scheduler.py record --agent a --id exp_001 --stage probe|full --delta 0.002 [--note "..."]
  python tools/scheduler.py status --agent a
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import AGENTS, ledger_event, now_iso, workspace

QUEUE_NAME = "EXPERIMENTS.jsonl"
MAX_WIP = 3          # C5: active probes + fulls
NOVELTY_BONUS = 1.2  # untried sources get a 20% priority boost
PROBE_GUIDE = "probe = ~10% data or 1-2 folds or a capped time box; answer only 'is there signal?'"


def queue_path(agent):
    return workspace(agent) / QUEUE_NAME


def load_queue(agent):
    path = queue_path(agent)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_queue(agent, rows):
    path = queue_path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def next_id(rows):
    nums = [int(r["id"].split("_")[1]) for r in rows if r.get("id", "").startswith("exp_")]
    return f"exp_{(max(nums) + 1) if nums else 1:03d}"


def priority(row, rows):
    """Value per cost, novelty-boosted. Simple on purpose — the probe stage is
    the real intelligence; ranking only decides what gets probed first."""
    delta = float(row.get("predicted_delta", 0) or 0)
    cost = max(0.1, float(row.get("cost_hours", 1) or 1))
    source_kind = str(row.get("source", "own")).split(":")[0]
    tried_kinds = {str(r.get("source", "own")).split(":")[0]
                   for r in rows if r.get("stage") in ("done", "killed", "full")}
    novelty = NOVELTY_BONUS if source_kind not in tried_kinds else 1.0
    return (delta / cost) * novelty


def wip(rows):
    return [r for r in rows if r.get("stage") in ("probe", "full")]


def cmd_add(args):
    rows = load_queue(args.agent)
    row = {
        "id": next_id(rows),
        "idea": args.idea,
        "source": args.source or "own",
        "predicted_delta": args.predicted_delta,
        "cost_hours": args.cost_hours,
        "stage": "queued",
        "probe_delta": None,
        "full_delta": None,
        "created": now_iso(),
        "updated": now_iso(),
        "notes": args.note or "",
    }
    rows.append(row)
    save_queue(args.agent, rows)
    print(f"Added {row['id']}: {args.idea}")
    print(f"  priority score: {priority(row, rows):.4f} (delta/cost, novelty-adjusted)")
    return 0


def cmd_next(args):
    rows = load_queue(args.agent)
    active = wip(rows)
    if len(active) >= MAX_WIP:
        ledger_event("C5", "prevented", args.agent, f"scheduler refused: {len(active)} experiments in flight")
        print(f"BLOCKED (C5): {len(active)} experiments already in flight (limit {MAX_WIP}).")
        for r in active:
            print(f"  {r['id']} [{r['stage']}]: {r['idea']}")
        print("Record their results first: scheduler.py record --id <id> --stage <stage> --delta <x>")
        return 1

    # promoted-but-unrun fulls take precedence over new probes
    promoted = [r for r in rows if r.get("stage") == "probe" and r.get("probe_delta") is not None]
    queued = sorted([r for r in rows if r.get("stage") == "queued"],
                    key=lambda r: -priority(r, rows))
    if promoted:
        r = promoted[0]
        print(f"RUN FULL: {r['id']} — {r['idea']}")
        print(f"  probe delta was {r['probe_delta']}; run full CV now, then "
              f"record --id {r['id']} --stage full --delta <x>")
        return 0
    if not queued:
        print("Queue empty. Add experiments (research, memory retrieve --anchor stuck, PLAN.md).")
        return 0

    r = queued[0]
    r["stage"] = "probe"
    r["updated"] = now_iso()
    save_queue(args.agent, rows)
    ledger_event("C13", "fired", args.agent, f"probe started {r['id']}")
    print(f"RUN PROBE: {r['id']} — {r['idea']}")
    print(f"  predicted delta {r['predicted_delta']}, est cost {r['cost_hours']}h, source {r['source']}")
    print(f"  {PROBE_GUIDE}")
    print(f"  then: scheduler.py record --agent {args.agent} --id {r['id']} --stage probe --delta <x>")
    if len(queued) > 1:
        print(f"  (ranked ahead of {len(queued)-1} queued experiment(s))")
    return 0


def cmd_record(args):
    rows = load_queue(args.agent)
    row = next((r for r in rows if r["id"] == args.id), None)
    if not row:
        print(f"ERROR: no experiment {args.id}")
        return 1
    row["updated"] = now_iso()
    if args.note:
        row["notes"] = (row.get("notes", "") + " | " + args.note).strip(" |")

    if args.stage == "probe":
        row["probe_delta"] = args.delta
        if args.delta > 0:
            print(f"{args.id}: probe delta {args.delta} > 0 — PROMOTED to full fidelity.")
            print(f"  next `scheduler.py next` will tell you to run it full.")
            ledger_event("C13", "fired", args.agent, f"{args.id} promoted (probe {args.delta})")
        else:
            row["stage"] = "killed"
            print(f"{args.id}: probe delta {args.delta} <= 0 — KILLED at probe cost. "
                  "That is the scheduler working, not a failure.")
            ledger_event("C13", "fired", args.agent, f"{args.id} killed at probe ({args.delta})")
    else:
        row["full_delta"] = args.delta
        row["stage"] = "done"
        print(f"{args.id}: full delta {args.delta} recorded — stage done.")
        if args.delta > 0:
            print("  If this beats the current best CV: submit through the score gate.")
        probe = row.get("probe_delta")
        if probe is not None and probe > 0 and args.delta <= 0:
            print("  NOTE: probe was positive but full run was not — record the "
                  "probe-fidelity mismatch in PROGRESS.md; if it recurs, raise probe fidelity.")
        src = str(row.get("source", ""))
        if src.startswith("card:"):
            print(f"  Write back to memory: memory_cli.py writeback --card {src[5:]} "
                  f"--competition <slug> --result {'hit' if args.delta > 0 else 'miss'} "
                  f"--predicted-delta {row['predicted_delta']} --actual-delta {args.delta}")
        if src.startswith("skill:"):
            print(f"  Log skill outcome: skills.py log-use --skill {src[6:]} "
                  f"--competition <slug> --outcome {'win' if args.delta > 0 else 'loss'}")
    save_queue(args.agent, rows)
    return 0


def cmd_status(args):
    rows = load_queue(args.agent)
    if not rows:
        print("Queue empty.")
        return 0
    order = {"probe": 0, "full": 1, "queued": 2, "done": 3, "killed": 4}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r.get("stage"), 9), -priority(r, rows)))
    print(f"{'id':<9} {'stage':<7} {'pred':<8} {'probe':<8} {'full':<8} {'prio':<7} idea")
    print("-" * 100)
    for r in rows_sorted:
        print(f"{r['id']:<9} {r.get('stage', '?'):<7} {str(r.get('predicted_delta')):<8} "
              f"{str(r.get('probe_delta')):<8} {str(r.get('full_delta')):<8} "
              f"{priority(r, rows):<7.3f} {r['idea'][:50]}")
    probes = [r for r in rows if r.get("probe_delta") is not None]
    kills = [r for r in rows if r.get("stage") == "killed"]
    if probes:
        print(f"\n{len(kills)}/{len(probes)} probes killed cheaply — "
              f"full-fidelity compute went to the survivors.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Successive-halving experiment scheduler")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add")
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.add_argument("--idea", required=True)
    p.add_argument("--predicted-delta", type=float, required=True,
                   help="expected CV improvement (metric units)")
    p.add_argument("--cost-hours", type=float, required=True)
    p.add_argument("--source", help="card:<id> | skill:<id> | url:<...> | own")
    p.add_argument("--note")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("next")
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("record")
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.add_argument("--id", required=True)
    p.add_argument("--stage", required=True, choices=["probe", "full"])
    p.add_argument("--delta", type=float, required=True,
                   help="measured CV delta vs current best (negative = worse)")
    p.add_argument("--note")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("status")
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
