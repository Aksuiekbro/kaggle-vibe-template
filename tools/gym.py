#!/usr/bin/env python3
"""Shadow gym: replay finished Kaggle competitions as training runs.

Finished competitions with late submission enabled still score against the real
private leaderboard — a ground-truth training loop for the whole system without
burning live-competition time. Runs are tagged with an arm (memory-on /
memory-off) so the memory A/B accumulates automatically.

Contamination note: the base model may recall winners of famous competitions,
which inflates absolute scores in BOTH arms; arm DELTAS remain meaningful.
Prefer recent competitions for cleaner reads.

Usage:
  python tools/gym.py start  --competition <slug> --arm memory-on|memory-off
  python tools/gym.py score  [--fetch] [--private-score X] [--rank R --teams N]
  python tools/gym.py end
  python tools/gym.py status
  python tools/gym.py report
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import AGENTS, PROJECT_ROOT, now_iso, read_jsonl, append_jsonl

GYM_DIR = PROJECT_ROOT / ".ai" / "gym"
CURRENT = GYM_DIR / "CURRENT.json"
RUNS = GYM_DIR / "RUNS.jsonl"
SNAPSHOT_FILES = ["COMPETITION.md", "shared/submissions/registry.json"]


def load_current():
    if not CURRENT.exists():
        return None
    with open(CURRENT) as f:
        return json.load(f)


def save_current(state):
    GYM_DIR.mkdir(parents=True, exist_ok=True)
    with open(CURRENT, "w") as f:
        json.dump(state, f, indent=2)


def cmd_start(args):
    if load_current():
        print("ERROR: a gym run is already active. Finish it with `gym.py end` first.")
        return 1

    run_id = f"gym_{now_iso()[:19].replace(':', '').replace('-', '')}"
    snap_dir = GYM_DIR / run_id / "snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for rel in SNAPSHOT_FILES:
        src = PROJECT_ROOT / rel
        if src.exists():
            dest = snap_dir / rel.replace("/", "__")
            shutil.copy2(src, dest)

    # Set aside live agent workspaces so the gym run starts clean and cannot
    # pollute (or be polluted by) live-competition state.
    ws_snap = GYM_DIR / run_id / "workspaces"
    stashed = []
    for agent in AGENTS:
        ws = PROJECT_ROOT / "agents" / agent / "workspace"
        if ws.exists() and any(ws.iterdir()):
            dest = ws_snap / agent
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ws), str(dest))
            ws.mkdir(parents=True)
            stashed.append(agent)
    if stashed:
        print(f"Live workspaces set aside for: {', '.join(stashed)} "
              "(restored automatically at gym.py end)")

    print(f"Setting up gym competition '{args.competition}' (this reuses tools/setup.py)...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "setup.py"),
         "--competition", args.competition],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("WARNING: setup.py returned nonzero — check data download, then continue.")

    state = {
        "run_id": run_id,
        "competition": args.competition,
        "arm": args.arm,
        "started": now_iso(),
        "private_score": None,
        "rank": None,
        "teams": None,
        "notes": args.note or "",
    }
    save_current(state)

    print(f"\nGYM RUN ACTIVE: {run_id}")
    print(f"  competition: {args.competition}   arm: {args.arm}")
    if args.arm == "memory-off":
        print("\n  ABLATION ARM — launch every agent with the environment variable:")
        print("    KAGGLE_MEMORY_OFF=1")
        print("  (memory_cli retrieve and skills list will refuse; everything else is identical)")
    print("\nRun the normal loop: predictions -> research -> experiments -> submit.")
    print("Late submissions still receive real private-LB scores.")
    print("When scored: gym.py score --fetch   (or enter manually), then gym.py end")
    return 0


def cmd_score(args):
    state = load_current()
    if not state:
        print("ERROR: no active gym run.")
        return 1

    if args.fetch:
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", state["competition"], "--csv"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                header = [h.strip().lower() for h in lines[0].split(",")]
                if len(lines) > 1 and "privatescore" in header:
                    idx = header.index("privatescore")
                    latest = lines[1].split(",")
                    score = latest[idx].strip()
                    if score and score.lower() != "none":
                        state["private_score"] = float(score)
                        print(f"Fetched private score: {score}")
            if state["private_score"] is None:
                print("Could not fetch a private score from kaggle CLI; enter it manually "
                      "with --private-score.")
        except Exception as e:
            print(f"Fetch failed ({e}); enter --private-score manually.")

    if args.private_score is not None:
        state["private_score"] = args.private_score
    if args.rank is not None:
        state["rank"] = args.rank
    if args.teams is not None:
        state["teams"] = args.teams
    save_current(state)

    print(f"Recorded: private_score={state['private_score']} "
          f"rank={state['rank']} teams={state['teams']}")
    if state["rank"] and state["teams"]:
        pct = 100.0 * (1 - state["rank"] / state["teams"])
        print(f"Percentile: {pct:.1f}")
    return 0


def cmd_end(args):
    state = load_current()
    if not state:
        print("ERROR: no active gym run.")
        return 1

    if state["private_score"] is None and state["rank"] is None:
        print("WARNING: ending a gym run with no recorded score — it will count as incomplete.")

    percentile = None
    if state.get("rank") and state.get("teams"):
        percentile = round(100.0 * (1 - state["rank"] / state["teams"]), 2)

    record = dict(state)
    record["ended"] = now_iso()
    record["percentile"] = percentile
    append_jsonl(RUNS, record)

    snap_dir = GYM_DIR / state["run_id"] / "snapshot"
    if snap_dir.exists():
        for dest_rel in SNAPSHOT_FILES:
            src = snap_dir / dest_rel.replace("/", "__")
            if src.exists():
                shutil.copy2(src, PROJECT_ROOT / dest_rel)
        print("Live-competition state restored from snapshot.")

    # Archive the gym workspaces (postmortem evidence), restore the live ones.
    ws_snap = GYM_DIR / state["run_id"] / "workspaces"
    if ws_snap.exists():
        archive = GYM_DIR / state["run_id"] / "gym-workspaces"
        for agent_dir in sorted(ws_snap.iterdir()):
            ws = PROJECT_ROOT / "agents" / agent_dir.name / "workspace"
            if ws.exists():
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ws), str(archive / agent_dir.name))
            shutil.move(str(agent_dir), str(ws))
        print(f"Live workspaces restored; gym workspaces archived under "
              f"{archive.relative_to(PROJECT_ROOT)} for the postmortem.")
    CURRENT.unlink()

    print(f"\nGYM RUN COMPLETE: {state['run_id']} ({state['arm']}) — "
          f"score={state['private_score']} percentile={percentile}")
    print("\nClose the learning loop now (this is the point of the gym):")
    print("  1. .ai/checklists/postmortem.md against the real winner writeups")
    print("  2. memory_cli.py retrieve --anchor postmortem  -> writebacks")
    print("  3. skills.py log-use for every skill used")
    print("  4. calibration.py report --write")
    print("  5. gym.py report — check the arm comparison")
    return 0


def cmd_status(args):
    state = load_current()
    if state:
        print(json.dumps(state, indent=2))
    else:
        print("No active gym run.")
        runs = read_jsonl(RUNS)
        if runs:
            print(f"{len(runs)} completed run(s). Use `gym.py report`.")
    return 0


def cmd_report(args):
    runs = read_jsonl(RUNS)
    if not runs:
        print("No completed gym runs yet.")
        return 0
    print(f"{'run_id':<22} {'competition':<28} {'arm':<11} {'score':<10} pct")
    print("-" * 82)
    for r in runs:
        print(f"{r.get('run_id', '?'):<22} {r.get('competition', '?'):<28} "
              f"{r.get('arm', '?'):<11} {str(r.get('private_score')):<10} "
              f"{r.get('percentile')}")

    by_arm = {}
    for r in runs:
        if r.get("percentile") is not None:
            by_arm.setdefault(r.get("arm"), []).append(r["percentile"])
    if len(by_arm) >= 2:
        print("\nArm comparison (mean percentile):")
        means = {}
        for arm, pcts in sorted(by_arm.items()):
            means[arm] = sum(pcts) / len(pcts)
            print(f"  {arm}: {means[arm]:.1f} (n={len(pcts)})")
        if "memory-on" in means and "memory-off" in means:
            delta = means["memory-on"] - means["memory-off"]
            n = min(len(by_arm["memory-on"]), len(by_arm["memory-off"]))
            print(f"\n  memory effect: {delta:+.1f} percentile points "
                  f"(n={n} per arm{'; small n — collect more runs' if n < 5 else ''})")
    else:
        print("\nRun both arms (memory-on and memory-off) to get the ablation comparison.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Shadow gym for finished competitions")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start")
    p.add_argument("--competition", required=True)
    p.add_argument("--arm", required=True, choices=["memory-on", "memory-off"])
    p.add_argument("--note")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("score")
    p.add_argument("--fetch", action="store_true", help="try to fetch via kaggle CLI")
    p.add_argument("--private-score", type=float)
    p.add_argument("--rank", type=int)
    p.add_argument("--teams", type=int)
    p.set_defaults(func=cmd_score)

    sub.add_parser("end").set_defaults(func=cmd_end)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("report").set_defaults(func=cmd_report)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
