#!/usr/bin/env python3
"""Sharing round manager. Handles copying best submissions and code for cross-agent collaboration."""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SHARED_BEST = PROJECT_ROOT / "shared" / "best"
REGISTRY_PATH = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
AGENTS = ["claude", "codex", "gemini"]


def load_registry():
    if not REGISTRY_PATH.exists():
        print("ERROR: Registry not found. Run setup.py first.")
        sys.exit(1)
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def start_sharing_round():
    """Copy each agent's best submission + workspace to shared/best/."""
    registry = load_registry()

    if registry.get("sharing_round_active"):
        print("ERROR: A sharing round is already active. Run 'end' first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Starting Sharing Round")
    print(f"{'='*60}\n")

    if SHARED_BEST.exists():
        shutil.rmtree(SHARED_BEST)
    SHARED_BEST.mkdir(parents=True)

    shared_agents = []

    for agent in AGENTS:
        best = registry["best_scores"].get(agent)
        if not best:
            print(f"  {agent}: No submissions yet — skipping")
            continue

        agent_dir = SHARED_BEST / agent
        agent_dir.mkdir(parents=True)

        best_sub = None
        for sub in registry["submissions"]:
            if sub["id"] == best["id"]:
                best_sub = sub
                break

        if best_sub:
            src_file = PROJECT_ROOT / best_sub["file"]
            if src_file.exists():
                shutil.copy2(src_file, agent_dir / src_file.name)

            info = {
                "submission_id": best_sub["id"],
                "score": best_sub.get("kaggle_score") or best_sub.get("local_score"),
                "approach": best_sub["provenance"]["approach"],
                "description": best_sub["description"],
            }
            with open(agent_dir / "INFO.json", "w") as f:
                json.dump(info, f, indent=2)

        workspace = PROJECT_ROOT / "agents" / agent / "workspace"
        if workspace.exists():
            dest_workspace = agent_dir / "workspace"
            dest_workspace.mkdir(parents=True)
            for item in workspace.iterdir():
                if item.is_file() and item.suffix in (".py", ".cpp", ".c", ".h", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".sh"):
                    shutil.copy2(item, dest_workspace / item.name)
                elif item.is_dir() and item.name not in ("__pycache__", ".git", "venv", ".venv"):
                    if sum(1 for _ in item.rglob("*") if _.is_file()) < 100:
                        shutil.copytree(item, dest_workspace / item.name, dirs_exist_ok=True,
                                       ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.o", "*.so"))

        print(f"  {agent}: Shared submission {best['id']} (score: {best.get('kaggle_score') or best.get('local_score', 'N/A')})")
        shared_agents.append(agent)

    registry["sharing_round_active"] = True
    registry["sharing_round_started"] = datetime.now(timezone.utc).isoformat()
    registry["sharing_round_agents"] = shared_agents
    save_registry(registry)

    print(f"\nSharing round ACTIVE. Agents can now read shared/best/")
    print(f"Shared agents: {', '.join(shared_agents)}")
    print(f"\nRun 'python tools/share.py end' to close the sharing round.\n")


def show_status():
    """Show current sharing round state."""
    registry = load_registry()

    if not registry.get("sharing_round_active"):
        print("\nNo active sharing round.")
        return

    print(f"\n{'='*60}")
    print("Sharing Round Status")
    print(f"{'='*60}")
    print(f"  Started: {registry.get('sharing_round_started', 'unknown')}")
    print(f"  Agents: {', '.join(registry.get('sharing_round_agents', []))}")

    start_time = registry.get("sharing_round_started", "")
    derivatives = [
        s for s in registry["submissions"]
        if s["provenance"]["type"] == "derivative" and s["timestamp"] > start_time
    ]
    if derivatives:
        print(f"\n  Derivatives produced during this round:")
        for d in derivatives:
            print(f"    {d['id']} by {d['agent']}: based on {d['provenance']['based_on']} — {d['description']}")
    else:
        print(f"\n  No derivatives produced yet.")
    print()


def end_sharing_round():
    """Close sharing round and lock shared/best/."""
    registry = load_registry()

    if not registry.get("sharing_round_active"):
        print("ERROR: No active sharing round to end.")
        sys.exit(1)

    start_time = registry.get("sharing_round_started", "")
    derivatives = [
        s for s in registry["submissions"]
        if s["provenance"]["type"] == "derivative" and s["timestamp"] > start_time
    ]

    print(f"\n{'='*60}")
    print("Ending Sharing Round")
    print(f"{'='*60}")
    print(f"  Derivatives produced: {len(derivatives)}")
    for d in derivatives:
        print(f"    {d['id']} by {d['agent']}: {d['description']}")

    registry["sharing_round_active"] = False
    registry["sharing_round_ended"] = datetime.now(timezone.utc).isoformat()
    save_registry(registry)

    if SHARED_BEST.exists():
        shutil.rmtree(SHARED_BEST)
        SHARED_BEST.mkdir()

    print(f"\nSharing round ENDED. shared/best/ is now empty.")
    print(f"Agents are back in isolation mode.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/share.py [start|status|end]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "start":
        start_sharing_round()
    elif command == "status":
        show_status()
    elif command == "end":
        end_sharing_round()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python tools/share.py [start|status|end]")
        sys.exit(1)
