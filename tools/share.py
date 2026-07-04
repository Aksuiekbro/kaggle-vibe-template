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
CONSOLIDATION_FILES = [
    "PLAN_DRAFT.md",
    "MEMORY_CANDIDATES.md",
    "PREDICTION.md",
    "POSTMORTEM.md",
]


def load_registry():
    if not REGISTRY_PATH.exists():
        print("ERROR: Registry not found. Run setup.py first.")
        sys.exit(1)
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def collect_consolidation_drafts():
    """Copy agent planning/memory drafts into a dated run folder for human consolidation."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT / ".ai" / "runs" / stamp / "consolidation"
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for agent in AGENTS:
        workspace = PROJECT_ROOT / "agents" / agent / "workspace"
        agent_dest = root / agent
        copied = []
        for name in CONSOLIDATION_FILES:
            src = workspace / name
            if src.exists():
                agent_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, agent_dest / name)
                copied.append(name)
        if copied:
            manifest["files"][agent] = copied

    with open(root / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    summary = root / "CONSOLIDATION.md"
    lines = [
        "# Sharing Round Consolidation",
        "",
        f"Created: {manifest['created']}",
        "",
        "Review these drafts before the next autonomous run:",
        "",
        "- PLAN_DRAFT.md: merge useful rows into PLAN.md",
        "- MEMORY_CANDIDATES.md: promote only reviewed evidence-backed cards into .ai/memory/",
        "- PREDICTION.md: copy open predictions into .ai/memory/predictions/INDEX.md",
        "- POSTMORTEM.md: convert durable findings into memory-card candidates",
        "",
        "| Agent | Files copied |",
        "|-------|--------------|",
    ]
    for agent in AGENTS:
        copied = manifest["files"].get(agent, [])
        lines.append(f"| {agent} | {', '.join(copied) if copied else '-'} |")
    summary.write_text("\n".join(lines) + "\n")

    return root, manifest


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
    if len(shared_agents) >= 2:
        print("\nEnsemble opportunity — check decorrelation across agent bests:")
        print("  python tools/stack.py correlation --files shared/best/*/[!I]*.csv")
        print("  python tools/stack.py blend / select-finals  (weights from CV only)")
    print("When this round ends, plan/memory/prediction drafts will be collected for consolidation.")
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

    consolidation_dir, manifest = collect_consolidation_drafts()

    registry["sharing_round_active"] = False
    registry["sharing_round_ended"] = datetime.now(timezone.utc).isoformat()
    registry["last_consolidation_dir"] = str(consolidation_dir.relative_to(PROJECT_ROOT))
    save_registry(registry)

    if SHARED_BEST.exists():
        shutil.rmtree(SHARED_BEST)
        SHARED_BEST.mkdir()

    print(f"\nSharing round ENDED. shared/best/ is now empty.")
    print(f"Agents are back in isolation mode.\n")
    print(f"Consolidation drafts collected: {consolidation_dir}")
    for agent, copied in manifest["files"].items():
        print(f"  {agent}: {', '.join(copied)}")
    if not manifest["files"]:
        print("  No draft files found.")
    print()


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
