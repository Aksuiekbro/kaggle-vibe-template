#!/usr/bin/env python3
"""Submission registry manager. Central source of truth for all submissions."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent.parent / "shared" / "submissions" / "registry.json"
AGENTS = ["claude", "codex", "gemini"]


def load_registry():
    if not REGISTRY_PATH.exists():
        return create_empty_registry()
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def create_empty_registry(competition="unknown"):
    registry = {
        "competition": competition,
        "submissions": [],
        "best_scores": {agent: None for agent in AGENTS},
        "daily_counts": {},
    }
    save_registry(registry)
    return registry


def next_id(registry):
    if not registry["submissions"]:
        return "sub_001"
    last_num = max(int(s["id"].split("_")[1]) for s in registry["submissions"])
    return f"sub_{last_num + 1:03d}"


def add_submission(registry, agent, file_path, local_score, cv_std, flags,
                   description, approach, method="unknown", kaggle_score=None,
                   provenance_type="original", based_on=None, status="submitted",
                   direction="maximize"):
    sub_id = next_id(registry)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    submission = {
        "id": sub_id,
        "agent": agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": str(file_path),
        "local_score": local_score,
        "kaggle_score": kaggle_score,
        "cv_std": cv_std,
        "flags": flags,
        "method": method,
        "description": description,
        "provenance": {
            "type": provenance_type,
            "based_on": based_on,
            "approach": approach,
        },
        "status": status,
    }

    registry["submissions"].append(submission)

    if today not in registry["daily_counts"]:
        registry["daily_counts"][today] = {agent: 0 for agent in AGENTS}
        registry["daily_counts"][today]["total"] = 0
    registry["daily_counts"][today][agent] = registry["daily_counts"][today].get(agent, 0) + 1
    registry["daily_counts"][today]["total"] = registry["daily_counts"][today].get("total", 0) + 1

    # The score gate compares against best_scores, so it must track the local
    # score at submission time — kaggle_score arrives later, if ever.
    if status == "submitted":
        score = kaggle_score if kaggle_score is not None else local_score
        if score is not None:
            current_best = registry["best_scores"].get(agent)
            prev = None
            if current_best:
                prev = current_best.get("kaggle_score")
                if prev is None:
                    prev = current_best.get("local_score")
            better = prev is None or (score > prev if direction == "maximize" else score < prev)
            if better:
                registry["best_scores"][agent] = {
                    "id": sub_id,
                    "kaggle_score": kaggle_score,
                    "local_score": local_score,
                }

    save_registry(registry)
    return sub_id


def get_agent_best(registry, agent):
    return registry["best_scores"].get(agent)


def get_recent_approaches(registry, agent, n=3):
    agent_subs = [s for s in registry["submissions"] if s["agent"] == agent and s["status"] == "submitted"]
    return [s["provenance"]["approach"] for s in agent_subs[-n:]]


def check_diversity(registry, agent, new_approach):
    recent = get_recent_approaches(registry, agent, n=3)
    if len(recent) >= 3 and all(a == new_approach for a in recent):
        return False, f"Last 3 submissions all used '{new_approach}'. Try a different approach."
    return True, ""


def print_status(registry):
    print(f"\n{'='*60}")
    print(f"Competition: {registry['competition']}")
    print(f"Total submissions: {len(registry['submissions'])}")
    print(f"{'='*60}\n")

    print("Best Scores:")
    print(f"{'Agent':<12} {'Score':<12} {'Submission':<12}")
    print("-" * 36)
    for agent in AGENTS:
        best = registry["best_scores"].get(agent)
        if best:
            score = best.get("kaggle_score", best.get("local_score", "N/A"))
            print(f"{agent:<12} {str(score):<12} {best['id']:<12}")
        else:
            print(f"{agent:<12} {'N/A':<12} {'N/A':<12}")

    print(f"\nSubmissions by agent:")
    for agent in AGENTS:
        count = len([s for s in registry["submissions"] if s["agent"] == agent])
        submitted = len([s for s in registry["submissions"] if s["agent"] == agent and s["status"] == "submitted"])
        rejected = len([s for s in registry["submissions"] if s["agent"] == agent and s["status"] == "rejected"])
        print(f"  {agent}: {count} total ({submitted} submitted, {rejected} rejected)")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_counts = registry["daily_counts"].get(today, {})
    if today_counts:
        print(f"\nToday's submissions: {today_counts.get('total', 0)}")
        for agent in AGENTS:
            print(f"  {agent}: {today_counts.get(agent, 0)}")

    flagged = [s for s in registry["submissions"] if s.get("flags")]
    if flagged:
        print(f"\nFlagged submissions:")
        for s in flagged[-5:]:
            print(f"  {s['id']} ({s['agent']}): {', '.join(s['flags'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/registry.py [status|init <competition>]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "status":
        registry = load_registry()
        print_status(registry)
    elif command == "init":
        competition = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        create_empty_registry(competition)
        print(f"Registry initialized for competition: {competition}")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
