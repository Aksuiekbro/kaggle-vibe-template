#!/usr/bin/env python3
"""Fake-practice linter.

Um_nik's self-deception taxonomy converted into executable checks. Each check
maps to a constitution rule; violations are logged to the rent ledger and
reported so they can be injected back into the agent's context at the decision
point instead of sitting in a philosophy document.

Usage:
  python tools/practice_lint.py [--agent claude] [--json]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import (
    AGENTS,
    MEMORY_ROOT,
    PROJECT_ROOT,
    extract_section,
    ledger_event,
    parse_table_rows,
    prediction_gate_status,
    read_jsonl,
    reading_log_path,
    workspace,
)

MAX_ACTIVE_EXPERIMENTS = 3
MAX_UNPROCESSED_READS = 5
MAX_PENDING_QUEUE = 10
RE_READ_LIMIT = 3


def _violation(rule, code, agent, detail):
    return {"rule": rule, "code": code, "agent": agent, "detail": detail}


def check_prediction_before_reading(agent):
    """C2: agent read gated material while the prediction gate was failing."""
    violations = []
    reads = read_jsonl(reading_log_path(agent))
    ok, _ = prediction_gate_status(agent)
    overrides = [r for r in reads if r.get("gate") == "override"]
    if overrides:
        violations.append(_violation(
            "C2", "GATE_OVERRIDDEN", agent,
            f"{len(overrides)} forced read(s) while the prediction gate failed",
        ))
    if reads and not ok:
        violations.append(_violation(
            "C2", "PREDICTION_INCOMPLETE_AFTER_READING", agent,
            f"{len(reads)} logged read(s) but PREDICTION.md is still incomplete",
        ))
    return violations


def check_reading_without_queue_change(agent):
    """C3: learning must change the experiment queue."""
    violations = []
    ws = workspace(agent)
    reads = read_jsonl(reading_log_path(agent))
    if not reads:
        return violations

    queue_files = ["STRATEGY.md", "PROGRESS.md", "PLAN_DRAFT.md", "RESEARCH.md"]
    mtimes = [
        (ws / name).stat().st_mtime
        for name in queue_files
        if (ws / name).exists()
    ]
    last_queue_change = max(mtimes) if mtimes else 0.0
    last_change_iso = datetime.fromtimestamp(last_queue_change, tz=timezone.utc).isoformat()

    unprocessed = [r for r in reads if r.get("ts", "") > last_change_iso]
    if len(unprocessed) >= MAX_UNPROCESSED_READS:
        violations.append(_violation(
            "C3", "READING_WITHOUT_QUEUE_CHANGE", agent,
            f"{len(unprocessed)} reads since the strategy/queue files last changed — "
            "convert them into experiment rows or explicit rejections",
        ))

    url_counts = {}
    for r in reads:
        url = r.get("url", "")
        url_counts[url] = url_counts.get(url, 0) + 1
    for url, count in url_counts.items():
        if count >= RE_READ_LIMIT:
            violations.append(_violation(
                "C3", "REPEATED_READING", agent,
                f"read {count}x without resolution: {url}",
            ))
    return violations


def check_wip_and_queue_flood(agent):
    """C5: at most 3 active experiments (scheduler queue + plan draft); no flooding."""
    violations = []

    exp_queue = workspace(agent) / "EXPERIMENTS.jsonl"
    in_flight = [r for r in read_jsonl(exp_queue) if r.get("stage") in ("probe", "full")]
    if len(in_flight) > MAX_ACTIVE_EXPERIMENTS:
        violations.append(_violation(
            "C5", "WIP_LIMIT_EXCEEDED", agent,
            f"{len(in_flight)} experiments in flight in the scheduler queue "
            f"(limit {MAX_ACTIVE_EXPERIMENTS}) — record their results before starting more",
        ))

    plan_draft = workspace(agent) / "PLAN_DRAFT.md"
    if not plan_draft.exists():
        return violations
    rows = parse_table_rows(plan_draft.read_text())

    def status_of(row):
        return row[5].lower() if len(row) > 5 else ""

    active = [r for r in rows if "active" in status_of(r)]
    pending = [r for r in rows if "pending" in status_of(r)]
    if len(active) > MAX_ACTIVE_EXPERIMENTS:
        violations.append(_violation(
            "C5", "WIP_LIMIT_EXCEEDED", agent,
            f"{len(active)} active experiments (limit {MAX_ACTIVE_EXPERIMENTS}) — finish or park some",
        ))
    if len(pending) > MAX_PENDING_QUEUE:
        violations.append(_violation(
            "C5", "QUEUE_FLOODING", agent,
            f"{len(pending)} pending rows — rank by expected impact and prune; "
            "a flooded queue is compliance theater",
        ))
    return violations


def check_submission_diversity(agent):
    """C7: no 3 consecutive submissions with the same approach."""
    violations = []
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    if not registry_path.exists():
        return violations
    with open(registry_path) as f:
        registry = json.load(f)
    subs = [
        s for s in registry.get("submissions", [])
        if s.get("agent") == agent and s.get("status") == "submitted"
    ]
    recent = [s["provenance"]["approach"] for s in subs[-3:]]
    if len(recent) == 3 and len(set(recent)) == 1:
        violations.append(_violation(
            "C7", "APPROACH_MONOCULTURE", agent,
            f"last 3 submissions all used '{recent[0]}' — change strategy, not hyperparameters",
        ))
    return violations


def check_self_promoted_memory(agent):
    """C8: no self-promotion — validated status requires a reviewer."""
    violations = []

    candidates = workspace(agent) / "MEMORY_CANDIDATES.md"
    if candidates.exists():
        for row in parse_table_rows(candidates.read_text()):
            if len(row) >= 6 and "validated" in row[5].lower():
                violations.append(_violation(
                    "C8", "SELF_PROMOTED_CANDIDATE", agent,
                    f"workspace memory candidate marked validated: '{row[0][:60]}' — "
                    "only cross-review promotes cards",
                ))
    return violations


def check_shared_memory_review():
    """C8 (shared): cards in .ai/memory/ marked validated must carry a reviewer."""
    violations = []
    for folder in ("patterns", "failures", "competitions"):
        d = MEMORY_ROOT / folder
        if not d.exists():
            continue
        for card in sorted(d.glob("*.md")):
            text = card.read_text()
            if "status: validated" not in text:
                continue
            review = extract_section_yaml_field(text, "reviewer")
            if not review:
                violations.append(_violation(
                    "C8", "VALIDATED_WITHOUT_REVIEWER", "shared",
                    f"{card.relative_to(PROJECT_ROOT)} is validated but has no reviewer",
                ))
    return violations


def extract_section_yaml_field(text, field):
    """Cheap scan for a filled `field:` line inside a card's yaml block."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return value
    return ""


def lint(agents, as_json=False):
    all_violations = []
    for agent in agents:
        if not workspace(agent).exists():
            continue
        all_violations += check_prediction_before_reading(agent)
        all_violations += check_reading_without_queue_change(agent)
        all_violations += check_wip_and_queue_flood(agent)
        all_violations += check_submission_diversity(agent)
        all_violations += check_self_promoted_memory(agent)
    all_violations += check_shared_memory_review()

    for v in all_violations:
        ledger_event(v["rule"], "violated", v["agent"], f"{v['code']}: {v['detail']}")

    if as_json:
        print(json.dumps({"violations": all_violations}, indent=2))
    else:
        if not all_violations:
            print("Practice lint: clean. No self-deception signatures detected.")
        else:
            print(f"Practice lint: {len(all_violations)} violation(s)\n")
            for v in all_violations:
                print(f"  [{v['rule']}/{v['code']}] {v['agent']}: {v['detail']}")
            print("\nSee .ai/constitution.md for the rules behind these checks.")

    return 1 if all_violations else 0


def main():
    parser = argparse.ArgumentParser(description="Fake-practice linter")
    parser.add_argument("--agent", choices=AGENTS, help="Lint one agent (default: all)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    agents = [args.agent] if args.agent else AGENTS
    sys.exit(lint(agents, args.json))


if __name__ == "__main__":
    main()
