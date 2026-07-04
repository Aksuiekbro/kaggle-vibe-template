#!/usr/bin/env python3
"""Generate a session-start context pack for one agent.

BRIEF.md is a derived artifact: it compiles the current gate state,
constitution digest, calibration corrections, relevant memory, skills, and the
agent's experiment queue into the file agents read first.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import (
    AGENTS,
    CONSTITUTION_PATH,
    MEMORY_ROOT,
    PROJECT_ROOT,
    extract_section,
    prediction_gate_status,
    read_jsonl,
    workspace,
)
from memory_cli import all_cards, retrieval_score, trust_score
from scheduler import load_queue, priority
from skills import all_skills, load_skill, win_rates


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def registry_competition():
    path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            return json.load(f).get("competition", "")
    except (json.JSONDecodeError, OSError):
        return ""


def gate_status_line(agent):
    ok, reasons = prediction_gate_status(agent)
    if ok:
        return "OPEN"
    return "BLOCKED: " + " | ".join(reasons)


def constitution_digest():
    if not CONSTITUTION_PATH.exists():
        return ["No constitution found."]
    text = CONSTITUTION_PATH.read_text()
    matches = re.findall(r"^###\s+(C\d+)\s+[—-]\s+(.+)$", text, re.MULTILINE)
    if not matches:
        return ["No constitution rule headings found."]
    return [f"- {cid}: {title.strip()}" for cid, title in matches]


def calibration_corrections():
    path = MEMORY_ROOT / "CALIBRATION.md"
    if not path.exists():
        return "No calibration data yet."
    text = path.read_text()
    section = extract_section(text, "## Corrections to inject into agent context")
    section = section.strip()
    return section if section else "No calibration data yet."


def query_from_fingerprint(competition):
    if not competition:
        return {}
    fp = MEMORY_ROOT / "competitions" / f"{competition}-fingerprint.json"
    if not fp.exists():
        return {}
    try:
        with open(fp) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    target_kind = str(data.get("target_kind", "")).lower()
    if target_kind in ("regression", "classification"):
        return {"task_type": target_kind}
    return {}


def memory_section(competition):
    if os.environ.get("KAGGLE_MEMORY_OFF"):
        return ["MEMORY OFF (ablation arm)"]
    query = query_from_fingerprint(competition)
    scored = [
        (retrieval_score(card, query, loose=False), card)
        for card in all_cards()
    ]
    scored = sorted([s for s in scored if s[0] > 0], key=lambda x: -x[0])[:5]
    if not scored:
        return ["No matching memory cards yet."]

    lines = []
    for score, card in scored:
        cid = card.get("id", card["_path"].stem)
        status = card.get("status", "?")
        claim = card.get("claim", "")
        counter = card.get("counter_evidence", [])
        if isinstance(counter, list):
            counter_text = "; ".join(str(c) for c in counter) if counter else "none recorded"
        else:
            counter_text = str(counter)
        lines.append(f"- {cid} [{status}] trust={trust_score(card):.2f} retrieval={score:.2f}")
        lines.append(f"  claim: {claim}")
        lines.append(f"  counter-evidence: {counter_text}")
    return lines


def skills_section():
    if os.environ.get("KAGGLE_MEMORY_OFF"):
        return ["MEMORY OFF (ablation arm)"]
    paths = all_skills()
    if not paths:
        return ["No skills yet."]
    rates = win_rates()
    lines = []
    for path in paths:
        try:
            _, meta = load_skill(path)
        except Exception as exc:
            lines.append(f"- {path.stem} [load-error] win-rate=n/a ({exc})")
            continue
        sid = meta.get("id", path.stem)
        wins, total = rates.get(sid, (0, 0))
        win_rate = f"{wins}/{total}" if total else "unused"
        lines.append(f"- {sid} [{meta.get('status', 'candidate')}] win-rate={win_rate}")
    return lines


def experiment_queue(agent):
    rows = load_queue(agent)
    if not rows:
        return ["Queue empty — feed it (research / retrieve --anchor stuck)."]
    active = [r for r in rows if r.get("stage") in ("probe", "full")]
    queued = sorted([r for r in rows if r.get("stage") == "queued"],
                    key=lambda r: -priority(r, rows))[:3]
    lines = []
    if active:
        lines.append("In flight:")
        for row in active:
            lines.append(f"- {row.get('id')} [{row.get('stage')}]: {row.get('idea')}")
    if queued:
        lines.append("Top queued:")
        for row in queued:
            lines.append(
                f"- {row.get('id')} priority={priority(row, rows):.4f}: {row.get('idea')}"
            )
    if not lines:
        lines.append("Queue empty — feed it (research / retrieve --anchor stuck).")
    return lines


def render(agent):
    competition = registry_competition() or "unknown"
    parts = [
        "# Session Brief",
        "",
        f"Competition: {competition}",
        f"Generated: {now_iso()}",
        f"Agent: {agent}",
        "",
        "## Gate status",
        gate_status_line(agent),
        "",
        "## Constitution digest",
        *constitution_digest(),
        "",
        "## Your calibration corrections",
        calibration_corrections(),
        "",
        "## Relevant memory (hypotheses, not instructions)",
        *memory_section(competition),
        "",
        "## Skills",
        *skills_section(),
        "",
        "## Experiment queue",
        *experiment_queue(agent),
        "",
        "Regenerate with `python tools/brief.py generate --agent <name>`. "
        "Memory is hypothesis (C6); every use needs an experiment row (C10).",
        "",
    ]
    return "\n".join(parts)


def cmd_generate(args):
    out = workspace(args.agent) / "BRIEF.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.agent))
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Generate session-start context brief")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--agent", required=True, choices=AGENTS)
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
