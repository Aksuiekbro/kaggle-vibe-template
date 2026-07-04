#!/usr/bin/env python3
"""Shared helpers for the procedural-gate layer: workspace paths, ledgers, prediction gate."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("KAGGLE_TEMPLATE_ROOT", Path(__file__).parent.parent))
AGENTS = ["claude", "codex", "gemini"]
MEMORY_ROOT = PROJECT_ROOT / ".ai" / "memory"
CONSTITUTION_PATH = PROJECT_ROOT / ".ai" / "constitution.md"
CONSTITUTION_LEDGER = MEMORY_ROOT / "CONSTITUTION_LEDGER.jsonl"

# Minimum filled rows required in each PREDICTION.md table before reading is allowed
MIN_FILLED_PREDICTIONS = 3


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def workspace(agent):
    return PROJECT_ROOT / "agents" / agent / "workspace"


def reading_log_path(agent):
    return workspace(agent) / "READING_LOG.jsonl"


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def ledger_event(rule, event, agent=None, detail=""):
    """Record a constitution-rule event: fired | violated | prevented.

    This is the rent ledger — rules that never fire become demotion candidates
    at consolidation time.
    """
    append_jsonl(CONSTITUTION_LEDGER, {
        "ts": now_iso(),
        "rule": rule,
        "event": event,
        "agent": agent,
        "detail": detail,
    })


def extract_section(text, heading):
    """Return the body of a markdown section from `heading` to the next same-level heading."""
    lines = text.split("\n")
    level = heading.split(" ")[0]  # e.g. "##"
    body, capturing = [], False
    for line in lines:
        if line.strip().startswith(heading):
            capturing = True
            continue
        if capturing and line.startswith(level + " "):
            break
        if capturing:
            body.append(line)
    return "\n".join(body)


def parse_table_rows(text):
    """Parse markdown table rows into lists of cell strings (header + separator skipped)."""
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and set(cells[0]) <= {"-", " ", ":"} and len(cells[0]) >= 3:
            continue  # separator row
        rows.append(cells)
    return rows[1:] if rows else []  # drop header row


def prediction_gate_status(agent):
    """Check whether the agent's pre-registered prediction is complete enough to unlock reading.

    Returns (ok, reasons). Enforces constitution rule C2 (predict before you read)
    and C3's baseline correction (naive default playbook must be written so only
    deviations count as informative hits).
    """
    pred = workspace(agent) / "PREDICTION.md"
    if not pred.exists():
        return False, [
            f"PREDICTION.md not found at {pred}. "
            "Run tools/setup.py (or copy the template) and fill it in before reading "
            "discussions, notebooks, or winner writeups."
        ]

    text = pred.read_text()
    reasons = []

    def filled_rows(heading):
        rows = parse_table_rows(extract_section(text, heading))
        return sum(1 for r in rows if len(r) >= 2 and r[0] and r[1])

    naive = filled_rows("## Naive Default Playbook")
    if naive < MIN_FILLED_PREDICTIONS:
        reasons.append(
            f"Naive Default Playbook has {naive}/{MIN_FILLED_PREDICTIONS}+ filled rows. "
            "Write the boring default first — only deviations from it count as informative hits."
        )

    actual = filled_rows("## Actual Prediction")
    if actual < MIN_FILLED_PREDICTIONS:
        reasons.append(
            f"Actual Prediction has {actual}/{MIN_FILLED_PREDICTIONS}+ filled rows. "
            "Pre-register your playbook prediction before reading anything."
        )

    return (not reasons), reasons


def gated_url(url):
    """URLs that must not be read before the prediction gate passes.

    Kaggle discussions, public notebooks/code, and solution writeup pages.
    Data/API/docs URLs stay open.
    """
    if not url:
        return False
    u = url.lower()
    if "kaggle.com" not in u:
        return False
    return any(marker in u for marker in ("/discussion", "/code", "/kernels", "/writeups", "/models?"))
