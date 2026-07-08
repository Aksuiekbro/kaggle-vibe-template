#!/usr/bin/env python3
"""Blocks direct Kaggle CLI calls that would leak private-leaderboard scores.

`tools/sync_scores.py` is the only sanctioned path to Kaggle submission scores:
it writes the PUBLIC score into the registry and appends the PRIVATE score (if
present) to an out-of-repo, operator-only file. Calling the raw `kaggle
competitions submissions` CLI directly prints both columns straight into the
calling agent's context. On finished/gym/late-submission competitions the
private column is already populated (not hidden by Kaggle the way it would be
mid-competition), so a direct CLI call silently defeats the anti-overfitting
firewall the same way manual LB-probing would -- an agent that sees privateScore
per submission can hand-correct individual predictions against it.

Usage:
  python tools/kaggle_guard.py hook --agent claude   # PreToolUse hook (stdin JSON)
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import ledger_event

SUBMISSIONS_RE = re.compile(r"\bkaggle\s+competitions\s+submissions\b")
SYNC_SCRIPT_RE = re.compile(r"sync_scores\.py")


def hook_mode(agent):
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed payload -- never block on our own bug

    command = (payload.get("tool_input", {}) or {}).get("command", "")
    if not command or not SUBMISSIONS_RE.search(command):
        return 0
    if SYNC_SCRIPT_RE.search(command):
        return 0  # the sanctioned wrapper invokes the same CLI internally; that's fine

    ledger_event("C1", "prevented", agent,
                 f"blocked raw kaggle submissions call: {command[:200]}")
    print(
        "Blocked: raw `kaggle competitions submissions` prints privateScore "
        "straight into your context, defeating the anti-overfitting firewall "
        "(see tools/sync_scores.py's docstring). On finished/gym/late-submission "
        "competitions the private column is already populated -- Kaggle isn't "
        "hiding it the way it would mid-competition, so this is the same failure "
        "mode as manual LB-probing.\n"
        "Use `python tools/sync_scores.py` instead: it writes only the PUBLIC "
        "score into the registry and sends PRIVATE scores to an out-of-repo, "
        "operator-only file. For submission status/counts (not scores), read "
        "shared/submissions/registry.json instead.",
        file=sys.stderr,
    )
    return 2


def main():
    parser = argparse.ArgumentParser(description="Kaggle private-score leakage guard")
    sub = parser.add_subparsers(dest="command", required=True)
    p_hook = sub.add_parser("hook", help="PreToolUse hook mode (stdin JSON)")
    p_hook.add_argument("--agent", default="claude", choices=["claude", "codex", "gemini"])
    args = parser.parse_args()
    if args.command == "hook":
        sys.exit(hook_mode(args.agent))


if __name__ == "__main__":
    main()
