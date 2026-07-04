#!/usr/bin/env python3
"""Prediction-gated reading tool.

Enforces constitution rule C2: no reading of competition discussions, public
notebooks, or winner writeups until the agent's PREDICTION.md is pre-registered.
This is Um_nik's "think before you open the editorial" implemented as a
mechanism instead of advice.

Usage:
  python tools/writeup.py check --agent claude [--url URL]
  python tools/writeup.py log   --agent claude --url URL [--kind writeup] [--note "..."] [--force]
  python tools/writeup.py hook            # Claude Code PreToolUse hook mode (reads JSON on stdin)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import (
    gated_url,
    ledger_event,
    now_iso,
    append_jsonl,
    prediction_gate_status,
    reading_log_path,
)

READ_KINDS = ["discussion", "notebook", "writeup", "docs", "paper", "blog", "other"]


def check(agent, url=None, quiet=False):
    ok, reasons = prediction_gate_status(agent)
    if ok:
        ledger_event("C2", "fired", agent, f"gate passed url={url or '-'}")
        if not quiet:
            print(f"ALLOWED: prediction gate passed for {agent}.")
        return 0
    ledger_event("C2", "prevented", agent, f"gate blocked url={url or '-'}")
    if not quiet:
        print(f"BLOCKED: prediction gate failed for {agent}.")
        for r in reasons:
            print(f"  - {r}")
        print("\nFill agents/%s/workspace/PREDICTION.md, then retry." % agent)
    return 1


def log_read(agent, url, kind, note, force=False):
    ok, reasons = prediction_gate_status(agent)
    gate = "allowed"
    if not ok:
        if not force:
            print("BLOCKED: cannot log a read while the prediction gate is failing.")
            for r in reasons:
                print(f"  - {r}")
            print("Use --force only for metric clarifications; overrides are logged.")
            ledger_event("C2", "prevented", agent, f"log blocked url={url}")
            return 1
        gate = "override"
        ledger_event("C2", "violated", agent, f"forced read url={url}")

    append_jsonl(reading_log_path(agent), {
        "ts": now_iso(),
        "agent": agent,
        "url": url,
        "kind": kind,
        "note": note or "",
        "gate": gate,
    })
    print(f"LOGGED ({gate}): {kind} {url}")
    if gate == "allowed":
        print("Reminder (C3): this read must produce an experiment row or an explicit rejection.")
    return 0


URL_IN_COMMAND_RE = re.compile(r"https?://[^\s'\"\\)>]+")


def hook_mode(agent_arg=None):
    """PreToolUse hook for Claude Code and Codex CLI (same payload shape).

    Blocks gated Kaggle URLs when the gate fails — whether the URL arrives via a
    web-fetch tool (`tool_input.url`) or embedded in a shell command
    (`tool_input.command`, e.g. curl/wget). Exit 2 blocks the tool call and
    feeds stderr back to the agent; exit 0 allows it.
    """
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed payload — never block on our own bug

    tool_input = payload.get("tool_input", {}) or {}
    url = tool_input.get("url", "")
    if not gated_url(url):
        url = ""
        command = tool_input.get("command", "")
        if command:
            for found in URL_IN_COMMAND_RE.findall(command):
                if gated_url(found):
                    url = found
                    break
    if not url:
        return 0

    agent = agent_arg or os.environ.get("KAGGLE_AGENT", "claude")
    ok, reasons = prediction_gate_status(agent)
    if ok:
        ledger_event("C2", "fired", agent, f"hook allowed url={url}")
        append_jsonl(reading_log_path(agent), {
            "ts": now_iso(),
            "agent": agent,
            "url": url,
            "kind": "auto",
            "note": "via PreToolUse hook",
            "gate": "allowed",
        })
        return 0

    ledger_event("C2", "prevented", agent, f"hook blocked url={url}")
    print(
        "Blocked by the Kaggle practice constitution (C2: predict before you read).\n"
        "Complete agents/%s/workspace/PREDICTION.md — both the Naive Default Playbook "
        "and the Actual Prediction tables — before reading discussions, public "
        "notebooks, or winner writeups.\nReasons:\n%s"
        % (agent, "\n".join(f"  - {r}" for r in reasons)),
        file=sys.stderr,
    )
    return 2


def main():
    parser = argparse.ArgumentParser(description="Prediction-gated reading tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check whether reading is allowed")
    p_check.add_argument("--agent", required=True, choices=["claude", "codex", "gemini"])
    p_check.add_argument("--url", default=None)
    p_check.add_argument("--quiet", action="store_true")

    p_log = sub.add_parser("log", help="Record a read in the reading ledger")
    p_log.add_argument("--agent", required=True, choices=["claude", "codex", "gemini"])
    p_log.add_argument("--url", required=True)
    p_log.add_argument("--kind", default="writeup", choices=READ_KINDS)
    p_log.add_argument("--note", default="")
    p_log.add_argument("--force", action="store_true",
                       help="Override the gate (logged as a violation)")

    p_hook = sub.add_parser("hook", help="PreToolUse hook mode (stdin JSON)")
    p_hook.add_argument("--agent", default=None, choices=["claude", "codex", "gemini"],
                        help="agent identity (default: $KAGGLE_AGENT or claude)")

    args = parser.parse_args()
    if args.command == "check":
        sys.exit(check(args.agent, args.url, args.quiet))
    elif args.command == "log":
        sys.exit(log_read(args.agent, args.url, args.kind, args.note, args.force))
    elif args.command == "hook":
        sys.exit(hook_mode(args.agent))


if __name__ == "__main__":
    main()
