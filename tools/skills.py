#!/usr/bin/env python3
"""Executable skill library.

Skills are verified, reusable code — not prose. Each skill file carries a META
dict (claim, scope, status, provenance) and a self_test() that re-verifies it
on every `skills.py test` run. Usage outcomes accumulate a per-skill win-rate
across competitions; skills that stop winning get flagged for demotion.

Usage:
  python tools/skills.py list [--task-type t]
  python tools/skills.py test [--skill ID]
  python tools/skills.py log-use --skill ID --competition SLUG --outcome win|neutral|loss [--delta X]
  python tools/skills.py stats
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import MEMORY_ROOT, append_jsonl, now_iso, read_jsonl

SKILLS_DIR = MEMORY_ROOT / "skills"
USAGE_LOG = SKILLS_DIR / "USAGE.jsonl"


def load_skill(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = getattr(mod, "META", None)
    if not isinstance(meta, dict) or "id" not in meta:
        raise ValueError(f"{path.name}: missing META dict with an 'id'")
    return mod, meta


def all_skills():
    if not SKILLS_DIR.exists():
        return []
    return sorted(p for p in SKILLS_DIR.glob("*.py") if not p.name.startswith("_"))


def win_rates():
    usage = read_jsonl(USAGE_LOG)
    by_skill = {}
    for u in usage:
        by_skill.setdefault(u.get("skill"), []).append(u)
    rates = {}
    for skill, entries in by_skill.items():
        outcomes = [e.get("outcome") for e in entries]
        wins = outcomes.count("win")
        rates[skill] = (wins, len(outcomes))
    return rates


def cmd_list(args):
    if os.environ.get("KAGGLE_MEMORY_OFF"):
        print("MEMORY OFF (gym ablation arm): skill library disabled for this run.")
        return 0
    skills = all_skills()
    if not skills:
        print("No skills yet. Add verified code to .ai/memory/skills/ (see README there).")
        return 0
    rates = win_rates()
    for path in skills:
        try:
            _, meta = load_skill(path)
        except Exception as e:
            print(f"  {path.name}: LOAD ERROR — {e}")
            continue
        scope = meta.get("scope", {})
        if args.task_type and args.task_type.lower() not in str(scope.get("task_type", "any")).lower():
            continue
        wins, total = rates.get(meta["id"], (0, 0))
        record = f"{wins}/{total} wins" if total else "unused"
        print(f"  {meta['id']}  [{meta.get('status', 'candidate')}]  ({record})")
        print(f"    claim: {meta.get('claim', '')}")
        print(f"    scope: {scope}")
        print(f"    file:  {path.name}\n")
    return 0


def cmd_test(args):
    skills = all_skills()
    if args.skill:
        skills = [p for p in skills if p.stem == args.skill or args.skill in p.stem]
        if not skills:
            print(f"ERROR: no skill matching '{args.skill}'")
            return 1
    failed = 0
    for path in skills:
        try:
            mod, meta = load_skill(path)
            test = getattr(mod, "self_test", None)
            if test is None:
                print(f"  FAIL {path.name}: no self_test() — unverified code is not a skill")
                failed += 1
                continue
            test()
            print(f"  PASS {meta['id']}")
        except Exception as e:
            print(f"  FAIL {path.name}: {e}")
            failed += 1
    print(f"\n{len(skills) - failed}/{len(skills)} skills verified.")
    return 1 if failed else 0


def cmd_log_use(args):
    append_jsonl(USAGE_LOG, {
        "ts": now_iso(),
        "skill": args.skill,
        "competition": args.competition,
        "outcome": args.outcome,
        "delta": args.delta or "",
    })
    print(f"Logged: {args.skill} -> {args.outcome} on {args.competition}")
    wins, total = win_rates().get(args.skill, (0, 0))
    print(f"Win-rate: {wins}/{total}")
    if total >= 3 and wins / total < 0.4:
        print("WARNING: below the 0.4 win-rate floor — demotion candidate at next consolidation.")
    return 0


def cmd_stats(args):
    rates = win_rates()
    if not rates:
        print("No skill usage recorded yet.")
        return 0
    print(f"{'skill':<40} {'wins':<6} {'uses':<6} rate")
    print("-" * 60)
    for skill, (wins, total) in sorted(rates.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1]))):
        print(f"{skill:<40} {wins:<6} {total:<6} {wins / max(1, total):.2f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Executable skill library")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list")
    p.add_argument("--task-type")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("test")
    p.add_argument("--skill")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("log-use")
    p.add_argument("--skill", required=True)
    p.add_argument("--competition", required=True)
    p.add_argument("--outcome", required=True, choices=["win", "neutral", "loss"])
    p.add_argument("--delta")
    p.set_defaults(func=cmd_log_use)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
