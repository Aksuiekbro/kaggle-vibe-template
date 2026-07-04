#!/usr/bin/env python3
"""Calibration ledger: confidence-vs-accuracy across scored predictions.

Joins each scored PREDICTION.md's "Actual Prediction" table (confidence column)
with its "Scoring After Close" table (HIT/PARTIAL/MISS) and reports how accurate
each confidence level actually is, overall and per category. The output is meant
to be injected back into agent context: measured self-knowledge beats felt
confidence.

Usage:
  python tools/calibration.py report [--write]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import MEMORY_ROOT, PROJECT_ROOT, extract_section, now_iso, parse_table_rows

ACCURACY = {"hit": 1.0, "partial": 0.5, "miss": 0.0}
CONF_LEVELS = ["high", "medium", "low"]


def normalize_conf(raw):
    r = raw.lower()
    if "high" in r:
        return "high"
    if "med" in r:
        return "medium"
    if "low" in r:
        return "low"
    return ""


def normalize_result(raw):
    r = raw.lower()
    for key in ACCURACY:
        if key in r:
            return key
    return ""


def prediction_files():
    files = list((MEMORY_ROOT / "predictions").glob("*.md"))
    agents_dir = PROJECT_ROOT / "agents"
    if agents_dir.exists():
        files += list(agents_dir.glob("*/workspace/PREDICTION.md"))
    return [f for f in files if f.name not in ("README.md", "INDEX.md")]


def scored_observations():
    """Yield (file, category, confidence, result, deviated) for every scored row."""
    obs = []
    for path in prediction_files():
        text = path.read_text()
        actual = parse_table_rows(extract_section(text, "## Actual Prediction"))
        scoring = parse_table_rows(extract_section(text, "## Scoring After Close"))
        results = {}
        for row in scoring:
            if len(row) >= 2 and row[0]:
                res = normalize_result(row[1])
                if res:
                    results[row[0].strip().lower()] = res
        if not results:
            continue
        for row in actual:
            if len(row) < 3 or not row[0]:
                continue
            category = row[0].strip().lower()
            if category not in results:
                continue
            conf = normalize_conf(row[2])
            deviated = "yes" in row[4].lower() if len(row) >= 5 else False
            if conf:
                obs.append((path, category, conf, results[category], deviated))
    return obs


def build_report(obs):
    lines = ["# Calibration Report", "", f"Generated: {now_iso()[:10]} — "
             f"{len(obs)} scored prediction row(s) across "
             f"{len({o[0] for o in obs})} prediction file(s).", ""]

    def acc(rows):
        return sum(ACCURACY[r[3]] for r in rows) / len(rows)

    lines.append("## Accuracy by stated confidence")
    lines.append("")
    lines.append("| Confidence | n | Accuracy | Reading |")
    lines.append("|------------|---|----------|---------|")
    corrections = []
    expected = {"high": 0.85, "medium": 0.6, "low": 0.35}
    for level in CONF_LEVELS:
        rows = [o for o in obs if o[2] == level]
        if not rows:
            continue
        a = acc(rows)
        gap = a - expected[level]
        if abs(gap) > 0.15 and len(rows) >= 4:
            verdict = "OVERCONFIDENT" if gap < 0 else "UNDERCONFIDENT"
            corrections.append(
                f'When you feel "{level}" confidence, your measured accuracy is '
                f"{a:.0%} (n={len(rows)}) — you are {verdict.lower()} by ~{abs(gap):.0%}.")
        else:
            verdict = "roughly calibrated" if len(rows) >= 4 else "n too small"
        lines.append(f"| {level} | {len(rows)} | {a:.0%} | {verdict} |")

    lines.append("")
    lines.append("## Accuracy by prediction category")
    lines.append("")
    lines.append("| Category | n | Accuracy |")
    lines.append("|----------|---|----------|")
    cats = sorted({o[1] for o in obs})
    for cat in cats:
        rows = [o for o in obs if o[1] == cat]
        lines.append(f"| {cat} | {len(rows)} | {acc(rows):.0%} |")

    deviations = [o for o in obs if o[4]]
    if deviations:
        lines.append("")
        lines.append(f"## Informative deviations: {acc(deviations):.0%} accurate "
                     f"(n={len(deviations)})")
        lines.append("")
        lines.append("Deviations from the naive default are the only predictions that "
                     "demonstrate judgment; this number is the honest measure of edge.")

    if corrections:
        lines.append("")
        lines.append("## Corrections to inject into agent context")
        lines.append("")
        for c in corrections:
            lines.append(f"- {c}")
    return "\n".join(lines) + "\n"


def cmd_report(args):
    obs = scored_observations()
    if not obs:
        print("No scored predictions yet. Calibration accumulates as postmortems score "
              "PREDICTION.md files (gym runs accelerate this).")
        return 0
    report = build_report(obs)
    print(report)
    if args.write:
        out = MEMORY_ROOT / "CALIBRATION.md"
        out.write_text(report)
        print(f"Written to {out.relative_to(PROJECT_ROOT)} — agents read the "
              "'Corrections' section at session start.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Calibration ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
