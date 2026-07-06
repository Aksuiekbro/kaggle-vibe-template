#!/usr/bin/env python3
"""Write Kaggle leaderboard results back into the submission registry.

Fetches this competition's submissions from the Kaggle API and fills in
`kaggle_score` (PUBLIC score only) for registry entries that lack it, matching
by description. Public-only is deliberate: in a live competition the private
score is hidden, so agents must never see per-submission private scores — on
finished competitions (gym/late-submission runs) private scores are appended to
a file OUTSIDE the repo for operator postmortems.

Run periodically (operator ritual or cron):
  python tools/sync_scores.py [--private-out /root/private_scores.csv]
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import PROJECT_ROOT

REGISTRY = PROJECT_ROOT / "shared" / "submissions" / "registry.json"


def fetch_kaggle_submissions(competition):
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR fetching submissions: {result.stderr.strip()[:200]}")
    # skip any warning lines before the CSV header
    lines = result.stdout.strip().splitlines()
    start = next((i for i, l in enumerate(lines) if l.lower().startswith("ref,")), 0)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-out", default=None,
                        help="append private scores here (keep OUTSIDE the repo)")
    args = parser.parse_args()

    with open(REGISTRY) as f:
        registry = json.load(f)
    competition = registry.get("competition")
    kaggle_rows = fetch_kaggle_submissions(competition)

    def score_of(row, field):
        v = (row.get(field) or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    by_desc = {}
    for row in kaggle_rows:
        by_desc.setdefault((row.get("description") or "").strip(), []).append(row)

    updated, private_lines = 0, []
    for sub in registry.get("submissions", []):
        if sub.get("status") != "submitted" or sub.get("kaggle_score") is not None:
            continue
        candidates = by_desc.get(sub.get("description", "").strip(), [])
        if not candidates:
            continue
        row = candidates.pop(0)  # oldest unmatched with same description
        public = score_of(row, "publicScore")
        if public is not None:
            sub["kaggle_score"] = public
            updated += 1
        private = score_of(row, "privateScore")
        if private is not None:
            private_lines.append(f"{sub['id']},{sub['agent']},{public},{private},"
                                 f"{sub.get('description', '')[:60]}")

    with open(REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"sync: {updated} registry entries updated with PUBLIC scores")

    if args.private_out and private_lines:
        out = Path(args.private_out)
        if str(out.resolve()).startswith(str(PROJECT_ROOT.resolve())):
            print("REFUSING to write private scores inside the repo (agent-visible).")
        else:
            with open(out, "a") as f:
                f.write("\n".join(private_lines) + "\n")
            print(f"private scores appended to {out} (operator-only)")

    print("\nNow check proxy quality:  python tools/verifiers.py cv-lb --agent <name>")


if __name__ == "__main__":
    main()
