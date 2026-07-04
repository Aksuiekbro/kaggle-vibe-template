#!/usr/bin/env python3
"""Empirical competition fingerprints.

"Similar enough" should be measured, not vibed. A fingerprint is a small vector
of dataset statistics; retrieval and upsolve-source selection key on fingerprint
distance to past competitions instead of the agent's self-reported similarity.
Pure stdlib; large files are sampled.

Usage:
  python tools/fingerprint.py compute --train t.csv [--test te.csv] [--target col] --slug my-comp [--write]
  python tools/fingerprint.py compare --slug my-comp
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import MEMORY_ROOT, PROJECT_ROOT, now_iso
from verifiers import sample_csv, to_float, numeric_drift, categorical_drift

FP_DIR = MEMORY_ROOT / "competitions"

# numeric fields compared in `compare`, with normalization spans
NUMERIC_FIELDS = {
    "log10_rows": 1.5,
    "log10_cols": 1.0,
    "pct_numeric": 0.4,
    "missing_rate": 0.2,
    "target_entropy": 1.5,
    "target_skew": 2.0,
    "drift_mean": 0.15,
}


def compute_fingerprint(train, test=None, target=None):
    import math

    header, rows = sample_csv(train)
    n_rows_sampled = len(rows)
    # true row count (cheap line count)
    with open(train) as f:
        n_rows = sum(1 for _ in f) - 1
    cols = {name: [r[i] for r in rows] for i, name in enumerate(header)}

    numeric_cols, missing = 0, 0
    for name, values in cols.items():
        if to_float([v for v in values if v != ""]) is not None:
            numeric_cols += 1
        missing += sum(1 for v in values if v == "")

    fp = {
        "log10_rows": round(math.log10(max(10, n_rows)), 3),
        "log10_cols": round(math.log10(max(1, len(header))), 3),
        "pct_numeric": round(numeric_cols / max(1, len(header)), 3),
        "missing_rate": round(missing / max(1, n_rows_sampled * len(header)), 4),
        "target_kind": "",
        "target_entropy": 0.0,
        "target_skew": 0.0,
        "drift_mean": 0.0,
    }

    if target and target in cols:
        tvals = [v for v in cols[target] if v != ""]
        tnum = to_float(tvals)
        cardinality = len(set(tvals))
        if tnum is not None and cardinality > 20:
            fp["target_kind"] = "regression"
            n = len(tnum)
            mean = sum(tnum) / n
            std = (sum((x - mean) ** 2 for x in tnum) / n) ** 0.5 or 1.0
            fp["target_skew"] = round(sum(((x - mean) / std) ** 3 for x in tnum) / n, 3)
        else:
            fp["target_kind"] = "classification" if cardinality <= 20 else "high-card"
            freqs = {}
            for v in tvals:
                freqs[v] = freqs.get(v, 0) + 1
            total = len(tvals)
            fp["target_entropy"] = round(
                -sum((c / total) * math.log(c / total) for c in freqs.values()), 3)

    if test:
        te_header, te_rows = sample_csv(test)
        te_cols = {name: [r[i] for r in te_rows] for i, name in enumerate(te_header)}
        drifts = []
        for name in header:
            if name == target or name not in te_cols:
                continue
            a = [v for v in cols[name] if v != ""]
            b = [v for v in te_cols[name] if v != ""]
            if not a or not b:
                continue
            an, bn = to_float(a), to_float(b)
            drifts.append(numeric_drift(an, bn) if an is not None and bn is not None
                          else categorical_drift(a, b))
        if drifts:
            fp["drift_mean"] = round(sum(drifts) / len(drifts), 4)

    return fp


def fingerprint_files():
    if not FP_DIR.exists():
        return []
    return sorted(FP_DIR.glob("*-fingerprint.json"))


def distance(a, b):
    d, n = 0.0, 0
    for field, span in NUMERIC_FIELDS.items():
        if field in a and field in b:
            d += min(2.0, abs(float(a[field]) - float(b[field])) / span)
            n += 1
    if a.get("target_kind") and b.get("target_kind") and a["target_kind"] != b["target_kind"]:
        d += 2.0
        n += 1
    return d / max(1, n)


def cmd_compute(args):
    fp = compute_fingerprint(args.train, args.test, args.target)
    fp["slug"] = args.slug
    fp["computed"] = now_iso()[:10]
    print(json.dumps(fp, indent=2))
    if args.write:
        FP_DIR.mkdir(parents=True, exist_ok=True)
        out = FP_DIR / f"{args.slug}-fingerprint.json"
        with open(out, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"\nWritten to {out.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_compare(args):
    target_file = FP_DIR / f"{args.slug}-fingerprint.json"
    if not target_file.exists():
        print(f"ERROR: no fingerprint for '{args.slug}' — run compute --write first.")
        return 1
    with open(target_file) as f:
        me = json.load(f)
    others = []
    for path in fingerprint_files():
        if path == target_file:
            continue
        with open(path) as f:
            other = json.load(f)
        others.append((distance(me, other), other))
    if not others:
        print("No other fingerprints stored yet. They accumulate from gym runs and postmortems.")
        return 0
    others.sort(key=lambda x: x[0])
    print(f"Most similar past competitions to '{args.slug}' (lower = closer):\n")
    for d, other in others[:10]:
        print(f"  {d:.3f}  {other.get('slug')}  "
              f"[{other.get('target_kind', '?')}, drift={other.get('drift_mean')}]")
    print("\nUse the closest matches to drive upsolve-source selection and "
          "memory retrieval scope; record the distance in the Similarity column.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Empirical competition fingerprints")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compute")
    p.add_argument("--train", required=True)
    p.add_argument("--test")
    p.add_argument("--target")
    p.add_argument("--slug", required=True)
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_compute)

    p = sub.add_parser("compare")
    p.add_argument("--slug", required=True)
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
