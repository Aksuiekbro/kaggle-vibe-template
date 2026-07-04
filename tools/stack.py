#!/usr/bin/env python3
"""Agent-level ensembling and decision-theoretic final selection.

Three heterogeneous agents produce more decorrelated errors than one agent's
variations — the classic Kaggle ensemble applied at the agent layer. Use during
sharing rounds. Pure stdlib; submissions are CSVs with an id column first and
numeric prediction column(s) after.

Usage:
  python tools/stack.py correlation  --files a.csv b.csv c.csv
  python tools/stack.py blend        --files a.csv b.csv [--weights 0.6 0.4] --out blend.csv
  python tools/stack.py select-finals --files a.csv b.csv c.csv --cv-scores 0.81 0.83 0.80 [--direction maximize]
"""

import argparse
import csv
import sys
from pathlib import Path


def read_submission(path):
    """Returns (header, {id: [floats]}). Assumes column 0 is the id."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = {}
        for row in reader:
            if len(row) != len(header):
                continue
            try:
                data[row[0]] = [float(v) for v in row[1:]]
            except ValueError:
                continue
    return header, data


def aligned_vectors(files):
    """Flatten predictions over the common id set, aligned across files."""
    loaded = [read_submission(f) for f in files]
    common = set(loaded[0][1])
    for _, data in loaded[1:]:
        common &= set(data)
    if not common:
        raise SystemExit("ERROR: no common ids across submissions")
    ids = sorted(common)
    vectors = []
    for _, data in loaded:
        vec = []
        for i in ids:
            vec.extend(data[i])
        vectors.append(vec)
    return loaded[0][0], ids, vectors, loaded


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def cmd_correlation(args):
    _, ids, vectors, _ = aligned_vectors(args.files)
    names = [Path(f).name for f in args.files]
    print(f"Prediction correlation over {len(ids)} common ids:\n")
    width = max(len(n) for n in names) + 2
    print(" " * width + "  ".join(f"{n[:12]:>12}" for n in names))
    for i, ni in enumerate(names):
        row = "  ".join(f"{pearson(vectors[i], vectors[j]):>12.3f}" for j in range(len(names)))
        print(f"{ni:<{width}}{row}")
    print("\nBlending pays when off-diagonal correlation < ~0.95; "
          "near 1.0 the submissions are the same opinion twice.")
    return 0


def cmd_blend(args):
    header, ids, _, loaded = aligned_vectors(args.files)
    k = len(args.files)
    weights = args.weights or [1.0 / k] * k
    if len(weights) != k:
        raise SystemExit("ERROR: --weights count must match --files count")
    total = sum(weights)
    weights = [w / total for w in weights]

    n_pred_cols = len(header) - 1
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in ids:
            blended = [
                sum(w * loaded[m][1][i][c] for m, w in enumerate(weights))
                for c in range(n_pred_cols)
            ]
            writer.writerow([i] + [f"{v:.8g}" for v in blended])
    print(f"Blended {k} submissions -> {args.out} (weights: "
          f"{', '.join(f'{w:.3f}' for w in weights)}, {len(ids)} rows)")
    print("Weights must come from CV, never from public LB (RULES.md Ensemble Discipline).")
    print("Evaluate and submit through the score gate like any other candidate.")
    return 0


def cmd_select_finals(args):
    if len(args.files) != len(args.cv_scores):
        raise SystemExit("ERROR: need one --cv-scores value per file")
    _, ids, vectors, _ = aligned_vectors(args.files)
    names = [Path(f).name for f in args.files]

    best_idx = (max if args.direction == "maximize" else min)(
        range(len(names)), key=lambda i: args.cv_scores[i])

    print(f"Final 1 (CV-best): {names[best_idx]} (CV {args.cv_scores[best_idx]})")

    if len(names) == 1:
        print("Only one candidate — no hedge available.")
        return 0

    # Hedge = the strongest candidate among the least correlated with the CV-best.
    # Pure decorrelation picks noise; so require the hedge to be within tolerance
    # of the best CV, then minimize correlation.
    tol = args.cv_tolerance
    span = max(args.cv_scores) - min(args.cv_scores) or 1.0
    candidates = []
    for i in range(len(names)):
        if i == best_idx:
            continue
        gap = abs(args.cv_scores[i] - args.cv_scores[best_idx]) / span
        candidates.append((i, pearson(vectors[best_idx], vectors[i]), gap))
    eligible = [c for c in candidates if c[2] <= tol] or candidates
    hedge = min(eligible, key=lambda c: c[1])

    print(f"Final 2 (hedge):   {names[hedge[0]]} (CV {args.cv_scores[hedge[0]]}, "
          f"corr with best {hedge[1]:.3f})")
    print("\nRationale: final 1 maximizes expected score; final 2 maximizes the chance "
          "one of the two survives a shake-up. Never pick finals by public LB "
          "(see failure card final-selection-by-public-lb).")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Agent-level stacking and final selection")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("correlation")
    p.add_argument("--files", nargs="+", required=True)
    p.set_defaults(func=cmd_correlation)

    p = sub.add_parser("blend")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_blend)

    p = sub.add_parser("select-finals")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--cv-scores", type=float, nargs="+", required=True)
    p.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    p.add_argument("--cv-tolerance", type=float, default=0.5,
                   help="hedge must be within this fraction of the CV span of the best")
    p.set_defaults(func=cmd_select_finals)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
