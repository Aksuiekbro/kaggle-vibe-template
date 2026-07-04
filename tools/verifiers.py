#!/usr/bin/env python3
"""Cheap mechanical verifiers.

Generation is expensive and unreliable; verifying properties of data and
submissions is cheap and reliable. Each check converts a class of silent
catastrophic failure (leakage, drift, LB-chasing) into a loud flag.
Pure stdlib; large files are sampled.

Usage:
  python tools/verifiers.py columns --train t.csv [--target col] [--test te.csv] [--id-col col]
  python tools/verifiers.py cv-lb  --agent claude
  python tools/verifiers.py folds  --cv-scores 0.81 0.82 0.80 ...
  python tools/verifiers.py self-test
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import PROJECT_ROOT

SAMPLE_ROWS = 5000
ID_LEAK_RHO = 0.10
DRIFT_FLAG = 0.10


def sample_csv(path, max_rows=SAMPLE_ROWS, seed=13):
    """Reservoir-sample rows; returns (header, rows)."""
    rng = random.Random(seed)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, row in enumerate(reader):
            if len(row) != len(header):
                continue
            if len(rows) < max_rows:
                rows.append(row)
            else:
                j = rng.randint(0, i)
                if j < max_rows:
                    rows[j] = row
    return header, rows


def to_float(values):
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            return None
    return out


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def looks_like_id(name, values):
    lname = name.lower()
    name_hit = lname in ("id", "index", "row_id") or lname.endswith("_id") or lname.startswith("id_")
    uniq = len(set(values)) / max(1, len(values))
    return name_hit or uniq > 0.98


def numeric_drift(train_vals, test_vals):
    """Mean absolute difference of decile quantiles, scaled by train IQR."""
    tr, te = sorted(train_vals), sorted(test_vals)

    def q(sorted_vals, p):
        idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
        return sorted_vals[idx]

    iqr = q(tr, 0.75) - q(tr, 0.25) or (abs(q(tr, 0.5)) or 1.0)
    diffs = [abs(q(tr, p / 10) - q(te, p / 10)) for p in range(1, 10)]
    return (sum(diffs) / len(diffs)) / abs(iqr)


def categorical_drift(train_vals, test_vals):
    """Half L1 distance between category frequency distributions."""
    def freqs(vals):
        f = {}
        for v in vals:
            f[v] = f.get(v, 0) + 1
        return {k: c / len(vals) for k, c in f.items()}
    ftr, fte = freqs(train_vals), freqs(test_vals)
    keys = set(ftr) | set(fte)
    return 0.5 * sum(abs(ftr.get(k, 0) - fte.get(k, 0)) for k in keys)


def cmd_columns(args):
    header, rows = sample_csv(args.train)
    cols = {name: [r[i] for r in rows] for i, name in enumerate(header)}
    findings = []

    target_vals = None
    if args.target:
        if args.target not in cols:
            print(f"ERROR: target '{args.target}' not in columns")
            return 1
        target_vals = to_float(cols[args.target])
        if target_vals is None:
            classes = sorted(set(cols[args.target]))
            mapping = {c: i for i, c in enumerate(classes)}
            target_vals = [float(mapping[v]) for v in cols[args.target]]

    test_cols = None
    if args.test:
        te_header, te_rows = sample_csv(args.test)
        test_cols = {name: [r[i] for r in te_rows] for i, name in enumerate(te_header)}

    for name, values in cols.items():
        if name == args.target:
            continue
        if len(set(values)) <= 1:
            findings.append(("CONSTANT_COLUMN", name, "single value — drop it"))
            continue
        is_id = looks_like_id(name, values) or (args.id_col and name == args.id_col)
        numeric = to_float(values)

        if target_vals is not None and numeric is not None:
            rho = abs(spearman(numeric, target_vals))
            if is_id and rho > ID_LEAK_RHO:
                findings.append(("ID_TARGET_LEAK", name,
                                 f"ID-like column correlates with target (|rho|={rho:.3f}) — "
                                 "ordering leak; the test set will not have it"))
            elif rho > 0.995:
                findings.append(("PERFECT_PREDICTOR", name,
                                 f"|rho|={rho:.3f} with target — almost certainly leakage"))

        if test_cols is not None and name in test_cols:
            te_vals = test_cols[name]
            te_numeric = to_float(te_vals)
            if numeric is not None and te_numeric is not None:
                d = numeric_drift(numeric, te_numeric)
            else:
                d = categorical_drift(values, te_vals)
            if d > DRIFT_FLAG:
                findings.append(("TRAIN_TEST_DRIFT", name,
                                 f"drift score {d:.3f} — random KFold may be optimistic; "
                                 "see card adversarial-validation-before-cv"))

    # duplicate columns
    sig = {}
    for name, values in cols.items():
        key = tuple(values[:200])
        if key in sig:
            findings.append(("DUPLICATE_COLUMN", name, f"identical to '{sig[key]}' (first 200 sampled rows)"))
        else:
            sig[key] = name

    if findings:
        print(f"Verifier findings ({len(findings)}):")
        for code, col, detail in findings:
            print(f"  [{code}] {col}: {detail}")
        return 1
    print(f"columns: clean ({len(header)} columns, {len(rows)} sampled rows).")
    return 0


def cmd_cv_lb(args):
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    if not registry_path.exists():
        print("No registry.")
        return 0
    with open(registry_path) as f:
        registry = json.load(f)
    pairs = [(s["local_score"], s["kaggle_score"])
             for s in registry.get("submissions", [])
             if s.get("agent") == args.agent and s.get("status") == "submitted"
             and s.get("local_score") is not None and s.get("kaggle_score") is not None]
    if len(pairs) < 3:
        print(f"cv-lb: only {len(pairs)} scored submission pair(s) — need >=3 for a read.")
        return 0
    local, lb = [p[0] for p in pairs], [p[1] for p in pairs]
    r = pearson(local, lb)
    sign_agree = sum(
        1 for i in range(1, len(pairs))
        if (local[i] - local[i - 1]) * (lb[i] - lb[i - 1]) > 0
    ) / max(1, len(pairs) - 1)
    print(f"cv-lb agreement for {args.agent}: pearson={r:.3f}, "
          f"delta sign agreement={sign_agree:.0%} over {len(pairs)} submissions")
    if r < 0.5 or sign_agree < 0.5:
        print("  WARNING: local CV is not tracking the leaderboard — fix validation before "
              "iterating further; improving a broken proxy is worse than useless.")
        return 1
    return 0


def cmd_folds(args):
    scores = args.cv_scores
    if len(scores) < 3:
        print("Need >=3 fold scores.")
        return 1
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    print(f"folds: mean={mean:.6f} std={std:.6f}")
    issues = 0
    if all(abs(scores[i] - scores[i + 1]) < 1e-6 for i in range(len(scores) - 1)):
        print("  WARNING: all folds identical — leakage or degenerate split.")
        issues += 1
    if std > 0.05 * abs(mean or 1):
        print("  WARNING: high fold variance (>5% of mean) — unstable model or bad split.")
        issues += 1
    return 1 if issues else 0


def cmd_reconcile(args):
    """Detect submissions that bypassed the score gate: compare Kaggle's own
    submission count for this competition against the registry's."""
    import subprocess
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    if not registry_path.exists():
        print("No registry.")
        return 0
    with open(registry_path) as f:
        registry = json.load(f)
    slug = registry.get("competition", "")
    gated = len([s for s in registry.get("submissions", []) if s.get("status") == "submitted"])
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", slug, "--csv"],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        print("kaggle CLI not available — cannot reconcile.")
        return 0
    if result.returncode != 0:
        print(f"Could not fetch Kaggle submissions: {result.stderr.strip()[:200]}")
        return 0
    kaggle_count = max(0, len([l for l in result.stdout.strip().split("\n") if l.strip()]) - 1)
    print(f"reconcile: kaggle shows {kaggle_count} submission(s), "
          f"registry gated {gated} for '{slug}'")
    if kaggle_count > gated:
        print(f"  WARNING: {kaggle_count - gated} submission(s) reached Kaggle without "
              "passing the score gate (direct CLI/web submission). All submissions "
              "must go through tools/submit.py (C1).")
        return 1
    return 0


def cmd_self_test(args):
    # spearman sanity
    assert abs(spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-9
    # drift: identical distributions -> ~0; shifted -> large
    rng = random.Random(1)
    a = [rng.gauss(0, 1) for _ in range(500)]
    b = [rng.gauss(0, 1) for _ in range(500)]
    c = [rng.gauss(3, 1) for _ in range(500)]
    assert numeric_drift(a, b) < 0.15, "same-distribution drift should be small"
    assert numeric_drift(a, c) > 1.0, "shifted-distribution drift should be large"
    assert categorical_drift(["x"] * 90 + ["y"] * 10, ["x"] * 10 + ["y"] * 90) > 0.5
    # id detection
    assert looks_like_id("user_id", ["1", "2", "3"])
    assert looks_like_id("foo", [str(i) for i in range(100)])
    assert not looks_like_id("age", ["30", "30", "41", "30"])
    print("verifiers: self-test passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Cheap mechanical verifiers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("columns")
    p.add_argument("--train", required=True)
    p.add_argument("--target")
    p.add_argument("--test")
    p.add_argument("--id-col")
    p.set_defaults(func=cmd_columns)

    p = sub.add_parser("cv-lb")
    p.add_argument("--agent", required=True)
    p.set_defaults(func=cmd_cv_lb)

    sub.add_parser("reconcile").set_defaults(func=cmd_reconcile)

    p = sub.add_parser("folds")
    p.add_argument("--cv-scores", type=float, nargs="+", required=True)
    p.set_defaults(func=cmd_folds)

    sub.add_parser("self-test").set_defaults(func=cmd_self_test)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
