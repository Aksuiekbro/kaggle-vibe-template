#!/usr/bin/env python3
"""Local evaluation runner with CV stats and overfitting flag detection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
COMPETITION_PATH = PROJECT_ROOT / "COMPETITION.md"


def parse_competition_md():
    """Extract evaluation config from COMPETITION.md."""
    config = {
        "metric": "unknown",
        "direction": "maximize",
        "cv_variance_threshold": 0.005,
    }
    if not COMPETITION_PATH.exists():
        return config

    content = COMPETITION_PATH.read_text()
    for line in content.split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("- **metric**:"):
            config["metric"] = line.split(":", 1)[1].strip().strip("*")
        elif line_lower.startswith("- **direction**:"):
            val = line.split(":", 1)[1].strip().strip("*").lower()
            if val in ("minimize", "maximize"):
                config["direction"] = val
        elif line_lower.startswith("- **cv variance threshold**:"):
            try:
                config["cv_variance_threshold"] = float(line.split(":", 1)[1].strip().strip("*"))
            except ValueError:
                pass
    return config


def compute_file_hash(filepath):
    """SHA256 hash of submission file for duplicate detection."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicate(filepath, registry):
    """Check if this exact file has been submitted before."""
    file_hash = compute_file_hash(filepath)
    for sub in registry.get("submissions", []):
        sub_path = PROJECT_ROOT / sub["file"]
        if sub_path.exists():
            if compute_file_hash(sub_path) == file_hash:
                return True, sub["id"]
    return False, None


def validate_submission_format(filepath):
    """Basic format validation — file exists, not empty, parseable."""
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {filepath}"
    if path.stat().st_size == 0:
        return False, "File is empty"

    if path.suffix == ".csv":
        try:
            with open(path) as f:
                header = f.readline().strip()
                first_row = f.readline().strip()
            if not header:
                return False, "CSV has no header"
            if not first_row:
                return False, "CSV has no data rows"
        except Exception as e:
            return False, f"Cannot read CSV: {e}"

    return True, "OK"


def detect_flags(cv_scores, cv_std, config, filepath, registry):
    """Detect overfitting and other red flags."""
    flags = []

    if cv_std > config["cv_variance_threshold"]:
        flags.append("HIGH_VARIANCE")

    if cv_scores and len(cv_scores) >= 3:
        diffs = [abs(cv_scores[i] - cv_scores[i + 1]) for i in range(len(cv_scores) - 1)]
        if all(d < 1e-6 for d in diffs):
            flags.append("POSSIBLE_LEAKAGE")

    is_dup, dup_id = check_duplicate(filepath, registry)
    if is_dup:
        flags.append(f"DUPLICATE_SUBMISSION:{dup_id}")

    agent_subs = [s for s in registry.get("submissions", []) if s.get("status") == "submitted"]
    if len(agent_subs) >= 3:
        recent_scores = [s["local_score"] for s in agent_subs[-3:] if s["local_score"] is not None]
        if len(recent_scores) >= 3:
            improvements = [recent_scores[i + 1] - recent_scores[i] for i in range(len(recent_scores) - 1)]
            if all(0 < imp < 0.001 for imp in improvements):
                flags.append("DIMINISHING_RETURNS")

    return flags


def evaluate(agent, filepath, cv_scores=None):
    """
    Run local evaluation.

    In a real competition, this would load validation data and compute
    the actual metric. For now, it validates format and checks for red flags.

    Agents should extend this with competition-specific evaluation logic
    by adding an evaluate_score() function to their workspace.
    """
    config = parse_competition_md()

    valid, msg = validate_submission_format(filepath)
    if not valid:
        return {
            "score": None,
            "metric": config["metric"],
            "cv_scores": [],
            "cv_mean": None,
            "cv_std": None,
            "flags": ["FORMAT_ERROR"],
            "fold_count": 0,
            "error": msg,
        }

    registry = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

    if cv_scores is None:
        cv_scores = []

    cv_mean = sum(cv_scores) / len(cv_scores) if cv_scores else None
    cv_std = (
        (sum((s - cv_mean) ** 2 for s in cv_scores) / len(cv_scores)) ** 0.5
        if cv_scores and len(cv_scores) > 1
        else 0.0
    )

    flags = detect_flags(cv_scores, cv_std, config, filepath, registry)

    agent_workspace = PROJECT_ROOT / "agents" / agent / "workspace"
    custom_eval = agent_workspace / "evaluate_score.py"
    score = cv_mean

    if custom_eval.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_eval", custom_eval)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "evaluate_score"):
                score = mod.evaluate_score(filepath)
        except Exception as e:
            flags.append(f"CUSTOM_EVAL_ERROR:{e}")

    result = {
        "score": score,
        "metric": config["metric"],
        "cv_scores": cv_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "flags": flags,
        "fold_count": len(cv_scores),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Local evaluation with CV stats and overfitting flags")
    parser.add_argument("--agent", required=True, choices=["claude", "codex", "gemini"])
    parser.add_argument("--file", required=True, help="Path to submission file")
    parser.add_argument("--cv-scores", type=float, nargs="+", help="CV fold scores (space-separated)")
    args = parser.parse_args()

    result = evaluate(args.agent, args.file, args.cv_scores)

    print(json.dumps(result, indent=2))

    if result["flags"]:
        print(f"\nWarnings:", file=sys.stderr)
        for flag in result["flags"]:
            if flag == "HIGH_VARIANCE":
                print(f"  WARNING: CV std ({result['cv_std']:.4f}) exceeds threshold", file=sys.stderr)
                print(f"  This may indicate overfitting. Consider more regularization.", file=sys.stderr)
            elif flag == "POSSIBLE_LEAKAGE":
                print(f"  WARNING: All CV fold scores are identical — possible data leakage", file=sys.stderr)
            elif flag.startswith("DUPLICATE_SUBMISSION"):
                dup_id = flag.split(":")[1]
                print(f"  WARNING: Identical to previous submission {dup_id}", file=sys.stderr)
            elif flag == "DIMINISHING_RETURNS":
                print(f"  WARNING: Last 3 submissions show <0.001 improvement — diminishing returns", file=sys.stderr)
            elif flag == "FORMAT_ERROR":
                print(f"  ERROR: {result.get('error', 'Invalid submission format')}", file=sys.stderr)

    return 0 if result["score"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
