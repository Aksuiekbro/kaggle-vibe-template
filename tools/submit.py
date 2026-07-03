#!/usr/bin/env python3
"""Score-gated submitter with MCP → CLI fallback."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from registry import load_registry, save_registry, add_submission, get_agent_best, check_diversity
from evaluate import evaluate, parse_competition_md


def submit_via_cli(competition, filepath, description):
    """Submit via kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit",
             "-c", competition,
             "-f", str(filepath),
             "-m", description],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "kaggle CLI not found. Install with: pip install kaggle"
    except subprocess.TimeoutExpired:
        return False, "Submission timed out after 120 seconds"


def submit_via_mcp(competition, filepath, description):
    """
    Submit via Kaggle MCP.
    This is a placeholder — actual MCP submission depends on the agent's
    MCP client configuration. Agents call this tool from their CLI context
    where MCP is available.
    """
    return False, "MCP submission requires agent's MCP context. Falling back to CLI."


def main():
    parser = argparse.ArgumentParser(description="Score-gated submission with MCP → CLI fallback")
    parser.add_argument("--agent", required=True, choices=["claude", "codex", "gemini"])
    parser.add_argument("--file", required=True, help="Path to submission file")
    parser.add_argument("--description", required=True, help="What changed in this submission")
    parser.add_argument("--approach", default="unknown", help="Approach name for provenance")
    parser.add_argument("--based-on", default=None, help="Submission ID this is based on (for derivatives)")
    parser.add_argument("--cv-scores", type=float, nargs="+", help="CV fold scores")
    parser.add_argument("--force", action="store_true", help="Skip score gate (use with caution)")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    registry = load_registry()
    config = parse_competition_md()
    competition = registry["competition"]

    print(f"\n{'='*60}")
    print(f"Submission Gate — Agent: {args.agent}")
    print(f"{'='*60}")

    print(f"\nEvaluating locally...")
    eval_result = evaluate(args.agent, str(filepath), args.cv_scores)

    if eval_result["score"] is None and not args.force:
        print(f"\nREJECTED: Evaluation failed")
        if eval_result.get("error"):
            print(f"  Error: {eval_result['error']}")
        print(f"  Flags: {', '.join(eval_result['flags'])}")
        add_submission(
            registry, args.agent, str(filepath),
            local_score=None, cv_std=0, flags=eval_result["flags"],
            description=args.description, approach=args.approach,
            status="rejected",
        )
        sys.exit(1)

    local_score = eval_result["score"] or 0.0
    cv_std = eval_result["cv_std"] or 0.0
    flags = eval_result["flags"]

    print(f"  Local score: {local_score}")
    if eval_result["cv_scores"]:
        print(f"  CV scores: {eval_result['cv_scores']}")
        print(f"  CV mean: {eval_result['cv_mean']:.6f}")
        print(f"  CV std: {cv_std:.6f}")
    if flags:
        print(f"  Flags: {', '.join(flags)}")

    current_best = get_agent_best(registry, args.agent)
    if current_best and not args.force:
        best_local = current_best.get("local_score", 0)
        direction = config.get("direction", "maximize")

        if direction == "maximize" and local_score <= best_local:
            print(f"\nREJECTED: Score {local_score} does not beat current best {best_local}")
            print(f"  Direction: {direction}")
            add_submission(
                registry, args.agent, str(filepath),
                local_score=local_score, cv_std=cv_std, flags=flags,
                description=args.description, approach=args.approach,
                status="rejected",
            )
            sys.exit(1)
        elif direction == "minimize" and local_score >= best_local:
            print(f"\nREJECTED: Score {local_score} does not beat current best {best_local}")
            print(f"  Direction: {direction}")
            add_submission(
                registry, args.agent, str(filepath),
                local_score=local_score, cv_std=cv_std, flags=flags,
                description=args.description, approach=args.approach,
                status="rejected",
            )
            sys.exit(1)

    diverse, diversity_msg = check_diversity(registry, args.agent, args.approach)
    if not diverse:
        print(f"\nWARNING: {diversity_msg}")
        print(f"  Submitting anyway, but consider changing strategy.")

    if current_best:
        best_score = current_best.get("local_score", "N/A")
        print(f"\nScore gate PASSED: {local_score} beats current best {best_score}")
    else:
        print(f"\nFirst submission for {args.agent} — no gate check needed")

    print(f"\nSubmitting to Kaggle...")

    mcp_ok, mcp_msg = submit_via_mcp(competition, filepath, args.description)
    method = "mcp"

    if not mcp_ok:
        print(f"  MCP: {mcp_msg}")
        print(f"  Falling back to kaggle CLI...")
        cli_ok, cli_msg = submit_via_cli(competition, str(filepath), args.description)
        method = "cli"

        if not cli_ok:
            print(f"  CLI: {cli_msg}")
            print(f"\nFAILED: Could not submit via MCP or CLI")
            add_submission(
                registry, args.agent, str(filepath),
                local_score=local_score, cv_std=cv_std, flags=flags,
                description=args.description, approach=args.approach,
                method=method, status="failed",
            )
            sys.exit(1)
        print(f"  CLI: {cli_msg}")
    else:
        print(f"  MCP: {mcp_msg}")

    provenance_type = "derivative" if args.based_on else "original"
    sub_id = add_submission(
        registry, args.agent, str(filepath),
        local_score=local_score, cv_std=cv_std, flags=flags,
        description=args.description, approach=args.approach,
        method=method, provenance_type=provenance_type,
        based_on=args.based_on, status="submitted",
    )

    print(f"\nSUBMITTED: {sub_id}")
    print(f"  Agent: {args.agent}")
    print(f"  Method: {method}")
    print(f"  Local score: {local_score}")
    print(f"  Provenance: {provenance_type}" + (f" (based on {args.based_on})" if args.based_on else ""))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
