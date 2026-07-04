#!/usr/bin/env python3
"""Competition workspace initializer. Downloads data and scaffolds the workspace."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AGENTS = ["claude", "codex", "gemini"]


def prediction_template(agent, competition_slug):
    created = datetime.now(timezone.utc).isoformat()
    return f"""# Prospective Prediction

Competition: {competition_slug}
Agent: {agent}
Created: {created}
Status: open

Complete this before reading current competition discussions, public notebooks, or winner-style solution threads. The goal is to pre-register your best playbook prediction so postmortems can measure whether research and memory improve judgment.

## Naive Default Playbook

Write the boring default for this competition type. These predictions are baseline expectations and do not count as informative hits unless the actual prediction makes a useful, specific deviation.

| Category | Default prediction |
|----------|--------------------|
| CV scheme | |
| Feature families | |
| Model families | |
| Ensembling/postprocessing | |
| Leakage/drift/metric traps | |

## Actual Prediction

| Category | Prediction | Confidence | Memory/source used | Deviation from default? |
|----------|------------|------------|--------------------|-------------------------|
| CV scheme | | | | |
| Feature families | | | | |
| Model families | | | | |
| Ensembling/postprocessing | | | | |
| Leakage/drift/metric traps | | | | |

## Informative Deviations

List the non-obvious predictions that should be scored most heavily later.

| Claim | Why it differs from default | Expected impact | How to test during competition |
|-------|-----------------------------|-----------------|--------------------------------|

## Scoring After Close

Do not fill this until private leaderboard results or reliable winner writeups are available.

| Category | HIT/PARTIAL/MISS | Winner evidence | Notes |
|----------|------------------|-----------------|-------|
| CV scheme | | | |
| Feature families | | | |
| Model families | | | |
| Ensembling/postprocessing | | | |
| Leakage/drift/metric traps | | | |

Baseline-adjusted score:
Memory cards to update:
"""


def plan_draft_template(agent):
    return f"""# {agent.title()} Plan Draft

Draft shared PLAN.md updates here during autonomous work. A human or coordinator consolidates these during sharing rounds.

| Idea | Priority | Evidence | Expected impact | Cost | Status | Notes |
|------|----------|----------|-----------------|------|--------|-------|
"""


def memory_candidates_template(agent):
    return f"""# {agent.title()} Memory Candidates

Draft memory-card candidates here. Do not write directly to `.ai/memory/` during active competition work.

| Claim | Scope | Evidence | Counter-evidence | Predicted impact | Status |
|-------|-------|----------|------------------|------------------|--------|
"""


def run_kaggle_cmd(args_list):
    """Run a kaggle CLI command."""
    try:
        result = subprocess.run(
            ["kaggle"] + args_list,
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return False, "", "kaggle CLI not found. Install with: pip install kaggle"
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def archive_previous_competition(competition_slug):
    """If a different competition was active, archive agent workspaces so stale
    STRATEGY/PREDICTION files can't leak into (or unlock gates for) the new one."""
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    if not registry_path.exists():
        return
    try:
        with open(registry_path) as f:
            old_slug = json.load(f).get("competition")
    except (json.JSONDecodeError, OSError):
        return
    if not old_slug or old_slug in ("unknown", competition_slug):
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = PROJECT_ROOT / ".ai" / "runs" / f"{stamp}-{old_slug}-final"
    moved = []
    for agent in AGENTS:
        ws = PROJECT_ROOT / "agents" / agent / "workspace"
        if ws.exists() and any(ws.iterdir()):
            dest = archive_root / agent
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ws), str(dest))
            ws.mkdir(parents=True)
            moved.append(agent)
    if moved:
        print(f"Previous competition '{old_slug}' workspaces archived to "
              f"{archive_root.relative_to(PROJECT_ROOT)} ({', '.join(moved)}).")
        print("Run the postmortem on them before deleting anything.\n")


def setup_competition(competition_slug):
    print(f"\n{'='*60}")
    print(f"Setting up workspace for: {competition_slug}")
    print(f"{'='*60}\n")

    archive_previous_competition(competition_slug)

    print("1. Creating directory structure...")
    dirs = [
        "shared/submissions",
        "shared/best",
        "shared/data",
        ".ai/prompts",
        ".ai/checklists",
        ".ai/memory",
        ".ai/memory/predictions",
        ".ai/memory/templates",
        ".ai/memory/patterns",
        ".ai/memory/failures",
        ".ai/memory/competitions",
        ".ai/memory/skills",
        ".ai/strategies",
        ".ai/reviews",
        ".ai/scratch",
        ".ai/runs",
    ]
    for agent in AGENTS:
        dirs.extend([
            f"agents/{agent}/workspace",
            f"agents/{agent}/submissions",
        ])

    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("   Done.")

    print("\n2. Initializing submission registry...")
    registry = {
        "competition": competition_slug,
        "submissions": [],
        "best_scores": {agent: None for agent in AGENTS},
        "daily_counts": {},
    }
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"   Registry: {registry_path}")

    predictions_index = PROJECT_ROOT / ".ai" / "memory" / "predictions" / "INDEX.md"
    if not predictions_index.exists():
        predictions_index.write_text(
            "# Open Prediction Index\n\n"
            "Prospective predictions copied here are awaiting postmortem scoring.\n\n"
            "| Competition | Agent | Prediction file | Created | Status | Scored date | Notes |\n"
            "|-------------|-------|-----------------|---------|--------|-------------|-------|\n"
        )

    print("\n3. Downloading competition data...")
    data_dir = PROJECT_ROOT / "shared" / "data"
    ok, stdout, stderr = run_kaggle_cmd([
        "competitions", "download",
        "-c", competition_slug,
        "-p", str(data_dir),
    ])
    if ok:
        print(f"   Downloaded to {data_dir}")
        zip_files = list(data_dir.glob("*.zip"))
        if zip_files:
            print("   Extracting zip files...")
            import zipfile
            for zf in zip_files:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(data_dir)
                zf.unlink()
            print("   Extracted.")
    else:
        print(f"   WARNING: Could not download data: {stderr}")
        print(f"   You may need to accept competition rules on kaggle.com first.")
        print(f"   Or download manually to {data_dir}")

    print("\n4. Fetching competition info...")
    ok, stdout, stderr = run_kaggle_cmd([
        "competitions", "list",
        "-s", competition_slug,
        "--csv",
    ])
    comp_info = ""
    if ok and stdout:
        lines = stdout.strip().split("\n")
        if len(lines) > 1:
            comp_info = lines[1]

    competition_md = PROJECT_ROOT / "COMPETITION.md"
    if competition_md.read_text().startswith("# Competition: [Name]"):
        content = competition_md.read_text()
        content = content.replace("[Name]", competition_slug)
        content = content.replace("[slug]", competition_slug)
        if comp_info:
            content = content.replace(
                "[Plain language description of what needs to be solved. Copy from competition page + your own understanding. Be specific about inputs, outputs, and constraints.]",
                f"[TODO: Fill in from competition page]\n\nKaggle info: {comp_info}",
            )
        competition_md.write_text(content)
        print(f"   Updated COMPETITION.md with competition slug")

    print("\n5. Creating agent workspace scaffolds...")
    for agent in AGENTS:
        workspace = PROJECT_ROOT / "agents" / agent / "workspace"

        progress = workspace / "PROGRESS.md"
        if not progress.exists():
            progress.write_text(f"# {agent.title()} Progress Log\n\n| Time | Action | Result | Score |\n|------|--------|--------|-------|\n")

        strategy = workspace / "STRATEGY.md"
        if not strategy.exists():
            strategy.write_text(
                f"# {agent.title()} Strategy\n\n"
                f"## Current Approach\n[To be developed after reading .ai/strategies/ and COMPETITION.md]\n\n"
                f"## What I've Tried\n\n"
                f"## What Works\n\n"
                f"## What Doesn't Work\n\n"
                f"## Next Experiments\n"
            )

        prediction = workspace / "PREDICTION.md"
        if not prediction.exists():
            prediction.write_text(prediction_template(agent, competition_slug))

        plan_draft = workspace / "PLAN_DRAFT.md"
        if not plan_draft.exists():
            plan_draft.write_text(plan_draft_template(agent))

        memory_candidates = workspace / "MEMORY_CANDIDATES.md"
        if not memory_candidates.exists():
            memory_candidates.write_text(memory_candidates_template(agent))

        print(f"   {agent}: PROGRESS.md, STRATEGY.md, PREDICTION.md, PLAN_DRAFT.md, MEMORY_CANDIDATES.md ready")

    print(f"\n{'='*60}")
    print("Setup complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Fill in COMPETITION.md with problem details")
    print(f"  2. Have agents complete workspace/PREDICTION.md before reading discussions or public notebooks")
    print(f"  3. Check shared/data/ for downloaded competition data")
    print(f"  4. Launch agents in separate terminals:")
    print(f"     - Claude Code: reads CLAUDE.md")
    print(f"     - Codex CLI: reads AGENTS.md")
    print(f"     - Gemini CLI: reads GEMINI.md")
    print(f"\n")


def main():
    parser = argparse.ArgumentParser(description="Initialize workspace for a Kaggle competition")
    parser.add_argument("--competition", required=True, help="Kaggle competition slug (e.g., santa-2025)")
    args = parser.parse_args()

    setup_competition(args.competition)


if __name__ == "__main__":
    main()
