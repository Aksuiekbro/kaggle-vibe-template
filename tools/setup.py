#!/usr/bin/env python3
"""Competition workspace initializer. Downloads data and scaffolds the workspace."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
AGENTS = ["claude", "codex", "gemini"]


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


def setup_competition(competition_slug):
    print(f"\n{'='*60}")
    print(f"Setting up workspace for: {competition_slug}")
    print(f"{'='*60}\n")

    print("1. Creating directory structure...")
    dirs = [
        "shared/submissions",
        "shared/best",
        "shared/data",
        ".ai/prompts",
        ".ai/checklists",
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

        print(f"   {agent}: PROGRESS.md, STRATEGY.md created")

    print(f"\n{'='*60}")
    print("Setup complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Fill in COMPETITION.md with problem details")
    print(f"  2. Check shared/data/ for downloaded competition data")
    print(f"  3. Launch agents in separate terminals:")
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
