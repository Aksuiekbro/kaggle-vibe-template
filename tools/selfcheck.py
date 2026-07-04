#!/usr/bin/env python3
"""System self-check: the template's acceptance criteria as one falsifiable exit code.

Runs the full battery — static parsing, unit self-tests, schema validation, and
functional tests of every gate and loop in a throwaway sandbox. Green means the
harness enforces what the docs claim it enforces. Run it after any change to
tools/ or .ai/, and at every consolidation.

Usage:
  python tools/selfcheck.py
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
PY = sys.executable

TOOLS = [
    "discipline.py", "writeup.py", "practice_lint.py", "memory_cli.py",
    "skills.py", "scheduler.py", "gym.py", "verifiers.py", "calibration.py",
    "stack.py", "fingerprint.py", "brief.py", "submit.py", "setup.py", "share.py",
    "evaluate.py", "registry.py", "selfcheck.py",
]

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f" — {detail}" if detail and not ok else ""))


def run(args, env_extra=None, stdin=None, cwd=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PY] + args, capture_output=True, text=True, env=env,
        input=stdin, cwd=cwd or REPO, timeout=120,
    )


FILLED_PREDICTION = """# Prospective Prediction

Status: open

## Naive Default Playbook

| Category | Default prediction |
|----------|--------------------|
| CV scheme | stratified 5-fold |
| Feature families | aggregations |
| Model families | LightGBM |

## Actual Prediction

| Category | Prediction | Confidence | Memory/source used | Deviation from default? |
|----------|------------|------------|--------------------|-------------------------|
| CV scheme | grouped KFold | high | none | yes |
| Feature families | lags | medium | none | yes |
| Model families | LightGBM | high | default | no |

## Scoring After Close

| Category | HIT/PARTIAL/MISS | Winner evidence | Notes |
|----------|------------------|-----------------|-------|
| CV scheme | HIT | | |
| Feature families | MISS | | |
| Model families | HIT | | |
"""


def main():
    print("# 1. Static: all tools parse")
    bad = []
    for name in TOOLS:
        path = REPO / "tools" / name
        try:
            ast.parse(path.read_text())
        except Exception as e:
            bad.append(f"{name}: {e}")
    check("all tools parse", not bad, "; ".join(bad))

    print("# 2. Unit self-tests")
    check("verifiers self-test", run(["tools/verifiers.py", "self-test"]).returncode == 0)
    check("skills self-tests", run(["tools/skills.py", "test"]).returncode == 0)
    check("memory card schema (repo)", run(["tools/memory_cli.py", "validate"]).returncode == 0)

    print("# 3. Functional: gates and loops in a sandbox")
    with tempfile.TemporaryDirectory(prefix="kaggle-selfcheck-") as tmp:
        sb = Path(tmp)
        shutil.copytree(REPO / ".ai", sb / ".ai",
                        ignore=shutil.ignore_patterns("runs", "gym", "*.jsonl"))
        ws = sb / "agents" / "claude" / "workspace"
        ws.mkdir(parents=True)
        env = {"KAGGLE_TEMPLATE_ROOT": str(sb)}

        # C2 gate
        r = run(["tools/writeup.py", "check", "--agent", "claude"], env)
        check("C2 gate blocks without prediction", r.returncode == 1)
        r = run(["tools/brief.py", "generate", "--agent", "claude"], env)
        brief = (ws / "BRIEF.md").read_text() if (ws / "BRIEF.md").exists() else ""
        check("brief generates with empty registry/calibration/queue",
              r.returncode == 0
              and "## Constitution digest" in brief
              and "## Gate status" in brief
              and "## Experiment queue" in brief)
        hook_in = json.dumps({"tool_name": "WebFetch",
                              "tool_input": {"url": "https://www.kaggle.com/c/x/discussion/1"}})
        r = run(["tools/writeup.py", "hook"], env, stdin=hook_in)
        check("C2 hook exits 2 on gated URL", r.returncode == 2)
        r = run(["tools/writeup.py", "hook"], env,
                stdin=json.dumps({"tool_input": {"url": "https://docs.python.org"}}))
        check("C2 hook allows non-gated URL", r.returncode == 0)
        curl_in = json.dumps({"tool_name": "Bash",
                              "tool_input": {"command": "curl -s 'https://www.kaggle.com/c/x/discussion/9'"}})
        r = run(["tools/writeup.py", "hook", "--agent", "codex"], env, stdin=curl_in)
        check("C2 hook blocks curl to gated URL (Bash payload)", r.returncode == 2)
        r = run(["tools/writeup.py", "hook"], env,
                stdin=json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}}))
        check("C2 hook allows unrelated Bash", r.returncode == 0)
        (ws / "PREDICTION.md").write_text(FILLED_PREDICTION)
        r = run(["tools/writeup.py", "check", "--agent", "claude"], env)
        check("C2 gate opens with filled prediction", r.returncode == 0)

        # memory lifecycle: promote gate, write-back, auto-downgrade
        run(["tools/memory_cli.py", "new", "--id", "t-card", "--claim", "test claim"], env)
        card = sb / ".ai" / "memory" / "patterns" / "t-card.md"
        card.write_text(card.read_text().replace("  task_type:", "  task_type: tabular"))
        r = run(["tools/memory_cli.py", "log-miss", "--card", "t-card",
                 "--competition", "c0", "--reason", "should have surfaced"], env)
        miss_path = sb / ".ai" / "memory" / "RETRIEVAL_MISSES.jsonl"
        misses = [json.loads(line) for line in miss_path.read_text().splitlines()] if miss_path.exists() else []
        check("retrieval miss log writes JSONL",
              r.returncode == 0 and misses and misses[-1].get("card") == "t-card")
        r = run(["tools/memory_cli.py", "amend-proposals"], env)
        check("amend proposals include retrieval health",
              r.returncode == 0 and "## Retrieval health" in r.stdout)
        r = run(["tools/memory_cli.py", "promote", "--card", "t-card", "--reviewer", "codex"], env)
        check("C8 promote refused without measured delta", r.returncode == 1)
        run(["tools/memory_cli.py", "writeback", "--card", "t-card", "--competition", "c1",
             "--result", "hit", "--actual-delta", "0.01"], env)
        r = run(["tools/memory_cli.py", "promote", "--card", "t-card", "--reviewer", "codex"], env)
        check("C8 promote passes after hit + reviewer", r.returncode == 0)
        comp_dir = sb / ".ai" / "memory" / "competitions"
        comp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (comp_dir / f"newer-{i}.md").write_text(f"# newer {i}\n")
        r = run(["tools/memory_cli.py", "retrieve", "--task-type", "tabular", "--loose"], env)
        check("retrieve warns validated cards nearing stale",
              r.returncode == 0 and "NEARING STALE" in r.stdout)
        r = run(["tools/memory_cli.py", "revalidation-due", "--agent", "claude"], env)
        check("revalidation-due suggests scheduler probe",
              r.returncode == 0 and "scheduler.py add" in r.stdout and "card:t-card" in r.stdout)
        dedup_root = sb / "dedup-root"
        (dedup_root / ".ai" / "memory" / "templates").mkdir(parents=True)
        shutil.copy2(REPO / ".ai" / "memory" / "templates" / "pattern-card.md",
                     dedup_root / ".ai" / "memory" / "templates" / "pattern-card.md")
        dedup_env = {"KAGGLE_TEMPLATE_ROOT": str(dedup_root)}
        run(["tools/memory_cli.py", "new", "--id", "dedup-a",
             "--claim", "use grouped folds for user entities"], dedup_env)
        run(["tools/memory_cli.py", "new", "--id", "dedup-b",
             "--claim", "apply color augmentations to image models"], dedup_env)
        r = run(["tools/memory_cli.py", "dedup"], dedup_env)
        check("dedup exits 0 for unrelated claims", r.returncode == 0)
        run(["tools/memory_cli.py", "new", "--id", "dedup-c",
             "--claim", "use target encoding with grouped folds"], dedup_env)
        run(["tools/memory_cli.py", "new", "--id", "dedup-d",
             "--claim", "target encoding grouped folds"], dedup_env)
        r = run(["tools/memory_cli.py", "dedup"], dedup_env)
        check("dedup flags near-duplicate claims",
              r.returncode == 1 and "dedup-c" in r.stdout and "dedup-d" in r.stdout)
        for c in ("c2", "c3", "c4"):
            run(["tools/memory_cli.py", "writeback", "--card", "t-card",
                 "--competition", c, "--result", "miss"], env)
        check("C10 three misses auto-downgrade",
              "status: candidate" in card.read_text())

        # scheduler: probe kill + WIP limit
        for i in range(4):
            run(["tools/scheduler.py", "add", "--agent", "claude", "--idea", f"i{i}",
                 "--predicted-delta", "0.01", "--cost-hours", "1"], env)
        for _ in range(3):
            run(["tools/scheduler.py", "next", "--agent", "claude"], env)
        r = run(["tools/scheduler.py", "next", "--agent", "claude"], env)
        check("C5 scheduler blocks 4th concurrent experiment", r.returncode == 1)
        r = run(["tools/scheduler.py", "record", "--agent", "claude", "--id", "exp_001",
                 "--stage", "probe", "--delta", "-0.001"], env)
        check("C13 negative probe is killed", "KILLED" in r.stdout)
        r = run(["tools/practice_lint.py", "--agent", "claude"], env)
        check("lint runs on sandbox", r.returncode in (0, 1))

        # ablation switch
        r = run(["tools/memory_cli.py", "retrieve", "--anchor", "classify"],
                {**env, "KAGGLE_MEMORY_OFF": "1"})
        check("gym ablation switch disables retrieval", "MEMORY OFF" in r.stdout)
        r = run(["tools/brief.py", "generate", "--agent", "claude"],
                {**env, "KAGGLE_MEMORY_OFF": "1"})
        brief = (ws / "BRIEF.md").read_text()
        check("brief honors memory-off ablation",
              r.returncode == 0 and "MEMORY OFF" in brief and "t-card" not in brief)

        # calibration reads a scored prediction
        pred_dir = sb / ".ai" / "memory" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "t.md").write_text(FILLED_PREDICTION)
        r = run(["tools/calibration.py", "report"], env)
        check("C14 calibration joins confidence with outcomes",
              "Accuracy by stated confidence" in r.stdout)

        # stack: blend correctness
        for name, v in (("a.csv", 0.0), ("b.csv", 1.0)):
            (sb / name).write_text("id,pred\n1," + str(v) + "\n2," + str(v) + "\n")
        run(["tools/stack.py", "blend", "--files", str(sb / "a.csv"), str(sb / "b.csv"),
             "--out", str(sb / "o.csv")], env)
        blended = (sb / "o.csv").read_text()
        check("stack blend averages correctly", "1,0.5" in blended)

        # fingerprint
        (sb / "tr.csv").write_text("id,x,y\n" + "\n".join(f"{i},{i%7},{i%2}" for i in range(50)))
        r = run(["tools/fingerprint.py", "compute", "--train", str(sb / "tr.csv"),
                 "--target", "y", "--slug", "t", "--write"], env)
        check("fingerprint computes and writes", r.returncode == 0)

        # linter catches self-promotion
        (ws / "MEMORY_CANDIDATES.md").write_text(
            "| Claim | Scope | Evidence | Counter-evidence | Predicted impact | Status |\n"
            "|---|---|---|---|---|---|\n| x | t | n | n | b | validated |\n")
        r = run(["tools/practice_lint.py", "--agent", "claude"], env)
        check("C8 lint catches self-promoted candidate",
              r.returncode == 1 and "SELF_PROMOTED" in r.stdout)

        # C1: the score gate must track local bests (kaggle scores arrive later)
        snippet = (
            "import sys, tempfile, json; from pathlib import Path;"
            f"sys.path.insert(0, r'{REPO / 'tools'}');"
            "import registry;"
            "registry.REGISTRY_PATH = Path(tempfile.mkdtemp()) / 'reg.json';"
            "reg = registry.create_empty_registry('t');"
            "registry.add_submission(reg, 'claude', 'a.csv', 0.5, 0, [], 'd', 'a', direction='minimize');"
            "assert registry.get_agent_best(reg, 'claude')['local_score'] == 0.5;"
            "registry.add_submission(reg, 'claude', 'b.csv', 0.7, 0, [], 'd', 'a', direction='minimize');"
            "assert registry.get_agent_best(reg, 'claude')['local_score'] == 0.5, 'minimize direction ignored';"
            "registry.add_submission(reg, 'claude', 'c.csv', 0.3, 0, [], 'd', 'a', direction='minimize');"
            "assert registry.get_agent_best(reg, 'claude')['local_score'] == 0.3;"
            "print('OK')"
        )
        r = run(["-c", snippet], env)
        check("C1 registry tracks local best (direction-aware)",
              r.returncode == 0 and "OK" in r.stdout, r.stderr.strip()[-200:])

    print("# 4. Repo hygiene")
    r = run(["tools/practice_lint.py"])
    check("repo lint clean", r.returncode == 0)
    hooks = json.loads((REPO / ".claude" / "settings.json").read_text())
    matchers = [g.get("matcher") for g in hooks.get("hooks", {}).get("PreToolUse", [])]
    check("Claude PreToolUse hooks cover WebFetch and Bash",
          "WebFetch" in matchers and "Bash" in matchers, str(matchers))
    claude_session = hooks.get("hooks", {}).get("SessionStart", [])
    check("Claude SessionStart generates brief",
          "brief.py" in json.dumps(claude_session), str(claude_session))
    codex_hooks = json.loads((REPO / ".codex" / "hooks.json").read_text())
    check("Codex PreToolUse hook wired", "PreToolUse" in codex_hooks.get("hooks", {}))
    check("Codex SessionStart generates brief",
          "brief.py" in json.dumps(codex_hooks.get("hooks", {}).get("SessionStart", [])))
    const = (REPO / ".ai" / "constitution.md").read_text()
    import re
    rules = re.findall(r"^### (C\d+)", const, re.MULTILINE)
    check("constitution has <= 15 rules", 0 < len(rules) <= 15, f"{len(rules)} rules")
    enforced = {"C1": "submit.py", "C2": "writeup.py", "C5": "scheduler.py",
                "C8": "memory_cli.py", "C13": "scheduler.py"}
    missing = [c for c, tool in enforced.items()
               if c in rules and not (REPO / "tools" / tool).exists()]
    check("every hard-gate rule has its tool", not missing, str(missing))

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 60}")
    print(f"Self-check: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed"
          + (f" — {len(failed)} FAILED" if failed else " — system verified"))
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
