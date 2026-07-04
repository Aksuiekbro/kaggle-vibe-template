Initialize workspace for a new Kaggle competition.

Usage: /setup-competition <competition-slug>

Steps:
1. Run `python tools/setup.py --competition $ARGUMENTS`
   - If the data download fails with a rules error, tell the human: "accept the
     competition rules at kaggle.com/c/$ARGUMENTS/rules, then say 'retry'".
     That browser click is the only step an agent cannot do.
2. Fill in COMPETITION.md — use the Kaggle MCP or `kaggle competitions list -s`
   for metadata; at MINIMUM set **Metric** and **Direction**
   (maximize/minimize), because tools/submit.py's score gate reads these:
   - Problem description
   - Evaluation metric and direction
   - Data description
   - Submission format
   - Known constraints
3. Run `python tools/selfcheck.py` — must be green before any agent loops start.
4. Do NOT read discussions or notebooks yet (C2) — but note that PREDICTION.md
   scaffolds are ready in each agent workspace.
5. Report: what was set up, selfcheck result, and what (if anything) needs the
   human — normally only the rules click.
