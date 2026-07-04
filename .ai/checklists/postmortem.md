# Competition Postmortem Checklist

Run this after a competition closes, after private leaderboard results are available, or after reliable winner writeups appear. This is the highest-signal learning loop in the template because it compares what the agents actually did against what won.

## Inputs

- Final private leaderboard score and rank/percentile, if available
- Final `STRATEGY.md`, `PROGRESS.md`, `RESEARCH.md`, and submission provenance for each agent
- Winning solution writeups, preferably top 3 when available
- Current competition discussions that explain metric quirks, leakage, or shake-up behavior

## Steps

1. Freeze the final agent state before reading winner writeups:
   - Preferred: create a git tag or commit for the final state before postmortem work begins.
   - Fallback: copy each relevant agent workspace into `.ai/runs/<date>/postmortem-freeze/<agent>/`.
   - Do not edit the freeze copy.
2. Record the final approach:
   - validation scheme
   - feature families
   - model families
   - ensembling/postprocessing
   - known risks and failed experiments
3. Read top winner writeups and note convergence:
   - `CONSENSUS`: top approaches share the same load-bearing pattern
   - `MIXED`: several different routes worked
   - `OUTLIER`: winner used a rare trick or leak-like edge
4. Diagnose the gap:
   - wrong validation split
   - missed leakage/drift signal
   - missed feature family
   - wrong model family
   - weak ensemble/postprocessing
   - compute/time allocation failure
   - public-LB overfit
   - implementation bug or experiment never run
5. Convert each gap into a memory-card candidate or explicit rejection.
6. Record predicted-vs-actual impact when a prior memory card influenced the work:
   `python tools/memory_cli.py retrieve --anchor postmortem` lists cards awaiting
   write-back; close each with `python tools/memory_cli.py writeback --card <id>
   --competition <slug> --result hit|partial|miss --actual-delta <x>`.
   Log skill outcomes: `python tools/skills.py log-use --skill <id> --competition <slug> --outcome win|neutral|loss`.
7. For each card that SHOULD have influenced this competition but was never retrieved, run `python tools/memory_cli.py log-miss --card <id> --competition <slug> --reason "<why it should have surfaced>"`.
8. Score any open prediction from `.ai/memory/predictions/INDEX.md` per category.
   Only deviations from the naive default playbook count as informative hits.
9. Assign follow-up experiments for future similar competitions.

## Output

Write a postmortem to `agents/<name>/workspace/POSTMORTEM.md` during active agent work. During human-led consolidation, promote durable findings into `.ai/memory/` using the memory governance checklist.

## Required Sections

```md
# Competition Postmortem

Competition:
Date:
Agent:
Final private score/rank:

## Final Approach

## Winner Convergence

## Gap Diagnosis

## What We Should Have Tried

## Memory Card Candidates

## Prior Memory Scorecard

## Rules or Templates To Update
```
