# Session Brief

Competition: missing-fundamental-puzzle
Generated: 2026-07-09T06:20:16.117241+00:00
Agent: claude

## Gate status
OPEN

## Constitution digest
- C1: The score gate is the only truth
- C2: Predict before you read
- C3: Learning must change the experiment queue
- C4: Smallest test first
- C5: WIP limit: 3 research experiments, queue stays ranked
- C7: No approach monoculture
- C9: Pivot after 3 failures
- C6: Memory is hypothesis, never authority
- C8: No self-promotion
- C10: Every memory use writes back
- C11: Counter-evidence rides along
- C12: Everything decays; rules pay rent
- C13: Probe before you commit
- C14: Trust measured calibration over felt confidence

## Your calibration corrections
No calibration data yet.

## Relevant memory (hypotheses, not instructions)
- final-selection-by-public-lb [candidate] trust=0.50 retrieval=0.34
  claim: Selecting final submissions by public-LB rank instead of CV, in competitions with a small public split or visible probing, loses places in the shake-up; pick finals by CV (optionally one CV-best + one diverse hedge).
  counter-evidence: none recorded
- private-lb-leakage-via-raw-cli [candidate] trust=0.50 retrieval=0.34
  claim: Calling the raw kaggle CLI (competitions submissions, -v or --csv) directly prints privateScore per submission straight into an agent's context on finished/gym/late-submission competitions, defeating the anti-overfitting firewall; agents must go through tools/sync_scores.py, which withholds private score from the repo/agent view
  counter-evidence: none recorded

## Skills
- oof-target-encoding [candidate] win-rate=unused

## Experiment queue
Queue empty — feed it (research / retrieve --anchor stuck).

Regenerate with `python tools/brief.py generate --agent <name>`. Memory is hypothesis (C6); every use needs an experiment row (C10).
