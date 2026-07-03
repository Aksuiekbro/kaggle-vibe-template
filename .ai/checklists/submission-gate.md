# Pre-Submission Quality Gate

Check every item before calling `tools/submit.py`. The tool enforces some of these automatically, but you should verify manually.

## Required Checks

- [ ] Submission file exists and is valid format (correct columns, correct row count)
- [ ] Local evaluation run: `python tools/evaluate.py --agent <name> --file <path>`
- [ ] Score beats your current best in the registry
- [ ] No overfitting flags raised by evaluate.py (or you can explain why they're false positives)
- [ ] CV variance is below the threshold set in COMPETITION.md
- [ ] This submission represents a meaningfully different approach from your last 3

## Provenance

- [ ] You can describe in one sentence what changed from your previous best
- [ ] If derivative work (based on another agent's submission), provenance is declared
- [ ] If ensemble, all components are documented

## Anti-Overfitting

- [ ] Solution tested on held-out validation data (not just training data)
- [ ] For optimization: tested on multiple random seeds / instances
- [ ] For ML: k-fold CV (k≥5) scores are consistent (low variance)
- [ ] You are NOT just tuning hyperparameters to fit the public LB
