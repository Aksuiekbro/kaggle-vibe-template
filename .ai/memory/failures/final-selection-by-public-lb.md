# Failure Card

```yaml
id: final-selection-by-public-lb
status: candidate
created: 2026-07-04
last_validated:
claim: Selecting final submissions by public-LB rank instead of CV, in competitions with a small public split or visible probing, loses places in the shake-up; pick finals by CV (optionally one CV-best + one diverse hedge).
scope:
  task_type: any
  metric_family: any
  modality: any
  split_risk: adversarial
  data_shape: small public split
  constraints: two final submissions
mechanism_hypothesis: the public split is a small sample; ranking by it selects for noise fit, which reverts on the larger private split.
evidence: []
counter_evidence: []
predictions: []
cost: zero — it is a selection policy, not a model change
risk: with a very large public split and stable folds, public LB can be informative; check split sizes first
supersedes: []
superseded_by:
review:
  reviewer:
  date:
  verdict:
```

## Notes

Seeded at template build (2026-07-04). This is the classic shake-up failure mode;
it lives in `failures/` so `retrieve --anchor pre-submit` always surfaces it before
final selection. Earns status only via our own postmortem write-backs.
