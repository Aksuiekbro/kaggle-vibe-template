# Pattern Card

```yaml
id: adversarial-validation-before-cv
status: candidate
created: 2026-07-04
last_validated:
claim: Before trusting random KFold, train a classifier to distinguish train from test rows; AUC > 0.6 means distribution shift and the CV scheme must change (temporal/grouped/importance-weighted) or local CV will be optimistic.
scope:
  task_type: tabular
  metric_family: any
  modality: tabular
  split_risk: hidden-drift
  data_shape: any
  constraints:
mechanism_hypothesis: train/test shift makes random-fold validation sample a different distribution than the leaderboard scores; adversarial AUC is a cheap detector for that mismatch.
evidence: []
counter_evidence: []
predictions: []
cost: low — one LightGBM/logistic fit on train-vs-test labels
risk: near-0.5 AUC does not prove safety for temporal targets; still check time structure
supersedes: []
superseded_by:
review:
  reviewer:
  date:
  verdict:
```

## Notes

Seeded at template build (2026-07-04) from standard Kaggle practice; deliberately
`candidate` with empty evidence — it earns trust only through write-backs on our
own competitions. Test: run adversarial validation in the first EDA hour; record
the AUC and whether acting on it changed CV-vs-LB agreement.
