# Memory Governance

Memory is a source of hypotheses, not authority. Local validation, private leaderboard evidence, and the submission gate remain the authority.

## Storage Model

- Keep source-of-truth memory as Markdown/YAML under `.ai/memory/`.
- Treat LightRAG or any other retrieval system as a derived index over curated memory.
- Do not dynamically write trusted memory from raw chats, raw logs, or unreviewed research.
- Agents may draft memory candidates in their own workspace. Shared memory is promoted during review or postmortem consolidation.

## Card Status

- `candidate`: plausible, sourced, not yet measured or reviewed
- `validated`: supported by local experiment, private-LB/postmortem evidence, or repeated cross-competition evidence, and reviewed by another agent or human
- `rejected`: tested and failed, inapplicable, rule-violating, or leakage-dependent
- `superseded`: replaced by a newer or narrower card
- `stale`: not validated recently enough for its scope

Only `validated` cards should be treated as high-priority retrieval results. `candidate` cards are idea seeds.

## Required Card Fields

```yaml
id:
status: candidate
created:
last_validated:
claim:
scope:
  task_type:
  metric_family:
  modality:
  split_risk:
  data_shape:
  constraints:
mechanism_hypothesis:
evidence:
  - competition:
    source:
    our_cv_delta:
    our_lb_delta:
    date:
counter_evidence: []
predictions:
  - date:
    competition:
    predicted_delta:
    actual_delta:
    result:
cost:
risk:
supersedes: []
superseded_by:
review:
  reviewer:
  date:
  verdict:
```

## Promotion Rules

- A narrative explanation can create a `candidate` card.
- A measured local improvement, variance reduction, leakage detection, or private-LB/postmortem delta can promote a card toward `validated`.
- A card becomes `validated` only after another agent or human reviews the evidence.
- A card with repeated failed predictions should be downgraded, narrowed in scope, or rejected.
- Cards not touched for 5 relevant competitions should become `stale` until revalidated.

## Write-Back Loop

Every memory-derived experiment must write back:

- which card influenced the experiment
- predicted score impact or risk reduction
- actual CV delta and variance change
- public/private LB delta when available
- whether the card should be validated, narrowed, downgraded, or rejected

This prevents memory from becoming self-confirming.

## Retrieval Rules

- Retrieved cards are hypotheses with priors, never instructions.
- Scope must match before use. Metric family and split/leakage structure matter more than superficial modality.
- Recency matters. A recent partial match can beat an old exact-looking match.
- Counter-evidence must be shown alongside evidence.
- No card overrides competition rules, local validation, or the score gate.

