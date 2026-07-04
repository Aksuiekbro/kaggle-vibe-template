# Kaggle Upsolve Protocol

Use this checklist when researching a competition, when stuck, or when moving into a new competition type. The goal is to treat past Kaggle winners like Codeforces editorials: learn the reusable pattern, re-derive why it might work, then validate it here.

## Core Loop

- [ ] Classify the current competition: task type, metric, modality, data size, split risk, leakage risk, and likely model family.
- [ ] Complete `agents/<name>/workspace/PREDICTION.md` before reading current competition discussions, public notebooks, or solution-style writeups.
- [ ] Find similar competitions before generic tutorials. Prefer same task type, same metric, same modality, and similar train/test structure.
- [ ] Read top winner writeups and high-signal discussions for patterns, not summaries. Prefer top 3 writeups when available.
- [ ] Convert each useful idea into an experiment row before implementation.
- [ ] Rank experiment rows by expected impact, cost, and risk before running them.
- [ ] Run the smallest local experiment that can validate or reject the idea.
- [ ] Promote only ideas with evidence into `STRATEGY.md`.

## Source Priority

1. Current competition discussions and metric clarifications
2. Winning solutions from similar past Kaggle competitions
3. High-scoring public notebooks for this competition
4. Official docs for the model, metric, or library being used
5. Academic papers and technical blogs
6. Codeforces / competitive-programming material for optimization, testing, randomized search, and implementation discipline

## Similarity Filters

Measure similarity before judging it: `python tools/fingerprint.py compute --train
<t.csv> --test <te.csv> --target <y> --slug <comp> --write`, then `fingerprint.py
compare --slug <comp>` ranks past competitions by dataset-statistics distance.
Record the distance in the Similarity column; the filters below explain *why* a
measured neighbor is or is not comparable:

- Task: regression, binary classification, multiclass, ranking, forecasting, CV, NLP, optimization
- Metric: RMSE, MAE, AUC, logloss, F1, MAP@K, custom metric
- Modality: tabular, text, image, time series, graph, geospatial, mixed
- Split: random, grouped, temporal, entity-based, hidden-drift, adversarial
- Data shape: row count, feature count, class balance, missingness, target distribution
- Constraint: compute limit, submission limit, external-data policy, leakage risk

## Experiment Row Format

Every learning source must produce one or more rows like this:

| Source | Similarity | Pattern claim | Mechanism hypothesis | Transfer experiment | Predicted impact | Cost | Validation plan | Risk | Result |
|--------|------------|---------------|----------------------|---------------------|------------------|------|-----------------|------|--------|

If no experiment is worth running, write the rejection reason in the `Result` column.

Keep at most 3 active research-derived experiments in progress at once.

## Winner Writeup Count

- Minimum: top 1 writeup for each similar competition
- Preferred: top 3 writeups for each similar competition
- Deep mode: top 5 only for highly similar, well-documented competitions

Record a convergence verdict:

- `CONSENSUS`: top approaches share the same load-bearing pattern
- `MIXED`: several different routes worked
- `OUTLIER`: the winner used a rare trick, leak-like edge, or unusually costly route

## Prospective Pattern Transfer Audit

Use this audit to measure whether memory and research improve future judgment.

1. Before reading current competition discussions, public notebooks, or solution-style writeups, write a dated prediction file in your workspace.
2. Predict:
   - expected CV scheme
   - likely feature families
   - likely model families
   - likely ensemble or postprocessing
   - likely leakage, drift, or metric traps
3. Record the naive default playbook for the competition type, then mark which predictions are meaningful deviations from that default.
4. Label every prediction with confidence and which memory cards or sources influenced it.
5. During sharing-round consolidation, copy or index open predictions in `.ai/memory/predictions/INDEX.md`.
6. After the competition closes and winner writeups appear, score the prediction against reality.
7. Write the predicted-vs-actual result back to the relevant memory cards.

Retrospective audits are allowed only for competitions after the agent model's knowledge cutoff. Record the cutoff date in the audit row. Randomly splitting old famous competitions into 7 learn / 5 test is not a valid audit for LLM agents.

## Prediction Scoring

Score each prediction category separately:

- CV scheme
- Feature families
- Model families
- Ensembling/postprocessing
- Leakage, drift, or metric traps

Use these labels:

- `HIT`: the predicted load-bearing pattern matches what top winners actually relied on
- `PARTIAL`: the prediction was directionally right but missed an important condition, scope, or implementation detail
- `MISS`: the prediction did not match the winner evidence or was too generic to be useful

Also compute a baseline-adjusted score:

- Write the naive default playbook before making the actual prediction.
- Generic defaults such as "GBM ensemble + careful CV" do not count as informative hits for tabular competitions unless the prediction added a specific, useful deviation.
- Score informative deviations separately from raw category hits.
- Memory cards get credit only when they influenced a prediction that beat the naive default.

## Alternating Learning Cycles

Alternate between broad pattern learning and narrow implementation:

1. `LEARN`: read a small batch of similar winners or tutorials.
2. `EXTRACT`: write reusable patterns and failure conditions.
3. `TEST`: run the smallest local experiment.
4. `REFINE`: update `STRATEGY.md` and the experiment queue.
5. `SHIP`: submit only if the score gate passes.

Do not spend more than 60 minutes consuming material before running at least one experiment, unless the human explicitly asks for a research-only pass.

## Promotion Rules

- A pattern may enter `STRATEGY.md` as validated only if it improved validation, reduced variance, exposed leakage, or produced measured postmortem evidence.
- A pattern that only explains a failure or ranks an experiment remains a candidate until tested.
- A pattern may enter `PLAN.md` if it is promising but untested.
- A pattern must be rejected if it only improves public LB, depends on leakage, violates competition rules, or cannot be validated locally.
