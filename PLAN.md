# Idea Bank & Strategy Decisions

Shared across all agents. Track ideas, what has been tried, and what works.

During active multi-agent runs, do not let multiple agents edit this file concurrently. Agents should draft queue items in their own workspace, then a human or designated coordinator consolidates them here.

## Ideas to Try

| # | Idea | Priority | Status | Agent | Notes |
|---|------|----------|--------|-------|-------|
| 1 | Reproduce prior-best pipeline (Python, since prior C++ code did not survive) | HIGH | in-progress | claude | exp_001 in agents/claude/workspace/EXPERIMENTS.jsonl |
| 2 | Implicit-ALS as a NEW merged candidate lane (not standalone, not a post-hoc feature) | MED | queued | claude | exp_004 |
| 3 | TE priors for genre-combo / artist-popularity-curve / duration-bucket | MED | queued | claude | exp_005 |
| 4 | Full-catalog scoring reranker (no candidate-gen stage) to remove recall bottleneck | MED | queued | claude | exp_007, see public kernel `daurenzhunussov/regression` |
| 5 | Sequential neural recommender (SASRec/GRU4Rec-style) on Kaggle GPU | LOW-MED | queued | claude | exp_002, only untried major model family from prior run |

## What's Been Tried

**IMPORTANT CONTEXT (2026-07-09):** This is our second attempt at this competition. A prior session (uncommitted scratch work, not on this git branch) already ran the full loop and reached public/private **0.38583** (beats 2nd place 0.36551; leaderboard winner is 0.40442). That prior run's docs live at `/root/prior-hearme/` (not code — the C++ implementation did not survive, only design docs/journals). Full mining report is condensed below; see `agents/claude/workspace/PROGRESS.md` (2026-07-09 entry) for the complete version.

| Idea | Agent | Result | Score | Why it worked/failed |
|------|-------|--------|-------|---------------------|
| Repeat/frequency-weighted personal history (recency+quality weighted) | codex (prior run) | worked | 0.335 | Beats "newest-unique" ranking; recency/quality weighting transfers |
| Item-transition reranking, no reserved slot | codex (prior run) | worked | 0.358 | Reserving a dedicated slot for transitions was worse than letting them compete for all 50 |
| Demographic popularity pools (top_genre/gender/age_bin) | codex (prior run) | worked | 0.359 | Genuinely new candidate source |
| Listen-quality weighting (down-weight low listened-ratio/short plays) | codex (prior run) | worked | 0.362 | Real transfer; metric rewards listened-fraction, not just hit |
| Split-collaborator-artist candidate lane (multi-artist string splitting) | codex (prior run) | worked | 0.362 | Another genuinely new signal |
| Component-score ML reranker (HGB) on top of C++ candidate engine | codex (prior run) | worked | 0.378 | Biggest jump: heuristic scores as features, learned reranker |
| Strict OOF target encoding (item/artist/genre priors) | codex (prior run) | worked | 0.381 | Single most important feature-family addition post-ML-transition |
| Title / artist-title OOF target encoding | codex (prior run) | worked | 0.386 (final: 0.38583) | Titles useless as a candidate SOURCE, but strong as a TE PRIOR — last successful new signal found |
| Standalone implicit ALS | codex (prior run) | failed | ~0.108 | Never integrated as a merged candidate lane, only tried standalone/as a late feature |
| Co-visitation (any form: primary ranker, supplemental lane, ML feature) | codex (prior run) | failed | n/a | Raises oracle recall but never rankable by the TE reranker; killed 5+ times |
| Increasing reranker tree capacity / ExtraTrees ensembling post-TE | codex (prior run) | failed | n/a | Severe local-CV-vs-official divergence (fold-mean-vs-official correlation ≈ −0.72) |
| Diversity/seen-item caps applied broadly (not just <50-history users) | codex (prior run) | failed | n/a | Metric rewards exact repeats; caps hurt 50-199-history band |
| Forcing unseen recommendation slots (seen_limit<50) for all users | codex (prior run) | failed | n/a | Only the <50-history-only segment benefits |

## Key Decisions
### Coordinator note — 2026-07-09 (use proven public recommenders, stop hand-rolling)

Mature open-source recsys implementations exist for exactly this data shape
(implicit feedback + metadata). Use them as CANDIDATE GENERATORS and scorers
feeding the existing ranker. Cheap-probe order (measure candidate recall@500
and direct MAP@50 on trusted CV for each):

1. `implicit` library (pip): ItemKNN(cosine/bm25) and ALS/BPR properly tuned —
   CPU-fast on sparse matrices.
2. EASE (closed-form linear, ~15 lines numpy): restrict to top-30-50k items by
   interaction count. Known to beat deep models on implicit feedback.
3. SAR-style co-occurrence with TIME DECAY (Microsoft Recommenders algorithm —
   reimplement the formula directly if the package is heavy). NOTE: this is the
   2-hour-winner-shaped algorithm; its score is also evidence about the winner.
4. Sequence models (SASRec via RecBole) ONLY after 1-3 are measured, on Kaggle
   GPU kernels (tools/kkernel.py --gpu), not locally.

Each generator = one experiment row with recall + MAP deltas. Blend candidates
from decorrelated sources before ranker; that is where the oracle gap
(0.569 -> 0.592+) actually closes.

### Coordinator note — 2026-07-09 (critical context: the winner had a 2-HOUR limit)

The original competition was a 2-hour sprint. The 0.40442 winner built something
SIMPLE and correct, not a deep pipeline. Before any further complex work:

1. Measure the exact MAP@50 (trusted temporal CV) of each 2-hour-plausible baseline:
   a. Pure repeat-listening: user's own tracks ranked by recency / frequency /
      listened_duration share. (Check FIRST whether re-listens are scoreable
      targets in the hidden window — if yes, this alone may explain 0.40.)
   b. Repeat-artist: unheard tracks by the user's top artists.
   c. Global/genre popularity in the last N days, personalized by user top_genre.
   d. Simple co-listen counts from the last week only.
2. Report the baseline table honestly. If any single baseline or trivial
   combination approaches 0.38+, the winning recipe is baseline + light ranking,
   and our complex ranker should be REBUILT AROUND it, not vice versa.
3. Implication of the time limit: 0.404 is a sprint record, not the signal
   ceiling. With our compute, target ABOVE 0.404, not at it.


- **2026-07-09 (claude):** Treat the prior 0.38583 run as a reproducible recipe (see PROGRESS.md), not a starting file — no code survived. First block's P0 is a Python reproduction (exp_001), since the original was C++ and never committed. Confirmed via `verifiers.py columns` that random interaction-row KFold is invalid here (user_id drift score 0.151, matches prior team's own adversarial-validation finding); must use temporal rolling folds (Aug1/Aug8/Aug16 cutoffs, 15-day forward window).
- **2026-07-09 (claude):** Chose to attack the "candidate recall is the ceiling, not reranker capacity" gap (the prior run's own postmortem conclusion) via two independent routes: (a) ALS as a genuinely new merged candidate lane (exp_004), (b) full-catalog scoring to remove the candidate-gen stage entirely (exp_007) — rather than repeating any reranker-capacity tuning, which is a confirmed dead end.

## Research Findings

- Competition is a course-cohort recsys exercise (team names are `City_Surname_Name`, e.g. Moscow/Almaty/Gyumri) — no winner writeups or discussion threads exist, and none appeared in the ~8 months since prior attempt checked. Re-confirmed 2026-07-09.
- Two public starter kernels exist (0 votes, likely low-scoring): `daurenzhunussov/problem1solution` (trivial ALS factors=3/iter=1 + submission-format validator) and `daurenzhunussov/regression` (full-catalog CatBoost regressor, GPU, scores all 348,754 items x 1,500 users, genre BoW via CatBoost text_processing, target = listened_percentage). Neither author's leaderboard score is identifiable. The full-catalog approach is structurally interesting (no candidate-recall bottleneck) — queued as exp_007.
- Dataset fingerprint computed 2026-07-09 (`.ai/memory/competitions/hear-me-personalized-music-recommender-fingerprint.json`): no comparable neighbors in the fingerprint store yet (first competition of this shape recorded).

## Kaggle Upsolve Queue

Use this table for winner writeups, tutorials, and similar-competition research. Each source must become an experiment or an explicit rejection.

Keep at most 3 active research-derived experiments in progress at once.

| Source | Similarity | Pattern claim | Mechanism hypothesis | Experiment | Predicted impact | Cost | Validation plan | Risk | Owner | Status | Result |
|--------|------------|---------------|----------------------|------------|------------------|------|-----------------|------|-------|--------|--------|

## Prospective Pattern Transfer Audits

Before reading current competition discussions or public notebooks, write a dated prediction. Score it after the competition closes and winner writeups appear.

| Competition | Date | Prediction file | Model cutoff | Naive default | Memory cards used | Raw hits | Informative hits | Misses | Scored date | Playbook update |
|-------------|------|-----------------|--------------|---------------|-------------------|----------|------------------|--------|-------------|-----------------|

## Postmortems

After private leaderboard reveal or reliable winner writeups, compare our final approach against top winners and write memory-card candidates.

| Competition | Agent | Final rank/score | Winner convergence | Main gap | Memory cards created | Follow-up |
|-------------|-------|------------------|--------------------|----------|----------------------|-----------|
