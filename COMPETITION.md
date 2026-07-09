# Competition: hear-me-personalized-music-recommender

Community competition: https://www.kaggle.com/competitions/hear-me-personalized-music-recommender

## Problem

Personalized music recommendation: for each of 1,500 test users, recommend 50
tracks they are likely to listen to next, from implicit-feedback listening logs.

## Data

`shared/data/`:
- `interactions.csv` — 24,798,186 rows: user_id, item_id, listened_duration, listened_datetime (Aug 2025 window)
- `item_metadata.csv` — 348,754 tracks: names, artists, duration, genres (Russian genre labels), like/dislike/download counts
- `user_metadata.csv` — 279,071 users: age_bin, children, gender, top_genre, per-user like/dislike/download counts
- `test.csv` — 1,500 user_ids to recommend for
- `starter-notebook-recsys-v2.ipynb` — gated by C2; do not open before PREDICTION.md is registered

## Evaluation

- **Metric**: MAP@50 (mean average precision over 50 recommendations per user)
- **Direction**: maximize
- **CV variance threshold**: 0.02

## Submission format

See sample in starter notebook context: one row per (user, ranked recommendation), 1,500 users x 50 items, with an id column.

## Benchmarks (do not read the prior attempt before PREDICTION.md is complete)

- Our prior attempt (July 2026, laptop instance): best public/private **0.38583**
- Leaderboard winner: **0.40442**; second place 0.36551 — our prior attempt already exceeds 2nd place
- **Prior-attempt knowledge is at `/root/prior-hearme/`** (strategies, experiment
  journals, progress logs from claude/codex/gemini). This is OUR OWN prior work —
  after your prediction is registered, mine it like a sharing round: what was
  tried (ALS candidates + target-encoded HGB rankers, per-user z-score blends,
  temporal CV Aug1->Aug8 / Aug8->Aug16), what its postmortem gaps were, and what
  the 0.386 -> 0.404 gap likely needs.

## Known constraints

- interactions.csv is 1.16 GB / 24.8M rows: the box has 4 GB RAM + 4 GB swap.
  Load with dtypes (int32/float32), usecols, or chunks; NEVER a naive read_csv.
  Heavy fits go to Kaggle kernels: `python3 tools/kkernel.py run --script ... --competition hear-me-personalized-music-recommender`
- ML venv: `~/ml/bin/python`. implicit/lightfm not installed — pip install into the venv as needed.
- Late-submission mode: real public/private scores return; treat as full-feedback run.
