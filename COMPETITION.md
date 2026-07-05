# Competition: missing-fundamental-puzzle

Community competition: https://www.kaggle.com/competitions/missing-fundamental-puzzle

## Problem

Predict the perceived pitch class (`Pitch_ID`) of short audio clips. The
competition name refers to the missing-fundamental phenomenon: the fundamental
frequency may be absent from the signal and must be inferred (e.g. from
harmonic structure).

## Data

- `shared/data/kaggle_dataset-20251026T143755Z-1-001/kaggle_dataset/`
- `train/` — 2330 mono WAVs, 44100 Hz, ~6.6 s each; labels in `train.csv` (Pitch_ID, Path)
- `test/` — 583 WAVs; `test.csv` lists paths
- 82 classes, imbalanced (3 to 56 samples per class)
- `starter-notebook-v3-missing-puzzle.ipynb` at data root — do NOT open before PREDICTION.md is complete (C2)

## Evaluation

- **Metric**: accuracy (inferred from leaderboard scores in [0,1]; top ~0.983) — verify on the overview page after predictions are registered
- **Direction**: maximize
- **CV variance threshold**: 0.02

## Submission format

CSV with header `Path,Pitch_ID` — one row per test WAV (see sample_submission.csv), 583 rows.

## Known constraints

- Leaderboard submissions are dated 2025-10; the competition may already be
  closed (late submission mode). Treat as a gym-grade run either way: full
  pipeline, honest scoring.
- ML Python lives in the venv: `~/ml/bin/python` (numpy/pandas/sklearn/LightGBM/XGBoost/Optuna, scipy via sklearn). No librosa yet — `~/ml/bin/pip install` as needed, or use scipy.signal + wave.
- Heavy training can ship to Kaggle compute: `python3 tools/kkernel.py run --script ... --competition missing-fundamental-puzzle`
