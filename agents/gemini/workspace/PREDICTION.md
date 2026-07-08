# Prospective Prediction

Competition: missing-fundamental-puzzle
Agent: gemini
Created: 2026-07-05T08:11:06.881332+00:00
Status: open

Complete this before reading current competition discussions, public notebooks, or winner-style solution threads. The goal is to pre-register your best playbook prediction so postmortems can measure whether research and memory improve judgment.

## Naive Default Playbook

Write the boring default for this competition type. These predictions are baseline expectations and do not count as informative hits unless the actual prediction makes a useful, specific deviation.

| Category | Default prediction |
|----------|--------------------|
| CV scheme | Stratified 5-Fold |
| Feature families | MFCCs, Spectral Centroid, Zero Crossing Rate |
| Model families | XGBoost / LightGBM |
| Ensembling/postprocessing | Simple average of class probabilities |
| Leakage/drift/metric traps | Class imbalance causing stratifying issues or empty folds |

## Actual Prediction

| Category | Prediction | Confidence | Memory/source used | Deviation from default? |
|----------|------------|------------|--------------------|-------------------------|
| CV scheme | Stratified 5-Fold with class grouping fallback for classes < 5 samples | 85% | standard ML practices | Yes, fallback for rare classes |
| Feature families | Mel-spectrogram, Chroma (STFT/CQT), Harmonic Product Spectrum (HPS) for missing fundamental, and Autocorrelation | 90% | signal processing theory | Yes, adding HPS and CQT |
| Model families | LightGBM / CatBoost on aggregated spectral features, plus simple 1D/2D CNN on Mel-spectrograms / CQT | 80% | standard audio classification | Yes, hybrid GBDT + CNN approach |
| Ensembling/postprocessing | Weighted probability ensemble of GBDT and CNN models, optimized via out-of-fold predictions | 75% | Kaggle ensembling | Yes, GBDT + CNN combination |
| Leakage/drift/metric traps | Small test set (583) and high imbalance causing public/private LB variance | 80% | competition specs | Yes, focusing on robust CV |

## Informative Deviations

List the non-obvious predictions that should be scored most heavily later.

| Claim | Why it differs from default | Expected impact | How to test during competition |
|-------|-----------------------------|-----------------|--------------------------------|
| Harmonic Product Spectrum (HPS) features will improve performance on missing fundamental cases | HPS aligns harmonics to isolate the fundamental frequency even when absent | ~0.05-0.10 accuracy boost | Compare validation performance with and without HPS features |
| Constant-Q Transform (CQT) features will outperform Mel-spectrogram features | CQT is log-spaced and matches pitch frequencies better than linear STFT/Mel | ~0.02-0.05 accuracy boost | Compare validation performance of CQT vs Mel-Spectrogram features |

## Scoring After Close

Do not fill this until private leaderboard results or reliable winner writeups are available.

| Category | HIT/PARTIAL/MISS | Winner evidence | Notes |
|----------|------------------|-----------------|-------|
| CV scheme | | | |
| Feature families | | | |
| Model families | | | |
| Ensembling/postprocessing | | | |
| Leakage/drift/metric traps | | | |

Baseline-adjusted score:
Memory cards to update:
