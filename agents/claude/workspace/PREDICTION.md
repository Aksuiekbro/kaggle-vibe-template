# Prospective Prediction

Competition: missing-fundamental-puzzle
Agent: claude
Created: 2026-07-05T08:11:06.880052+00:00
Status: open

Complete this before reading current competition discussions, public notebooks, or winner-style solution threads. The goal is to pre-register your best playbook prediction so postmortems can measure whether research and memory improve judgment.

## Naive Default Playbook

Write the boring default for this competition type. These predictions are baseline expectations and do not count as informative hits unless the actual prediction makes a useful, specific deviation.

| Category | Default prediction |
|----------|--------------------|
| CV scheme | Stratified K-fold (5-fold) on the target class |
| Feature families | MFCCs, mel-spectrogram, chroma, spectral centroid/rolloff |
| Model families | Gradient boosting (LightGBM/XGBoost) on tabular audio features, or a CNN on mel-spectrogram |
| Ensembling/postprocessing | Average several seeds/folds; simple softmax blend of 2-3 models |
| Leakage/drift/metric traps | Class imbalance biasing accuracy toward majority classes; train/test split leakage via duplicate or near-duplicate clips |

## Actual Prediction

| Category | Prediction | Confidence | Memory/source used | Deviation from default? |
|----------|------------|------------|--------------------|-------------------------|
| CV scheme | StratifiedKFold likely breaks because several of the 82 classes have as few as 3 samples (can't stratify into 5 folds cleanly); expect the winning approach to use fewer folds (3) or grouped/leave-few-out handling for rare classes, plus adversarial validation to rule out train/test drift | 0.55 | adversarial-validation-before-cv (memory) + row counts in COMPETITION.md | yes — default assumes clean 5-fold stratification works out of the box |
| Feature families | Because the fundamental is literally missing, naive autocorrelation/YIN-style F0 estimators will fail outright — winning features are harmonic-spacing / comb-filter / cepstrum-based (inferring F0 from overtone spacing) or a CNN reading log-mel spectrograms directly so the network learns the harmonic template itself | 0.6 | domain reasoning from competition name/description only | yes — default assumes standard MFCC/pitch features transfer, but the whole point of "missing fundamental" is that direct pitch detectors break |
| Model families | With only 2330 train clips across 82 classes (many <10 samples), a from-scratch CNN likely overfits; expect engineered harmonic features + gradient boosting (or a heavily regularized/pretrained CNN) to beat a raw CNN baseline | 0.5 | general small-data intuition | yes — default treats GBM and CNN as equally likely; here dataset size pushes toward GBM/feature-engineering |
| Ensembling/postprocessing | Given severe class imbalance (3-56 samples/class), expect class-weighting or oversampling of rare classes to matter more than blending; naive averaging without addressing imbalance likely underperforms | 0.5 | class distribution in COMPETITION.md | yes — default assumes generic seed-averaging suffices |
| Leakage/drift/metric traps | Suspect clips could be multiple takes/segments from the same underlying recording session (6.6s fixed length, mono, same sample rate) creating near-duplicate train/test leakage if not grouped by source recording | 0.4 | domain reasoning | yes — default only flags generic duplicate-row leakage, this is a specific same-session-audio hypothesis |

## Informative Deviations

List the non-obvious predictions that should be scored most heavily later.

| Claim | Why it differs from default | Expected impact | How to test during competition |
|-------|-----------------------------|-----------------|--------------------------------|
| Direct F0/autocorrelation pitch detectors fail on this data because the fundamental is missing by construction | This is the namesake phenomenon of the competition, not a generic audio-classification default | High — determines whether feature engineering should target harmonics instead of F0 | Compare accuracy of an autocorrelation-based F0 feature vs. harmonic-spacing/cepstrum feature in an ablation |
| Rare classes (3-10 samples) make standard 5-fold stratification infeasible for a meaningful slice of classes | Default competition playbooks assume comfortable per-class counts | Medium-high — wrong CV scheme gives noisy/misleading local scores | Check min per-class count vs. fold count before trusting CV; try 3-fold or grouped scheme and compare variance |
| Near-duplicate/same-session leakage between train and test | Not mentioned in COMPETITION.md but plausible given fixed-length uniform-format clips | Medium — if true, local CV without grouping will overestimate LB score | Run adversarial validation (train vs test classifier) and check audio similarity/hashing across split |

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
