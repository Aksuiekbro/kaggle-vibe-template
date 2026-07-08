# Solution Review

Reviewer: gemini
Subject: claude's submission sub_004
Score: 8.5 / 10
Status: SOLID

## Approach Summary
Claude used `librosa` to extract a wide range of spectral and chroma features: MFCC (mean, std), Chroma STFT, Chroma CQT, spectral contrast, spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, zero-crossing rate, and Tonnetz. They also added a cepstral-domain spectral autocorrelation feature (correlating the log-magnitude spectrum) to target the missing fundamental harmonic structure. The model was trained using LightGBM with 3-fold cross-validation, achieving an OOF accuracy of 0.5681 and a public LB accuracy of 0.55172.

## Strengths
- Uses high-signal pitch-related features like Chroma CQT and Tonnetz, which are well-suited for musical pitch representation.
- Employs cepstral-domain spectral autocorrelation (correlating log-magnitude STFT), which is robust for capturing harmonic spacing (comb-like filters) for missing fundamentals compared to raw time-domain autocorrelation.
- Offloaded the full feature extraction and training to Kaggle compute via `kkernel.py`, bypassing local resource constraints.

## Improvement Ideas
- **Hyperparameter Optimization**: The LightGBM model used manual hyperparameters (`learning_rate=0.05, num_leaves=31, min_child_samples=3, subsample=0.8, colsample_bytree=0.8`). Hyperparameter tuning via Optuna (like in Gemini's `exp_002`) should yield a significant boost.
- **5-Fold Cross-Validation**: Moving from 3-fold to 5-fold CV will reduce validation variance and provide more robust out-of-fold predictions.
- **Feature Fusion**: Gemini's time-domain Autocorrelation (ACF) lags and Harmonic Product Spectrum (HPS) features can be combined with Claude's librosa-extracted features to capture both time-domain and frequency-domain pitch cues.

## Risks
- **High Fold Variance**: The CV standard deviation across folds was high (std = 0.2067), which suggests instability in evaluation, likely caused by severe class imbalance (some classes have as few as 3 samples) in a 3-fold split.
- **Distribution Shift**: The adversarial validation AUC of 0.55-0.56 suggests mild shift between train and test.

## Could Combine With
- Combine Claude's features (MFCC, Chroma, Spectral Contrast, Tonnetz, Cepstral ACF) with Gemini's HPS and time-domain ACF features.
- Train LightGBM/CatBoost on the combined features with Optuna tuning.
