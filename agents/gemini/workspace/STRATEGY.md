# Gemini Strategy

## Current Approach
- Features: Hybrid approach ensembling models trained on (1) Full features (1628 RMS-normalized audio features) and (2) Grid-only features (1134 pitch-related grid features + 2 robust pitch estimations, completely excluding noise/loudness/quality metrics to eliminate covariate shift).
- Model: 9-model super ensemble blend optimizing weights of 6 baseline models and 3 grid-only models (PCA-MLP, PCA-SVM, LightGBM) using COBYLA optimizer under 5-Fold Cross Validation. The grid-only models receive a combined weight of 0.39, adding robustness against distribution shift.

## What I've Tried
- LightGBM baseline with manual features.
- LightGBM hyperparameter tuning with Optuna (`exp_002`).
- Out-of-fold ensembling: Blending LightGBM trained on baseline features and LightGBM trained on Combined-Clean features boosted CV Accuracy to **0.7618 (std=0.0175)**.
- MLP Classifier on combined clean features (`exp_006`) with standard scaling, Adam solver, learning rate 0.002, and `(256, 128)` architecture. This achieved **0.8695 (std=0.0056)** accuracy (submitted as `sub_007`).
- Blended MLP and LightGBM Classifier on clean features (`sub_008`), yielding **0.8811 (std=0.0082)**.
- Multi-seed (5 seeds) MLP Classifier ensemble ensembled with LightGBM (`sub_009`), achieving CV accuracy of **0.91245 (std=0.0061)**.
- Blended Multi-seed MLP and LGBM on concatenated exp003 features and high-resolution dense pitch grid features, achieving a personal best CV accuracy of **0.92232 (std=0.0159)** (submitted as `sub_010`).
- Blended Multi-seed MLP and LGBM on concatenated exp003 features and enhanced grid features (810 features), achieving CV accuracy of **0.93262 (std=0.0185)** (submitted as `sub_011`).
- Blended Multi-Seed MLP, Linear SVM (C=0.05), and LightGBM on the concatenated features, achieving a CV accuracy of **0.93476 (std: 0.0157)** (submitted as `sub_012`).
- Multi-seed (5 seeds) MLP Classifier with PCA=512 dimensionality reduction, achieving a new personal best CV accuracy of **0.95451 (std: 0.0090)** (submitted as `sub_014`).
- Trained Linear/RBF SVMs on PCA-512 features. Blended all 5 models (PCA-MLP, PCA-SVM, Raw MLP, Raw LGBM, and Raw SVM) using COBYLA optimizer to search for optimal weights, reaching a new personal best CV accuracy of **0.95579 (std: 0.0117)** (submitted as `sub_015`).
- Extracted RMS-normalized features (FFT, HPS, normalized ACF) and new Cepstrum quefrency grid features (1628 features total) to resolve loudness/gain mismatch. Trained and blended PCA-MLP, PCA-SVM, Raw MLP, and LightGBM. Reached a new personal best CV accuracy of **0.96524 (std: 0.0068)** (submitted as `sub_017`).
- Trained and blended additional MLP architectures (PCA-MLP (256, 256, 128) and Raw-MLP (512, 256, 128)) under 5-Fold CV on V1 normalized features to add diversity. The 6-model optimized ensemble blend boosted CV Accuracy to a new best of **0.96824 (std: 0.0085)** (submitted as `sub_018`).

## What Doesn't Work
- Repeating boolean array masks (e.g., `(freqs >= low_f) & (freqs < high_f)`) over 1,120 loops per audio file was extremely slow.
- Local hyperparameter tuning using Optuna for multiclass GBDTs (LightGBM and XGBoost) with 82 classes is too slow locally because they train 82 trees per iteration per fold.
- XGBoost training/tuning: extremely slow on CPU for 82 classes multiclass classification, and achieves much lower accuracy (~0.6270) compared to LightGBM and linear SVM.
- RandomForest and ExtraTrees classifiers: achieve lower accuracy (~0.6356 and ~0.6210) compared to MLP and SVM.
- Deep neural networks without regularization or with too high of a learning rate (e.g. Config 4 / Config 0) got lower performance.
- Stacking meta-classifiers (Logistic Regression, MLP meta-classifiers) on collinear base model probability predictions: does not beat a simple weighted probability blend due to high class-count overfitting.
- RBF Kernel SVMs on high-dimensional features: yield poor performance (~0.85 CV) compared to Linear SVMs (~0.925 CV).
- Adding raw/log-transformed Librosa CQT, Chroma, MFCC, Tonnetz, and Contrast features (exp_011): actually degraded the single PCA-MLP Fold 0 accuracy from 0.95494 to 0.94635. High dimensional collinearity of these features caused overfitting.
- Adversarially-weighted CV: calculating weights via `w = p / (1 - p)` from adversarial validation probabilities yields extremely high variance (Std=5.6, Max=300) which makes the weighted evaluation extremely noisy and unreliable as a proxy.
- Pitch quality features (such as `yin_std` and `diff_midi` representing tracker disagreement) suffer from severe covariate shift (adversarial AUC of 0.9743) and cause feature-induced overfitting when included.

## Next Experiments
1. Evaluate whether training a 1D/2D CNN on Mel-spectrograms or CQT spectrograms can add further diversity.
2. Build more specialized classifiers trained only on harmonic-evidence features (similar to Claude's sub_020) to continue shifting the ensemble weight toward robust pitch features.
