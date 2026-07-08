# Solution Review

Reviewer: claude
Subject: gemini's submission sub_018
Score: 8.0 / 10
Status: SOLID

## Approach Summary
`agents/gemini/workspace/extract_all_features_norm.py` RMS-normalizes each clip to
target RMS 0.1 (`:24-32`), then builds 1628 features: 120 log-frequency FFT-energy
bins, 200 harmonic-product-spectrum candidate-frequency scores, 174 downsampled ACF
lags, plus a dense 81-note MIDI pitch-candidate grid (`midi_grid = np.arange(30,111)`,
`:119`) with per-candidate harmonic/ACF/cepstrum-lag features (`:127-207`).
`train_norm_models_v3.py` runs 5-fold CV, per fold fitting `StandardScaler`+`PCA(512)`
on train only (`:71-79`, no leakage — transform-only on val/test), training two
PCA-MLPs, two raw-MLPs, and a linear SVM, then blends all six model families
(including baseline LGBM) with COBYLA-optimized weights (`results_norm_blend_v3.json`),
reaching CV 0.96824 (std 0.0085).

## Strengths
- The MIDI-grid harmonic/ACF/cepstrum features (`:119-217`) are the strongest
  domain fit I've seen either agent build for this task: instead of asking "what
  pitch is present," each of 81 candidate fundamentals gets its own evidence score
  from harmonic energy + autocorrelation + cepstrum lag — exactly the right
  representation for a missing-fundamental task where the literal f0 bin is absent.
  `pitch_yin_mapping.csv` backs this up: several `Pitch_ID` classes have
  `std_midi` of 8-16+ semitones (e.g. `Pitch_ID=1`: median_midi 57.4, std_midi
  16.29) — plain YIN pitch detection on the raw signal is unstable/wrong for
  these clips, which is consistent with the fundamental genuinely being absent
  and a naive detector locking onto the wrong partial. A grid-scored,
  harmonic-evidence approach sidesteps that instability.
- PCA is correctly fit per-fold on training data only (`train_norm_models_v3.py:77`)
  — no transductive leakage into CV, despite touching test data for the final
  `X_te_pca` transform in the same block (transform, not fit).
- RMS normalization directly targets the exact shift claude's `exp_003` adversarial
  validation flagged (train-vs-test AUC 0.9984, driven by contrast/zcr/loudness-like
  features) — gemini independently converged on the same mitigation
  (`STRATEGY.md`: "completely eliminates the 2.2x gain/loudness covariate shift").

## Improvement Ideas
- **No adversarial-validation check on the new normalized+grid feature set.**
  Gemini's own exp_003 (`EXPERIMENTS.jsonl:3`, feature fusion with claude's librosa
  features) was killed for a -0.0009 delta, but there's no adversarial AUC number
  logged anywhere in `agents/gemini/workspace/` for `X_train_norm.npy` /
  `X_test_norm.npy` specifically. Given the magnitude of the reported CV gain
  (0.955 -> 0.965 -> 0.968 across `sub_015`/`sub_017`/`sub_018`), running the same
  adversarial-validation probe claude used (`probe_adversarial.py`) on these exact
  1628 features would confirm the shift is actually closed rather than assumed.
- **Ensemble weights are extremely concentrated and somewhat inconsistent with
  reported single-model strength.** `results_norm_blend_v3.json` puts 0.627 weight
  on `pca_mlp_256_256_128` and only 0.00019/0.00005 on the two raw-MLP variants and
  the SVM — three of six models contribute <0.2% combined. COBYLA optimizing
  weights directly against the same 5-fold OOF predictions it's blending risks
  overfitting the blend weights to fold-specific noise (classic stacking
  overfit risk flagged in gemini's own `STRATEGY.md` "What Doesn't Work" section
  for meta-classifiers, but the same risk applies to COBYLA-weight blending, just
  with fewer effective parameters). A nested/held-out weight-fitting split, or a
  simple average vs. the optimized blend comparison, would show how much of the
  0.968 is real vs. blend-weight overfit.
- 7/82 classes have only 3-4 train samples (per claude's earlier sub_002 review) —
  worth confirming `StratifiedKFold(n_splits=5)` isn't still silently degenerate
  for those classes at this feature set; if unresolved it would inflate variance
  in exactly the classes ensemble-weight-fitting is most likely to overfit to.

## Risks
- **Local CV (0.968) vs. confirmed Kaggle LB score: registry shows `kaggle_score:
  null` for sub_017/018/019** (`shared/submissions/registry.json`) — these
  submissions are recorded as `status: submitted` but with no LB confirmation
  captured yet. Given the CV-inflation risks above (rare-class fold degeneracy,
  blend-weight overfitting) plus the general instruction in `RULES.md`'s
  Anti-Overfitting Protocol to distrust CV until LB-confirmed, treat 0.968 as
  provisional until an actual Kaggle score lands in the registry.
- 1628 features against ~2330 training rows (82 classes) is a wide, high-cardinality
  feature set for MLP/SVM; PCA-512 mitigates dimensionality but the grid features
  are highly correlated by construction (adjacent MIDI notes score similarly),
  which is exactly the kind of collinearity gemini's own `STRATEGY.md` flagged
  as degrading a PCA-MLP fold when raw librosa CQT/Chroma/MFCC were added
  (`exp_011`, -0.00858) — worth a sanity check that PCA-512 isn't just re-encoding
  redundant grid columns rather than adding real signal capacity.

## Could Combine With
- Claude's `exp_004` (in flight as of this review — per-clip RMS normalization of
  the librosa-feature pipeline) is directly validated by gemini's independent
  convergence on the same fix; once claude's full-fidelity run lands, an
  adversarial-validation comparison of gemini's grid features vs. claude's
  spectral/chroma features (both post-normalization) would show whether they
  carry complementary signal worth stacking, or whether gemini's grid features
  already dominate and librosa features would just add collinear noise (as
  `exp_011` suggests for the raw/log-transformed versions).
- Claude's cepstral spectral-autocorrelation feature (`spec_acf_lag*` in
  `extract_features.py:92-100`, correlating the log-magnitude STFT) targets the
  same "harmonic spacing implies missing f0" cue as gemini's HPS/grid-harmonic
  features from a coarser angle (6 fixed lags vs. an 81-candidate grid) — likely
  redundant with gemini's finer-grained version rather than complementary.
