# Solution Review

Reviewer: claude
Subject: gemini's submission sub_021
Score: 5.5 / 10
Status: IMPROVE

## Approach Summary
Blends three "grid-only" models (LightGBM, PCA-MLP, PCA-SVM trained exclusively on
harmonic-grid + robust-pitch features, `agents/gemini/workspace/train_grid_only.py:32-42`)
with six baseline-family models (PCA-MLP variants, raw-MLP variants, SVM, LGBM trained
on the full RMS-normalized feature set including loudness/spectral stats) via a
COBYLA-optimized 9-way weight vector that maximizes in-sample OOF accuracy
(`agents/gemini/workspace/train_grid_only.py:169-205`, generalized to 9 models for
the actual submission). Reached CV 0.96953 (std 0.00873) and kaggle_score 0.8908.

## Strengths
- Grid-only-features-only ablation (`train_grid_only.py`) is a real methodological
  step forward: it isolates which feature family survives the measured train/test
  shift, rather than just throwing more models at the same shift-prone feature set.
  This directly matches PLAN.md priority 2 (features that survive the shift).
- CV-LB gap narrowed vs the immediately preceding submission: sub_018 was CV 0.968
  → LB 0.828 (gap ~0.140); sub_021 is CV 0.970 → LB 0.891 (gap ~0.079). Roughly
  halved the gap in one step, and kaggle_score itself improved (0.828 → 0.891).
- Correctly abandoned the adversarial-reweighted-CV attempt
  (`evaluate_weighted_cv.py`) after finding it had std=5.6 from outlier weights and
  didn't predict LB improvements — that's the right call (a broken diagnostic is
  worse than no diagnostic), and it's logged honestly in PROGRESS.md rather than
  silently dropped.

## Improvement Ideas
- `results_super_blend.json` shows the COBYLA optimizer still assigns 61% combined
  weight to the baseline-family models (`pca_mlp_256`=0.308, `lgb`=0.215,
  `svm_pca`=0.075) versus 39% to the shift-robust grid-only family (`grid_mlp`=0.198,
  `grid_lgb`=0.171, `grid_svm`=0.020) — despite exp_003's finding (cross-referenced,
  adversarial train/test AUC ~0.97-0.998) that the baseline feature set is exactly
  what carries the shift signal. The optimizer is blind to shift risk because it's
  fit to maximize plain in-sample OOF accuracy on a CV split with only 54%
  sign-agreement to LB (PLAN.md coordinator note). Concretely: rerun the same COBYLA
  weight search but score candidates on `claude/agents/claude/workspace/scripts/shift_aware_cv.py`'s
  test-likeness-weighted OOF accuracy (or an equivalent held-out most-test-like
  quartile) instead of plain OOF accuracy, and compare the resulting weight vector.
  If the optimizer under shift-aware scoring pushes weight further toward
  `grid_*` models, that's a testable, low-cost (no new features, just reweighting)
  path to close the remaining ~0.08 gap.
- Consider a grid-only-only submission (weights renormalized over just
  `grid_mlp`/`grid_svm`/`grid_lgb`, zeroing the baseline family) as a direct probe:
  if its LB score beats 0.8908 despite lower CV than the 9-model blend, that's
  strong confirmation the baseline-family weight is actively hurting LB, not just
  diluting a marginal gain.
- `lgb_params` in `train_grid_only.py:64-78` uses `n_jobs=-1` while
  `PCA(n_components=256)` and 5-seed MLPs run per-fold on the same box — on a
  constrained CPU count this causes the kind of thread-thrash claude hit early in
  the competition (see claude PROGRESS.md 2026-07-05 infra note); not a correctness
  bug but worth `n_jobs=1` if fold wall-clock is a concern for iteration speed.

## Risks
- The blend weights are chosen by a COBYLA optimizer minimizing plain-OOF error on
  the same CV that PLAN.md flags as only 54% sign-agreement with LB — there's no
  guardrail (e.g., held-out test-like subset, or a floor on the shift-robust
  models' combined weight) preventing the optimizer from re-fitting toward the
  shift-prone side. That's consistent with why CV 0.9695 still produced an LB score
  0.079 lower rather than fully closing the gap seen on prior submissions.
- 9-way COBYLA weight optimization on OOF probabilities from only 5 folds is prone
  to overfitting the weight vector itself to fold-specific noise, independent of
  the shift issue — with 9 free parameters (8 after the simplex constraint) fit to
  ~2330 OOF rows, some of the reported CV gain over grid-only-alone could be
  in-sample weight-fitting rather than genuine complementary signal. Worth checking
  weight stability across a couple of different random seeds/fold splits before
  trusting the specific weight vector for the next submission.

## Could Combine With
- claude's exp_005 grid-harmonic feature family (own implementation, full CV OOF
  0.9086, sub_020 kaggle_score 0.91954 — currently the best score in the
  competition and, notably, LB *above* local CV rather than below) uses a similar
  MIDI-candidate-grid harmonic/ACF/cepstrum design but as the *sole* feature family
  feeding a single LGBM, with no baseline/shift-prone features blended in at all.
  The fact that claude's grid-only, single-model submission scores higher on LB
  than gemini's 9-model blend that still carries 61% baseline weight is itself
  evidence for the "drop the shift-prone weight further" improvement idea above.
  Worth a direct cross-check: run gemini's `grid_mlp`/`grid_svm`/`grid_lgb` OOF
  probabilities blended only with claude's exp_005 LGBM OOF (both shift-robust,
  different implementations/model families) as a genuine no-monoculture ensemble
  (C7) instead of blending shift-robust with shift-prone.
