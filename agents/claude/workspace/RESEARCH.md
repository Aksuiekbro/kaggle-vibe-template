# Claude Research Log

## 2026-07-06 block 2 (post-exp_015 pivot, C9)

Trigger: 3 consecutive negative probes this block on "extend grid+LGBM"
(exp_013 LGBM+MLP ensemble, exp_014 widened ACF/cepstrum lag window, exp_015
temporal-windowed harmonic features). Per STRATEGY.md priority 3 and C9,
pivoting to /research for a fresh angle instead of a 4th variant of the same
family.

Queries used:
- "missing fundamental pitch perception algorithm subharmonic summation
  virtual pitch model" (web search)
- "kaggle missing-fundamental-puzzle competition" (web search -- no results;
  this is a small community competition, not indexed/discussed anywhere
  public. No forum, no public notebooks, no prior LB writeups to upsolve.
  Confirms COMPETITION.md's framing: treat as gym-grade, no external
  leakage risk from public solutions, but also no free lunch from upsolving.)

### Source: Virtual pitch / missing fundamental (Wikipedia + Hermes 1988 SHS)
- **Similarity**: Directly the phenomenon this competition is named after --
  perceived pitch of a complex tone when the literal fundamental frequency is
  absent from the signal, inferred from harmonic spacing.
- **Approach**: Classical pitch-perception models (Goldstein's optimum
  processor / harmonic sieve, Terhardt's virtual pitch, Hermes 1988
  Subharmonic Summation/SHS) score candidate f0s by matching spectral peaks
  to expected harmonic positions, same as our exp_005 grid features. SHS's
  specific refinement: sum spectral energy at each candidate's harmonics
  weighted by an *exponentially decaying* factor per harmonic number
  (compression ratio ~0.84 in the original paper) instead of equal weighting,
  specifically to suppress octave errors (candidate = true f0 / 2) that
  equal-weighted harmonic-sieve scoring is prone to.
- **Score**: n/a (psychoacoustics literature, not a Kaggle leaderboard).
- **Implementation difficulty**: Low (just change the harmonic_scores sum in
  extract_grid_features.py from equal-weighted to decay-weighted).
- **Mechanism hypothesis**: down-weighting higher harmonics could reduce
  octave-confusion between adjacent-in-log-frequency candidate classes.
- **Applicability**: Low-to-medium, and rejected without a full probe --
  reasoning: our current features already give LGBM a `harm_low` (harmonics
  1-3) and `harm_high` (harmonics 4-8) split per candidate on top of
  `harm_sum`; LGBM can already learn arbitrary relative weightings of these
  two sub-sums via tree splits. A hand-crafted exponential-decay reweighting
  is a special case of what the model can already synthesize from existing
  columns, and this block's exp_014/exp_015 already showed that adding
  *more* engineered variants of the same harmonic-scoring feature dilutes
  LGBM's split budget rather than helping (both negative at probe). Explicit
  rejection per C3 -- not queuing as its own experiment; the harmonic-sieve
  vs SHS distinction is very likely already subsumed by harm_low/harm_high.
- **Experiment**: none (rejected).
- **Predicted impact**: n/a (rejected).
- **Risk**: would be a 4th consecutive variant of the same feature family
  this block -- exactly the monoculture-on-one-approach pattern C9 is meant
  to stop.

### Redirect: full-scale MLP-alone probe (not from external research, but a
process gap this research pass surfaced)
- Reviewing why the SHS idea felt weak led to reviewing what evidence we
  actually have against non-tree models on the grid feature family: the
  only MLP number on record (probe_mlp_grid.py, OOF 0.2096 vs LGBM 0.5876)
  was measured on the SAME 300-row/65-class probe subsample that already
  gave the wrong sign twice this competition for LGBM hyperparameters
  (exp_010: probe +0.079 vs full -0.0043; exp_012 confirmed via
  higher-fidelity single-holdout). A 128x64 MLP on ~150 rows/fold is
  underparameterized-data territory almost by construction -- the tiny-probe
  verdict "MLP is far worse" may be a probe-fidelity artifact, not a
  full-scale fact.
- **Experiment queued**: exp_016 -- MLP-alone (StandardScaler + MLPClassifier
  128x64) on the FULL 2330-row/82-class grid features, using a single
  stratified 80/20 holdout (exp_012's cost-fix: full row/class coverage,
  cut fold count instead of row count) vs LGBM on the identical split.
- **Predicted impact**: if MLP-alone full-scale accuracy is within ~0.05 of
  LGBM's, it becomes a viable second model family for a within-grid-family
  blend (the direction exp_008/exp_013 couldn't properly test because SVM's
  cost blew up the full run and the LGBM+MLP-only probe was still on the
  same misleading tiny subsample). If it's still far worse at full scale,
  that closes off model-family diversification on this feature set
  entirely and confirms C7 is satisfied by the grid-vs-generic feature-level
  diversity already measured (exp_006), not by needing model-level
  diversity too.
- **Cost**: ~15-45 min (per exp_012's actual-vs-predicted cost lesson,
  budgeting for the higher end given MLP is typically slower per fit than
  LGBM at this feature count).
- **Risk**: none beyond compute time; no leakage, no rule risk.

## Recommended priority order
1. exp_016 (full-scale MLP-alone, single holdout) -- resolves whether C7
   model-family diversity is worth pursuing further on this feature set.
2. If exp_016 is positive: full LGBM+MLP blend at full scale (proper 3-fold),
   this time with real full-scale evidence instead of a misleading tiny probe.
3. If exp_016 is negative or informative-null: close off model-family
   diversification on the grid feature set for good; remaining C7 coverage
   already comes from grid-vs-generic feature-level diversity (exp_006).
   Next fresh angle after that would need to come from a genuinely different
   data view (e.g. raw-waveform/spectrogram deep features) rather than more
   variants of hand-engineered grid features.

## 2026-07-06 block 3 (research pass while exp_017 full CV runs)

Trigger: exp_016 landed positive (MLP full-holdout beats LGBM, +0.0215) and
exp_017 (proper 3-fold LGBM/MLP/blend CV) is dispatched and running in the
background per STRATEGY.md's queued priority order. Scheduler queue is empty
(exp_017 is the only in-flight item), so per the /day workflow, running a
/research pass now for the *next* round instead of idling.

Queries used:
- "missing-fundamental-puzzle kaggle competition pitch classification" (web
  search -- still no results, confirms the block-2 finding that this is an
  unindexed gym-grade competition with no forum/public notebooks to upsolve)
- "CREPE convolutional pitch estimation CNN spectrogram deep learning F0"
- "HarmoF0: Logarithmic Scale Dilated Convolution For Pitch Estimation" (arxiv 2205.01019)
- "missing fundamental pitch perception residue pitch machine learning classification harmonics"
- "audio data augmentation small dataset imbalanced classes pitch shifting time stretching SpecAugment"

### Source: CREPE (arxiv 1802.06182) and HarmoF0 (arxiv 2205.01019)
- **Similarity**: Both are deep-learning F0/pitch estimators; HarmoF0
  specifically targets harmonic-structure representation via a dilated,
  log-frequency-scale convolution designed to align its receptive field with
  harmonic spacing -- structurally the same idea as our exp_005 grid feature
  (score energy at each candidate f0's harmonic positions), just learned via
  conv weights instead of hand-computed.
- **Approach**: CREPE runs a 6-layer CNN directly on 1024 raw-waveform
  samples, outputs a 360-bin log-frequency probability distribution. HarmoF0
  uses dilated convolutions whose dilation rates are chosen in log-frequency
  space so a single conv layer's receptive field lines up with a candidate
  f0's harmonic series, learned end-to-end from a spectrogram input.
- **Score**: n/a (both are pitch-*tracking* benchmarks in cents-accuracy on
  monophonic instrument/speech corpora with hours of training audio, not
  comparable to our 82-class single-label accuracy metric).
- **Implementation difficulty**: High (training a CNN from scratch) for the
  literal architecture; the PDF's architecture detail didn't extract cleanly
  via WebFetch, but the abstract-level design principle (log-frequency
  receptive field aligned to harmonic spacing) is already what exp_005's
  hand-built grid does, just non-learned.
- **Mechanism hypothesis**: a learned conv filter bank could in principle
  find harmonic-alignment patterns we didn't hand-engineer (e.g. combinations
  of harmonic ranges other than our harm_low/harm_high split, or phase/timing
  cues raw features drop).
- **Applicability**: Low, and rejected without a probe. CREPE/HarmoF0 are
  trained on hours of labeled monophonic pitch-tracking data (continuous f0
  regression, not 82-way single-label classification); our train set is 2330
  clips across 82 classes (avg ~28/class, min 3/class). PREDICTION.md
  pre-registered this exact risk before reading anything ("with only 2330
  train clips across 82 classes... a from-scratch CNN likely overfits").
  A from-scratch CNN (raw-waveform or spectrogram) here would need to learn
  both a harmonic-alignment filter bank *and* an 82-way classifier from a
  dataset smaller than CREPE's per-class validation slice alone. Not worth
  a probe given the SVM-cost-blowup lesson (exp_008) about misjudging cost
  for architectures not suited to this data regime -- rejecting on the same
  small-data-overfit reasoning as the original prediction, not re-deriving it.
- **Experiment**: none queued. Explicit rejection: raw/spectrogram CNN
  pitch-estimation architectures need orders of magnitude more labeled data
  per class than this competition provides.
- **Predicted impact**: n/a (rejected pre-probe).
- **Risk**: would have been high compute cost for a near-certain overfit
  given per-class sample counts; correctly filtered out by C4 (smallest test
  first) reasoning before spending a probe cycle.

### Source: Frontiers — Harmonic Training and the Formation of Pitch
Representation in a Neural Network Model of the Auditory Brain
- **Similarity**: Directly studies whether a neural network can learn to
  classify missing-fundamental ("residue") pitch, the same phenomenon this
  competition is named after.
- **Approach**: A network trained only on pure-tone frequencies could not
  distinguish missing-fundamental pitches at all; a network trained on
  harmonic complex tones (including their f0) learned to classify
  missing-fundamental stimuli correctly, but only for higher-pitched sounds.
- **Score**: n/a (auditory neuroscience modeling paper).
- **Implementation difficulty**: n/a (not a direct experiment; a finding
  about what training signal is necessary).
- **Mechanism hypothesis**: a model only sees the missing-fundamental cue if
  its training data/features expose harmonic relationships, not raw
  frequency content -- exactly why exp_002's generic MFCC/chroma features
  (OOF 0.57) badly underperform exp_005's harmonic-grid features (OOF 0.91).
- **Applicability**: Confirmatory, not new-lever. This is independent
  validation (from auditory neuroscience, not audio ML) of the design choice
  already made in exp_005 and already the single biggest lever measured this
  competition. No new experiment -- logged as corroborating evidence for the
  existing MEMORY_CANDIDATES.md grid-harmonic-features card.
- **Experiment**: none (confirmatory of exp_005, already done).
- **Predicted impact**: n/a.
- **Risk**: none.

### Source: Data Augmentation and Deep Learning Methods in Sound
Classification: A Systematic Review (MDPI 2022, 11(22):3795)
- **Similarity**: Directly addresses this competition's per-class sample
  imbalance (3-56 samples/class, avg ~28) via audio augmentation rather than
  reweighting (exp_009 already showed class_weight='balanced' hurts both
  accuracy and balanced_accuracy at probe scale).
- **Approach**: Survey of augmentation techniques for sound classification --
  noise injection, time-shifting, time-stretching, gain/volume jitter, mixup,
  and spectrogram-domain SpecAugment (time/frequency masking) -- reports
  these help most specifically for underrepresented classes in imbalanced
  sound datasets.
- **Score**: n/a (systematic review, no single benchmark number).
- **Implementation difficulty**: Low-medium: time-stretch, noise injection,
  and gain jitter are all a few lines with `scipy.signal`/numpy on already-
  loaded WAV arrays; would need to re-run grid-feature extraction on
  augmented copies of rare-class clips only (targeted, not full-dataset,
  to control cost).
- **Mechanism hypothesis**: synthesizing extra training examples for the
  17-class tail with under-10 samples gives LGBM more split-decision support
  for those classes without changing the loudness-shift or harmonic content
  that the grid features rely on.
- **Applicability**: Medium-high, WITH ONE HARD CONSTRAINT: pitch-shifting is
  explicitly *not* usable here -- it would change the ground-truth Pitch_ID
  itself, since pitch is literally the label. Only label-preserving
  transforms apply: additive noise, time-stretch (preserves harmonic ratio
  structure, only changes duration), and gain/RMS jitter (exp_004 already
  normalizes RMS, so gain jitter before normalization is a no-op after
  feature extraction -- drop this one). This narrows the augmentation set to
  noise injection + time-stretch only.
- **Experiment**: exp_018 (queued) -- augment only the rare-tail classes
  (<10 samples/class, ~17 of the 82 classes per COMPETITION.md's count) with
  2-3x synthetic copies via additive Gaussian noise (SNR-controlled) and mild
  time-stretch (±5-10%, librosa/resample-based, no pitch change), extract
  grid features on the augmented set, and compare full CV accuracy
  (especially per-class/balanced accuracy on the augmented classes) against
  exp_005's un-augmented baseline. Probe first on the existing 300/100
  subsample's rare classes before a full run, per C4/C13.
- **Predicted impact**: +0.01 to +0.03 overall accuracy, larger on
  balanced_accuracy / rare-class recall specifically (this targets the tail,
  not the head, so plain accuracy may under-state the benefit -- track
  balanced_accuracy alongside plain accuracy per exp_009's precedent).
- **Risk**: Leakage risk if augmentation is applied before the CV split
  (augmented copies of a rare clip could land in both train and validation
  fold) -- MUST augment strictly inside each fold's training data only, never
  before the split. This is the main way this experiment could produce a
  falsely optimistic number; flag prominently in the script and check for it
  in review.

## Recommended priority order (block 3)
1. exp_017 (in flight) -- proper 3-fold CV confirmation of exp_016's
   LGBM/MLP/blend result; nothing else should be promoted to full ahead of
   this since it's the direct next step in an already-positive chain.
2. exp_018 (new, this research pass) -- rare-tail-class augmentation
   (noise + time-stretch only, fold-safe), the first genuinely new feature/
   data-side lever since exp_005 itself. Probe on the existing 300/100
   subsample's rare classes first.
3. Raw-waveform/spectrogram CNN (CREPE/HarmoF0-style): rejected pre-probe,
   data regime is 1-2 orders of magnitude too small per class for a
   from-scratch pitch-estimation CNN. Would need external pretrained
   weights (e.g. a released CREPE checkpoint) to be worth reconsidering --
   not pursued this block since COMPETITION.md doesn't flag external
   pretrained model use either way and it's a much bigger lift than exp_018.

## 2026-07-07 block 4 (queue empty after exp_024/028, C9-style pivot to research)

Trigger: scheduler queue emptied after exp_024 (feature-shift-vs-importance
drop, full delta -0.0004, closed) and exp_028 (frozen CREPE embedding probe,
delta -0.0206, closed -- see below). exp_006/020/022/023/024 have now all
failed to explain sub_022's CV-up/LB-down drop via model/blend/feature-level
diagnostics; no new lead from `memory_cli retrieve --anchor stuck` or
PLAN.md. Ran this research pass per the /day workflow's queue-empty step.

Queries used:
- "kaggle missing-fundamental-puzzle competition 2026" (web search -- no
  results again; confirmed still a small, unindexed community competition,
  consistent with block-2/3 findings)
- "small labeled audio dataset few-shot classification pretrained embeddings
  VGGish PANNs OpenL3 transfer learning" (web search)

### exp_028: frozen CREPE (torchcrepe, pitch-specific pretrained CNN) as feature extractor -- REJECTED at probe

- **Source**: queued from block 3's /research pass (own idea, motivated by
  cross-reviewing gemini's harmonic-grid approach + general CNN-pitch-
  tracking literature)
- **Similarity**: same modality (mono WAV, missing-fundamental pitch), but
  CREPE is trained for single-note monophonic f0 *estimation* on instrument/
  singing datasets, not this competition's fixed 82-class Pitch_ID taxonomy
- **Approach**: `torchcrepe` (model='tiny', frozen, no fine-tuning) 5th-
  maxpool embedding, mean+std pooled over time to a 512-dim vector per clip,
  concatenated onto exp_005's 405 grid-harmonic features (917 total)
- **Score**: n/a (this repo's use, not a published result)
- **Implementation difficulty**: Medium -- required pinning
  torch==2.11.0+cpu / torchaudio==2.11.0+cpu (torchcrepe pulls torchaudio,
  which lags torch's PyPI-default release by several minor versions --
  mismatched CUDA-runtime-linked wheels raised `OSError: libcudart.so.13`
  until both were pinned to matching `+cpu` builds from the PyTorch CPU
  index). Checkpoint is bundled with the pip package, no separate download.
- **Mechanism hypothesis**: a pretrained pitch-tracking CNN's internal
  representation might encode harmonic/periodicity structure more robustly
  than our hand-built grid features, especially under the train/test shift
  exp_003/004 measured.
- **Applicability**: same 291-row/65-class probe subsample and LGBM config
  as `probe_lgbm_grid.py` for direct comparability to the known exp_005
  reference (0.5876).
- **Experiment**: `extract_crepe_embed.py` + `probe_exp028_crepe.py`
- **Predicted impact**: +0.02 (per scheduler queue entry)
- **Risk**: compute cost (mitigated: 'tiny' model, scipy resample_poly
  instead of librosa's default resampler cut per-file cost from ~8s to
  ~0.7s), rules ambiguity on external pretrained models (COMPETITION.md/
  RULES.md silent either way; treated as allowed since it's used only as a
  frozen feature extractor, no test labels touched, same standard as any
  other third-party library)
- **Result**: grid-only OOF=0.5876 (exact match to exp_005's known
  reference) vs grid+CREPE OOF=0.5670 (delta **-0.0206**); CREPE-embedding-
  alone OOF=0.1649 (**-0.4227** below grid-only). Recorded via
  `scheduler.py record --id exp_028 --stage probe --delta -0.0206` --
  KILLED at probe cost, C9.
- **Why it failed (hypothesis)**: CREPE is trained to find *the* fundamental
  of a monophonic tone; this competition's clips are specifically
  constructed so the literal fundamental is absent/ambiguous (the
  competition's namesake phenomenon), and Pitch_ID is a designed class
  label, not CREPE's continuous f0 estimate target. The domain mismatch is
  severe enough that even a frozen embedding actively hurts (adds noise
  dimensions) rather than just failing to help. Consistent with the
  pre-registered PREDICTION.md risk and block-3's pre-probe rejection of
  raw-waveform CNNs for the same reason (data regime + task mismatch).

### exp_029: frozen PANNs (Cnn14, AudioSet-trained general audio-event CNN) as feature extractor -- probe launched, result pending

- **Source**: this research pass -- multiple 2023-2024 audio-transfer-
  learning papers (EUSIPCO, arXiv) found general-purpose AudioSet-pretrained
  embeddings (VGGish/OpenL3/PANNs) transfer well to small novel-task
  datasets, sometimes better than domain-specific pretrained models, because
  they encode broad timbral/spectral structure rather than one narrow task
  target.
- **Similarity**: different mechanism from exp_028 -- Cnn14 is trained for
  527-class general sound-event *tagging* (not pitch estimation), so it is
  not looking for "the" fundamental at all; it should instead capture
  timbral/harmonic-envelope cues, which is closer to what distinguishes this
  competition's Pitch_ID classes (harmonic-complex missing-fundamental
  tones) than CREPE's single-f0 target.
- **Approach**: `panns_inference` (Cnn14, AudioSet, frozen, no fine-tuning),
  2048-dim global-pooled penultimate-layer embedding per clip, resampled to
  32kHz via `scipy.signal.resample_poly` (same fast-resample fix as
  exp_028).
- **Implementation difficulty**: Low -- pure-PyPI install, reused the
  torch==2.11.0+cpu env already fixed for exp_028; checkpoint (~327MB)
  auto-downloads from Zenodo/GCS on first use (~22s on this box's
  connection), then loads from local cache (~5s) on subsequent runs.
- **Mechanism hypothesis**: broad timbral embedding adds signal the grid
  features miss, OR (given exp_028's result) any generic externally-
  pretrained embedding underperforms this task's hand-built harmonic-grid
  features regardless of source domain, because 82 finely-spaced note
  classes need pitch-height precision no general-purpose sound-event
  classifier was trained to preserve.
- **Experiment**: `extract_panns_embed.py` + `probe_exp029_panns.py` (same
  291-row/65-class subsample, same LGBM config as exp_028).
- **Predicted impact**: +0.01 (lower than exp_028's prior +0.02 estimate,
  given exp_028's negative result already weakens the general "external
  pretrained embedding helps here" prior)
- **Risk**: same rules-ambiguity note as exp_028 (external pretrained model,
  frozen feature extractor only)
- **Result**: grid-only OOF=0.5876 vs grid+PANNs OOF=0.5739, delta
  **-0.0137**; PANNs-alone OOF=0.0515 (**-0.5361** below grid-only -- even
  weaker standalone than exp_028's CREPE, 0.1649). Recorded via
  `scheduler.py record --id exp_029 --stage probe --delta -0.0137` --
  KILLED at probe, C9.
- **Why it failed (hypothesis)**: this task needs fine, semitone-level pitch
  discrimination across 82 classes. PANNs pools its representation for
  coarse sound-*event* identity (is this a dog bark or a car horn), not
  pitch height -- the opposite failure mode from CREPE (which collapses to
  a single, often-wrong f0 estimate) but the same outcome: neither
  pretrained objective preserves what this task's labels actually
  discriminate on.

**External-pretrained-audio-embedding family is now closed on this
dataset**: 2 architecturally distinct pretrained CNNs (CREPE -- narrow,
pitch-specific; PANNs/Cnn14 -- broad, general-purpose event tagging) both
hurt when concatenated onto the grid features, and both are dramatically
worse standalone. No reason to expect a 3rd architecture (e.g. OpenL3,
VGGish) would behave differently -- both ends of the specificity spectrum
already failed. This is a genuinely new, well-evidenced negative result
worth sharing with gemini/codex if they consider pretrained-embedding
features on this dataset.

## Recommended priority order (block 4, updated)
1. exp_028/exp_029 both closed (external pretrained embeddings, 2/2
   negative) -- do not revisit this family without a new mechanism.
2. Remaining fresh angles are thin. Consider a deeper dig into the still-
   unexplained sub_022-style CV-LB drop directly (5 diagnostics have now
   failed to explain it: exp_006, 020, 022, 023, 024), or a prospective
   pattern-transfer audit once the competition's discussion/winner
   writeups become available.
3. sub_024 (exp_025's augmented blend, local CV 0.9524) was submitted this
   block to test whether the augmentation fix improves LB transfer vs.
   sub_022's same-architecture blend. **Result already back**: kaggle
   publicScore 0.89655 / privateScore 0.87775 -- still worse than sub_020's
   publicScore 0.91954 / privateScore 0.89242 on BOTH splits, despite the
   higher local CV (0.9524 vs 0.9086) and despite augmentation being a
   genuine data-level fix (not just blend-weight tuning). This means the
   LGBM+MLP blend *architecture itself* (not the lack of augmentation, not
   the specific blend weight) is the common factor behind every CV-up/
   LB-down drop observed so far -- sub_020 (single LGBM model, no blend)
   remains the best submission on every split (CV, public, private) despite
   being the simplest. Strengthens (does not just repeat) the existing
   "don't trust blend-only CV gains on this dataset" memory candidate: it
   now has a same-architecture, augmentation-controlled confirmation, not
   just the original single data point.
4. **IMPORTANT, flagged to the user, not yet resolved**: while checking
   sub_024's kaggle_score via `kaggle competitions submissions`, found 7
   `codex_*`-named submissions from 2026-07-05 scoring up to privateScore
   1.00000 -- with zero corresponding entries in `shared/submissions/
   registry.json` and a completely empty `agents/codex/workspace/
   PROGRESS.md`. These bypassed the score gate and the shared daily
   submission budget, and a perfect private score on a task with a
   confirmed, still-unexplained train/test shift is a strong leakage/
   overfitting red flag rather than a plausible modeling win. Not
   investigated further pending user guidance -- do not treat 1.00000 as a
   real bar to chase, and do not assume codex's untracked approach is safe
   to imitate.

## Research pass (block 24, 2026-07-07) -- queue-refill while exp_043/044 occupy both CPUs

Queue was empty per `scheduler.py next` at block start. No competition-specific
discussion/notebook exists (this is a late/gym-grade competition; general web
search for "missing-fundamental-puzzle kaggle" returns nothing competition-
specific). Searched adjacent literature instead, per charter priority 2/4.

- **Source**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6445256/ ("Robust
  Harmonic Features for Classification-Based Pitch Estimation")
- **Similarity**: classification-based (not regression) pitch estimation from
  harmonic structure -- same problem shape as this competition's grid-harmonic
  feature family (exp_005).
- **Approach**: 5 per-F0-candidate features derived from harmonic energy
  distribution: harmonic-energy ratio, subharmonic-summation amplitude ratio,
  harmonic-frequency deviation from ideal integer multiples, ratio of
  identified harmonic partials, and odd-to-even harmonic energy ratio (o2e)
  specifically to catch half-pitch/subharmonic errors (if candidate c is
  actually f_true/2, energy at c's odd multiples is missing since true energy
  only sits at multiples of f_true=2c, which are all even multiples of c).
- **Score**: not stated (paper is a features/methods paper, not a leaderboard).
- **Implementation difficulty**: Low -- reuses the same per-candidate harmonic
  energy computation the grid features already do.
- **Mechanism hypothesis**: o2e specifically targets a self-consistency check
  (is what I'm calling harmonic 1,2,3.. of this candidate actually complete)
  rather than exp_035's neighbor-difference framing (own low harmonics vs
  octave-neighbor's low harmonics) -- different arithmetic, same general
  target (octave/subharmonic ambiguity), which is directly relevant since
  exp_034 already confirmed 58% of OOF errors cluster at 7/12/19/24-semitone
  distances.
- **Applicability**: high on paper, but the grid-harmonic-feature-family axis
  is **closed per C9**: 4/4 prior variants (exp_007, exp_014, exp_015, and
  exp_035 -- which specifically targeted the same octave-confusion mode with
  a different neighbor-ratio formula) all reversed sign, with exp_035 being a
  full-CV-confirmed negative (topq -0.0051), not just a probe artifact.
  MEMORY_CANDIDATES.md already states explicitly: "do not attempt further
  grid-harmonic feature-family variants on this dataset regardless of how
  well-motivated."
- **Experiment**: none queued. Explicit rejection, not silent drop.
- **Predicted impact**: n/a (rejected pre-probe)
- **Risk**: repeating exp_035's exact failure mode (dilutes LGBM/XGBoost split
  budget with one more near-collinear per-candidate ratio column) for a
  family with a 0/5 track record if this were tried.

- **Source**: https://medium.com/data-science/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872
  and https://arxiv.org/pdf/2407.14381 ("Improving GBDT Performance on
  Imbalanced Datasets: An Empirical Study of Class-Balanced Loss Functions")
- **Similarity**: same problem shape (severe multiclass imbalance, 3-56
  samples/class, tree-based models already the confirmed best family here).
- **Approach**: focal loss as a custom GBDT objective -- gradient/Hessian
  reweight *by prediction confidence* (the gamma hard-example-mining term),
  not by raw class frequency.
- **Score**: reported accuracy gains over plain log-loss on imbalanced
  benchmarks in both sources (not directly comparable numbers to this task).
- **Implementation difficulty**: Medium -- needs a custom multiclass
  objective (explicit gradient/Hessian, OvR decomposition) passed to LGBM/
  XGBoost's `objective=` argument; no new features, existing grid-harmonic
  columns unchanged.
- **Mechanism hypothesis**: distinct from exp_009's already-killed
  `class_weight='balanced'` (probe delta -0.0309) -- that reweights every
  sample in a class equally by inverse frequency regardless of whether the
  model already gets it right; focal loss reweights per-sample by how wrong
  the *current* prediction is, so easy majority-class rows stop dominating
  the gradient even without touching rare-class weights directly. Different
  mechanism, not a re-run of a closed lever.
- **Applicability**: plausible fresh axis -- imbalance-handling via
  `class_weight` is closed (exp_009) but hardness-based reweighting has not
  been tried on this dataset, and it composes with (does not replace)
  exp_037/038's data-level rare-class augmentation.
- **Experiment**: queued as exp_046 (see below) -- probe first (single 80/20
  full-data holdout per exp_012's fidelity lesson) on top of exp_042's
  current-best XGBoost+more_oversample-augmented config, scored on the same
  plain/weighted/topq metrics.
- **Predicted impact**: +0.01 (small; augmentation + model-family levers
  already captured most of the easy imbalance-handling gain, so this is
  a refinement, not expected to match exp_040/042-sized wins)
- **Risk**: custom-objective Hessians for multiclass are easy to get subtly
  wrong (silent convergence-quality bugs rather than crashes) -- probe must
  sanity-check against plain-objective LGBM on a class it's known to handle
  (e.g. the majority class) before trusting the delta.

## Recommended priority order (block 24)

1. Let exp_044 (GBDT-only blend, in flight) land -- first ensemble candidate
   since the freeze; highest-value open question right now.
2. exp_046 (focal-loss GBDT objective, queued this block) is the only fresh,
   non-closed axis found this pass -- probe once a CPU frees up.
3. Grid-harmonic feature-family additions (any formula, including o2e)
   remain rejected per C9 -- do not revisit without a qualitatively new
   mechanism outside harmonic-ratio arithmetic.
4. Stacking meta-learners on GBDT OOF probabilities are also closed
   (exp_021, probe delta -0.0129) -- do not re-suggest plain logistic-
   regression stacking; exp_044's soft-vote blend is testing a different
   (non-learned-weight) combination mechanism, not a re-run of exp_021.

## Research pass, block 26 (2026-07-07)

exp_044 (GBDT blend), exp_045/exp_046 (focal-loss, importance-weighting) all
landed and were killed since the last pass -- imbalance-handling and
ensembling axes are now both closed 2/2 and 2/2 respectively. Queue was
empty going into this block; ran a fresh research pass per C9/the /day
workflow's "queue empty -> research" branch.

Queries: "pseudo-labeling self-training small labeled dataset train test
distribution shift tabular classification", "CORAL domain adaptation feature
alignment train test shift small dataset technique", "missing fundamental
pitch perception harmonic template matching classification machine
learning". Reads logged via `writeup.py log` (3 papers, see command history).

- **Source**: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep
  Domain Adaptation" (arxiv/1607.01719); classic (non-deep) CORAL is the
  base method being adapted here.
- **Similarity**: unsupervised domain adaptation for exactly this
  competition's problem shape -- labeled source (train) and unlabeled
  target (test) with a real, measured covariate shift (exp_003 adversarial
  AUC 0.97-0.998).
- **Approach**: whiten source features by source covariance, recolor with
  target covariance -- a single linear transform fit from feature matrices
  alone (no target labels used, so it's a legitimate train-time transform,
  not leakage).
- **Score**: not competition-specific; CORAL is a general method, not a
  leaderboard result to compare against.
- **Implementation difficulty**: Low -- closed-form linear algebra
  (two covariance matrices, one eigendecomposition-based matrix square
  root/inverse-square-root, one matmul), no new training loop.
- **Mechanism hypothesis**: genuinely new axis. Every shift-mitigation tried
  so far operated on the LOSS (class_weight exp_009, focal-loss exp_045,
  importance-weighting exp_046 -- all killed) or per-clip SCALE
  (RMS-normalization exp_004) or added DATA (augmentation exp_018/033/
  037/038). None directly transformed the feature distribution's
  second-order statistics to match test. If the exp_003/exp_006 shift is
  well-approximated by a covariance-level mismatch, CORAL could close it
  directly instead of routing around it.
- **Applicability**: uncertain -- exp_006 already showed grid-harmonic
  features have a small topq penalty despite a high raw adversarial AUC
  (the shift and the class-relevant signal mostly don't overlap for this
  feature family), so there may be little room left for a feature-alignment
  fix to help, and an aggressive linear transform risks removing
  class-relevant variance along with shift variance.
- **Experiment**: queued as exp_048 (scheduler id exp_047) --
  `scripts/probe_exp048_coral_alignment.py`. Single 80/20 full-data holdout
  (exp_012 fidelity lesson), CORAL-align (train+augmented) features to the
  REAL test set's covariance, refit XGBoost+more_oversample (exp_042
  config), score plain/weighted/topq vs the untransformed exp_042
  equivalent on the same split. Includes a CORAL(X,X)-is-identity sanity
  check before trusting any delta (same discipline as exp_045/046's
  plumbing sanity checks).
- **Predicted impact**: +0.015 (moderate-optimistic; genuinely new
  mechanism, but exp_006's finding above is a real reason to expect a small
  or even negative effect).
- **Risk**: linear covariance alignment assumes the dominant shift is
  second-order and feature-wide; if the real shift is concentrated in a
  few dimensions or is non-linear, CORAL could distort informative
  dimensions along with noise ones. Also computationally the cheapest
  fresh lever found this pass (closed-form, no retraining loop needed for
  the transform itself), so worth trying even if the prior is not strongly
  positive.
- Second query ("pseudo-labeling / self-training under distribution shift
  for tabular data") surfaced a plausible but higher-risk alternative
  (density-weighted pseudo-label acceptance) -- deprioritized behind CORAL
  because it needs careful fold-honest implementation to avoid label
  leakage and compounds noise risk on this dataset's already-thin rare
  classes (3-56 samples/class). Queue as a follow-up only if CORAL is
  positive and a genuinely new lever is still needed.
- Third query (missing-fundamental pitch-template literature) reconfirmed
  exp_005's harmonic-template design matches neuroscience models (Power
  Series Template model, bioRxiv/2023.01.27.525831) but produced no new
  actionable lever -- same convergence pattern as block 3's Frontiers
  paper. Logged and closed, no experiment row needed beyond the citation.

## Recommended priority order (block 26)

1. exp_048 (CORAL feature alignment, probe running this block) is the only
   fresh, non-closed axis found this pass -- let it land, sanity-check the
   identity-transform check before trusting the delta.
2. If exp_048 is negative, the fresh-idea well from research is close to
   dry for this dataset (imbalance handling, ensembling, TTA, feature-family
   variants, model-family tuning, and now domain-adaptation-by-covariance
   are all closed or pending closure) -- next pass should look specifically
   for techniques from Kaggle discussion threads on similarly-shaped
   audio-classification-with-shift competitions, if any are findable, rather
   than another generic ML-literature sweep.
3. Grid-harmonic feature-family additions remain rejected per C9 (4/4
   negative, exp_007/014/015/035) -- do not revisit.
4. exp_042 (XGBoost+more_oversample, full topq +0.0343 vs LGBM-plain
   baseline) submitted this block as sub_025 -- first submission since the
   freeze validated purely against the coordinator's shift-aware gate
   rather than raw CV; outcome pending (public LB score not yet synced).

## 2026-07-07 block 27 (exp_048 CORAL result landed, closes domain-adaptation-by-covariance)

exp_048 (CORAL feature-space alignment, probe) landed: **negative on the
gating metric** -- plain -0.0343, weighted +0.0585, topq -0.0256. Mixed
cross-metric signal (weighted alone would have looked promising) but
topq -- the coordinator's stated gate -- says no. Matches the pre-registered
risk in the exp_047/048 queue entry almost exactly: exp_006 already showed
grid features are largely shift-robust in effect, so an aggressive
full-covariance transform removes some class-relevant variance along with
shift variance. KILLED at probe cost (`scheduler.py record --id exp_047
--stage probe --delta -0.0256`). Domain-adaptation-by-covariance is now
closed alongside the other shift-mitigation axes (loss-reweighting,
per-clip scale is separate and stays open/positive, augmentation).

Re-confirmed this block: the competition itself has no real discussion
forum or public notebooks to search (WebFetch on the discussion URL
returned only the Kaggle homepage tagline, consistent with block 2's
finding and COMPETITION.md's note that this may be a closed/late-submission
community competition) -- the "Kaggle-discussion-specific" research
recommended in block 26's priority order is not available as a lever here;
sticking with general ML literature + own-diagnostic-driven ideas.

New queries used (general ML literature, since Kaggle-discussion search is
a dead end for this competition):
- `test-time batch normalization statistics adaptation covariate shift
  tabular gradient boosting` (web search)

Found: test-time batch-statistics-calibration literature (arxiv/2110.04065
and related) for deep nets under covariate shift consistently warns that
substituting/mixing full batch statistics can distort discriminative
structure, and that gentler calibration (rescaling only, no full covariance
rotation) tends to preserve it better. This maps directly onto why exp_048
may have gone net-negative: CORAL's off-diagonal rotation can mix feature
correlations along with matching 2nd-order shift statistics. Queued as
**exp_049**: diagonal-only (per-feature mean/std rescale, zero off-diagonal
terms) alignment to test's marginal stats -- same probe harness as exp_048
for a direct comparable delta. This is the last variant on this axis worth
trying (a strictly lower-capacity special case of the just-closed
full-covariance transform); if it also misses topq, feature-distribution
alignment as a family is closed for good and the next block should look
for a genuinely different mechanism (not another alignment-transform
variant) per C9.

## Research pass, block 28 (2026-07-07) -- feeding the queue while exp_054 hyperparameter probe runs

Context: feature-alignment/elimination family (CORAL, diagonal, adversarial-
importance elimination) is now 3/3 negative at full-CV-or-equivalent scale
(exp_047/048/049/050). exp_051 (fold-honest pseudo-labeling, real test rows
added as training data) just landed its first full 3-fold CV result: small
but genuine positive (topq +0.0017, plain +0.0021, weighted +0.0011 vs
exp_042 baseline) -- the only open, not-yet-closed shift-mitigation
mechanism left. This pass looks for ways to extend/improve that specific
mechanism rather than starting a new generic-ML-literature sweep from
scratch, per block 26/27's finding that this competition has no discussion
forum (WebFetch on the discussion URL still returns only the Kaggle
homepage tagline -- reconfirmed, not re-logged).

Queries used:
- `iterative self-training pseudo-labeling confidence threshold small
  imbalanced multiclass tabular gradient boosting 2025` (web search)
- `Kaggle small imbalanced audio classification winning solution few
  samples per class rare class augmentation` (web search)
- `manifold mixup tabular features gradient boosted trees small dataset
  augmentation` (web search)
- `"Dataset Balancing Can Hurt Model Performance" arxiv 2307.00079 summary
  findings` (web search, following up a hit from the second query)

### Finding 1: regularized (density-weighted) pseudo-label acceptance for tabular self-training

- **Source**: [Revisiting Self-Training with Regularized Pseudo-Labeling for
  Tabular Data](https://ar5iv.labs.arxiv.org/html/2302.14013) (arxiv
  2302.14013)
- **Similarity**: tabular self-training with an unlabeled pool, explicitly
  targets gradient-boosted-tree-compatible methods (no gradient-based
  consistency regularization needed) -- matches our XGBoost+real-unlabeled-
  test-set setup (exp_051) exactly.
- **Approach**: replace a raw-confidence pseudo-label threshold with a
  combined score f(x) = (alpha*gamma + 1)*c/(alpha+1), where c is classifier
  confidence and gamma is the empirical feature-likelihood of the row given
  its predicted class (per-feature binned histograms, cheap, no extra
  training loop). This keeps pseudo-labels in high-density regions of the
  predicted class instead of accepting any high-confidence-but-possibly-
  atypical row.
- **Score**: reported rank ~1.8 vs ~3.3 (lower better) vs vanilla fixed-
  threshold self-training across multiclass benchmarks; largest gains with
  limited labeled data (their 500-label healthcare setting saw 23-29% F1
  gains) -- directionally matches our regime (2330 rows/82 classes, 3-56
  samples/class in the rare tail).
- **Implementation difficulty**: Low -- reuses exp_051's exact fold-honest
  harness (`full_cv_exp051_pseudolabel.py`), only the acceptance-scoring
  function changes; feature-likelihood via binned histograms is near-free
  next to the XGBoost refits already being paid for.
- **Mechanism hypothesis**: exp_052's threshold sweep (0.95/0.85/0.7) found
  a non-monotonic optimum at 0.85, suggesting raw confidence alone is an
  imperfect proxy for "this pseudo-label is trustworthy" -- some high-
  confidence rows may still be atypical/mislabeled outliers that a density
  check would catch, and some borderline-confidence rows may be typical and
  worth keeping. A combined score could sharpen the trade-off curve exp_052
  already found evidence for, rather than just re-finding the same optimum
  under a different name.
- **Applicability**: Direct -- no adaptation needed beyond wiring the
  likelihood term into the existing acceptance-mask computation.
- **Experiment**: queued as **exp_055** (scheduler id exp_054,
  `scheduler.py add`) -- probe first (single 80/20 holdout, same split as
  exp_051/052) sweeping alpha in {0.25, 0.5, 1.0}, scored against exp_051's
  own threshold=0.85 result (not just the exp_042 no-pseudo-labeling
  baseline) so the comparison isolates the acceptance-mechanism change.
- **Predicted impact**: +0.01 (modest; exp_051's whole mechanism only moved
  topq by +0.0017, so even a "clear win" for the acceptance function has a
  low ceiling here -- flagging this expectation now so a small full-CV
  delta isn't over-read later).
- **Risk**: extra hyperparameter (alpha) tuned against the same 80/20
  holdout used for the exp_052 threshold sweep -- some risk of holdout
  overfitting if both are tuned together; mitigate by validating the
  chosen alpha at full 3-fold CV before trusting it, same discipline as
  exp_050/051.

### Finding 2 (counter-evidence, C11): balancing/rebalancing can hurt on shifted eval sets

- **Source**: [Dataset Balancing Can Hurt Model
  Performance](https://arxiv.org/abs/2307.00079) (ICASSP 2023, arxiv
  2307.00079)
- **Similarity**: AudioSet-scale audio tagging with severe class imbalance,
  evaluated on both a public split and a held-out set collected under the
  same conditions -- structurally close to our own public-LB-vs-private-LB
  (and CV-vs-LB) divergence problem.
- **Approach**: measures effect of class-balancing on a published eval set
  vs. an unpublished one collected the same way.
- **Score**: balancing improved the public eval metric but *hurt* the
  unpublished-set metric; no evidence it helped rare classes relative to
  common ones net of that trade-off.
- **Implementation difficulty**: N/A (not an experiment to run, a risk
  flag).
- **Mechanism hypothesis**: rebalancing during training changes the
  effective prior the model learns, which can overfit to whichever split
  the tuning was validated against, mirroring exactly the CV-up/LB-down
  mechanism the coordinator flagged this competition already has (PLAN.md
  notes #1/#2).
- **Applicability**: This competition's own rare-class augmentation
  (`more_oversample`, exp_037/038 winner, composed into exp_042/sub_025)
  was already validated against the shift-aware topq metric rather than
  raw CV, which is closer to this paper's recommended caution than the
  failure mode it describes -- but it's a reason to keep watching sub_025's
  real private-LB outcome once available rather than treating exp_042 as
  fully de-risked. **No new experiment queued** -- this is counter-evidence
  attached to exp_037/038/042's existing record, not an actionable lever.
- **Experiment**: none (explicit rejection as a new lever; recorded as
  counter-evidence only, per C11).
- **Predicted impact**: n/a.
- **Risk**: n/a (the risk this raises is about interpretation of past
  results, not a new mechanism to test).

### Finding 3: TabMDA (transformer-based manifold data augmentation) -- rejected

- **Source**: [TabMDA: Tabular Manifold Data Augmentation for Any Classifier
  using Transformers with In-context
  Subsetting](https://arxiv.org/pdf/2406.01805) (arxiv 2406.01805)
- **Similarity**: tabular data augmentation compatible with any downstream
  classifier including GBDTs, explicitly targets small-data regimes.
- **Approach**: uses a pretrained in-context tabular transformer to build a
  latent manifold, then augments by sampling/perturbing in that latent
  space (training-free at the augmentation step, but requires a large
  pretrained tabular foundation model as the encoder).
- **Score**: not evaluated in a regime close to ours in the abstract/skim
  (general small-tabular-dataset benchmarks, not audio-derived 400+-dim
  handcrafted features with 82 highly imbalanced classes).
- **Implementation difficulty**: High -- needs a pretrained tabular
  foundation-model encoder (e.g. TabPFN-style), which is not in this
  project's dependency set and would need new infra (network access /
  model download) just to try.
- **Mechanism hypothesis**: nonlinear latent-space mixing could create more
  diverse synthetic rare-class rows than the current SNR/rate-based audio-
  domain augmentation (exp_018/033/037 `more_oversample`) -- but that
  augmentation is already shift-aware-validated as the family winner, and
  exp_048/049 already demonstrated that tree-based split-finding is
  invariant to per-feature monotonic transforms, so any benefit here would
  have to come specifically from cross-feature nonlinear mixing, not
  rescaling -- unproven for this feature set.
- **Applicability**: Low confidence given the infra cost and lack of a
  close-match benchmark; the existing audio-domain augmentation directly
  encodes known-valid perturbations (SNR/pitch-preserving time changes),
  which a generic tabular-manifold method would have no way to guarantee it
  respects (e.g. it could rediscover something close to pitch-shift, which
  exp_018's design notes already flagged as label-invalidating for this
  task).
- **Experiment**: none -- rejected, not queued.
- **Predicted impact**: n/a (rejected).
- **Risk**: infra cost, weak similarity to this task's feature modality,
  and no guardrail against generating physically-invalid synthetic rows
  (unlike the domain-aware SNR augmentation already in use).

## Recommended priority order (block 28)

1. Let exp_054 (scheduler id exp_053, XGBoost hyperparameter coordinate
   search) finish and record -- already running this block.
2. exp_055 (density-regularized pseudo-label acceptance, probe first) is
   the one fresh, actionable lever this pass found -- extends exp_051's
   already-positive-at-full-CV mechanism rather than reopening a closed
   family.
3. Dataset-balancing counter-evidence (Finding 2) does not change any
   current decision but should be weighed if sub_025's private-LB score
   (once synced) diverges further from its 0.94252 public score than the
   competition's already-large CV/LB gap would predict.
4. TabMDA (Finding 3) rejected -- infra cost too high relative to expected
   payoff given tree-monotonic-invariance already limits what a manifold
   transform could add here.

## Research pass, block 33 (2026-07-07) -- pseudo-labeling sub-axis nearly closed, looking for a genuinely new mechanism family

Context: pseudo-labeling's two sub-axes (acceptance-scoring: raw threshold
vs density-regularized, closed exp_054/exp_055; iteration count and
threshold resolution: exp_056/exp_057, running this block) are the last
open items in the existing queue. Per PLAN.md block 32's note, once these
close the well is close to dry (feature families, model families,
ensembling, hyperparameters, domain adaptation, imbalance handling all
closed 3+/3+ negative). This pass looks for a mechanism family that has NOT
been tried yet, rather than another variant of a closed family.

Queries used:
- `prototypical network few-shot audio classification imbalanced classes
  small sample per class` (web search)
- `pitch class octave decomposition hierarchical classifier chroma missing
  fundamental pitch estimation` (web search)
- `"deep layered learning" pitch estimation chroma octave two-stage
  singing` (web search)

### Finding 4: hierarchical chroma+octave decomposition (deep layered learning pattern)

- **Source**: [Deep Layered Learning in MIR](https://arxiv.org/pdf/1804.07297)
  (arxiv 1804.07297, Elowsson) and [Modeling Music Modality with a
  Key-Class Invariant Pitch Chroma CNN](https://arxiv.org/pdf/1906.07145)
  (arxiv 1906.07145)
- **Similarity**: both address pitch/key estimation by decomposing the
  target into an octave-invariant chroma (pitch-class-mod-12) component and
  a separate octave/register component, rather than one flat classifier
  over the full pitch range -- directly matches this competition's own
  data shape (Pitch_ID 0-81, confirmed integer/semitone-indexed, i.e.
  `chroma = Pitch_ID % 12`, `octave = Pitch_ID // 12`).
- **Approach**: train the first stage/layer to predict octave-invariant
  chroma (pooling or feature-summarizing across octave-shifted views),
  then a second stage resolves register/octave on top of the chroma
  prediction. Ablations in the deep-layered-learning literature attribute
  most of the performance gain to this factorization, not to a bigger
  model.
- **Score**: not directly comparable (different task/dataset -- key/chord
  estimation, not isolated-clip pitch classification), qualitative
  ablation result only (chroma-layered variant beats flat baseline).
- **Implementation difficulty**: Low-Medium -- no new features needed,
  reuses exp_042's existing grid-harmonic feature set; just requires
  refactoring the 82-way multiclass target into two heads (12-way chroma
  classifier + per-chroma or global octave classifier) and a combine step
  at inference (argmax chroma x argmax octave, or joint scoring).
- **Mechanism hypothesis**: exp_034 already measured that 58.0% of this
  model's own OOF misclassifications land at exactly the semitone
  distances (7/12/19/24) this decomposition targets -- octave errors are
  the dominant, quantified failure mode. A hierarchical decomposition
  changes what the *training objective* rewards (a chroma-head mistake
  and an octave-head mistake are scored/penalized independently), unlike
  the two mechanisms already tried for this exact failure mode: exp_035
  (added one new *feature* to the same flat 82-way objective, full CV
  topq -0.0051) and exp_036 (a *post-hoc decision rule* on top of the
  already-trained flat classifier's outputs, probe -0.088). Neither
  touched the objective/target structure itself, which is what this
  literature pattern actually changes.
- **Applicability**: Medium-High. The class-imbalance angle is a second,
  independent reason to expect this to help here specifically: 82 raw
  classes with 3-56 samples each collapses to 12 chroma classes (much
  better populated per class, since every octave's samples pool together)
  and a per-chroma octave count that is typically small (this dataset's
  ~82/12 ≈ 6.8 octave span). This directly addresses the extreme sparsity
  that model-family/hyperparameter tuning couldn't reach.
- **Experiment**: exp_058 (own) -- fold-honest probe: 2-head LGBM (or
  XGBoost, matching exp_042/sub_025's model family) on the existing grid
  features, head A = chroma (Pitch_ID % 12, 12-way), head B = octave
  (Pitch_ID // 12, ~7-way, conditioned on true chroma at train time /
  predicted chroma at inference), combined prediction = chroma*12+octave.
  Score against exp_042/sub_025's shift-aware topq metric on the same
  split. C9 flag: this is the 3rd attempt at exploiting exp_034's
  octave-confusion finding (after exp_035 feature, exp_036 decision-rule,
  both negative) -- but it is a materially different mechanism class
  (objective/target restructuring, not feature-add or post-hoc-rule), so
  treated as a fresh, not repeated, attempt per C9's "different approach"
  bar. If this also reverses/fails, the octave-confusion lever is fully
  closed (3/3 negative) and should not be revisited a 4th time.
- **Predicted impact**: uncertain magnitude given exp_035/036's mixed
  results on the same underlying pattern, but this is the only variant of
  the three that changes the actual learning objective, so probe first
  and do not promote to full CV without a clear probe-stage signal.
- **Risk**: added inference-pipeline complexity (two models instead of
  one); if chroma-head errors and octave-head errors are not independent
  in practice (e.g. a wrong chroma prediction usually drags the octave
  head wrong too, since features are shared), the decomposition may not
  actually decouple anything and could underperform the flat classifier
  that at least optimizes the joint target directly.

### Finding 5: prototypical networks / metric learning for extreme per-class imbalance -- rejected for now

- **Source**: [Prototypical Networks for Few-shot
  Learning](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning)
  and [Episodic fine-tuning prototypical networks for optimization-based
  few-shot learning: application to audio
  classification](https://ar5iv.labs.arxiv.org/html/2410.05302) (arxiv
  2410.05302)
- **Similarity**: metric-learning classification (distance to per-class
  prototype embeddings) explicitly designed for few-shot/imbalanced
  regimes, evaluated on audio tasks -- matches this competition's 3-56
  samples/class shape better than a generic 82-way softmax classifier in
  principle.
- **Approach**: learn an embedding function (typically a small NN), form
  each class's prototype as the mean embedding of its support examples,
  classify a query by nearest prototype (softmax over negative distances).
- **Score**: not benchmarked on a directly comparable task (standard
  few-shot benchmarks use held-out *novel* classes at test time; here all
  82 classes are seen during training, which is a closed-set problem, not
  true few-shot generalization -- a materially different setting from what
  these papers evaluate).
- **Implementation difficulty**: Medium -- would need a differentiable
  embedding network (the existing MLP family could be repurposed as the
  encoder), episodic training loop, and prototype-based inference, none of
  which exists in this project's pipeline yet.
- **Mechanism hypothesis**: prototype-based classification could be more
  sample-efficient per class than the current softmax/GBDT split-finding
  approach, since a prototype is just a mean embedding rather than a
  learned per-class decision boundary.
- **Applicability**: Low-Medium, and lower priority than Finding 4. The
  closed-set-vs-true-few-shot mismatch above is a real difference from the
  literature this pattern comes from; more importantly, this competition's
  MLP-family experiments (exp_016/017/019 and the whole blend family) are
  already the most shift-fragile results measured here (sub_022/024, LB
  -0.02 to -0.04 vs LGBM-alone despite better CV) -- a from-scratch NN
  encoder is exactly the model family this dataset has already punished on
  the private/public split, for reasons never fully explained (exp_020's
  clean adversarial-quartile-gap check did not predict the sub_022 drop).
- **Experiment**: none queued yet -- rejected as lower-priority than
  Finding 4 given the higher implementation cost and the standing
  shift-fragility evidence against NN-encoder approaches on this dataset.
  Revisit only if Finding 4 (hierarchical decomposition) also fails and a
  fresh mechanism family is needed again.
- **Predicted impact**: n/a (not queued).
- **Risk**: NN-encoder shift-fragility (see Applicability), closed-set
  mismatch with the literature's actual evaluation setting, and highest
  implementation cost of any candidate considered this pass.

## Recommended priority order (block 33)

1. Let exp_056 (iterative self-training, scheduler id exp_055) and exp_057
   (threshold fine-tune, scheduler id exp_056) finish -- both launched
   block 32, still running. Record both via `scheduler.py record` once
   `total time:` lands in their logs.
2. exp_058 (hierarchical chroma+octave decomposition, Finding 4) is the one
   fresh, actionable lever this pass found -- queue via `scheduler.py add`,
   probe once a CPU core frees up. Treat as the 3rd and decisive attempt
   on exp_034's octave-confusion finding (C9): if it also fails, close the
   lever for good.
3. Finding 5 (prototypical networks) explicitly rejected pre-probe --
   higher implementation cost, closed-set/few-shot literature mismatch,
   and this dataset has already shown NN-encoder approaches are the most
   shift-fragile family tried. Revisit only if exp_058 also fails.

## Research pass (block 35, 2026-07-07): all block-33 candidates now closed, fresh mechanism needed

exp_058 (hierarchical chroma+octave decomposition) landed negative (topq flat
-0.0000) since the last pass, and the confidence-threshold sweep on
pseudo-labeling is now bracketed (0.75 confirmed peak at full CV +0.0069,
0.60/0.65/0.70 all worse). Per STRATEGY.md's own note, the open-axis well was
close to dry: feature families, model families, ensembling, hyperparameters,
domain adaptation (covariate-shift alignment: CORAL/diagonal, 2/2 negative),
imbalance handling (class_weight/focal-loss, negative), and most of
pseudo-labeling are all closed.

### Finding 6: EM-based label-prior-shift correction (Saerens, Latinne &
Decaestecker 2002)

- **Source**: [Adjusting the Outputs of a Classifier to New a Priori
  Probabilities: A Simple Procedure](https://www.researchgate.net/publication/11608620_Adjusting_the_Outputs_of_a_Classifier_to_New_a_Priori_Probabilities_A_Simple_Procedure)
  (Saerens et al., Neural Computation 14(1), 2002); modern robustness
  follow-up: [Maximum Likelihood with Bias-Corrected Calibration is
  Hard-to-Beat at Label Shift Adaptation](https://arxiv.org/pdf/1901.06852)
  (Alexandari et al., 2019/2021) -- confirms the EM procedure remains
  competitive vs newer methods and that calibrating the base classifier first
  improves it further.
- **Similarity**: this competition has 82 severely imbalanced classes (train
  3-56/class, test ~7/class average over 583 rows) and a confirmed train/test
  distribution shift (adversarial AUC 0.97, exp_003/exp_006). Every domain-
  adaptation attempt so far (CORAL, diagonal alignment) targeted *covariate*
  shift in the feature space and failed. Label-*prior* shift -- the test set's
  class mix differing from train's -- is a distinct, untested mechanism: with
  only ~7 test rows/class on average, sampling noise alone could shift the
  realized test class proportions materially away from train's, on top of any
  systematic curation difference.
- **Approach**: classic EM procedure -- initialize test-set class-prior
  estimate at the train prior, then alternate (a) re-weighting each test row's
  posterior by test_prior/train_prior ratio and renormalizing, (b) updating
  the test-prior estimate as the mean of the re-weighted posteriors, until
  convergence (typically <10 iterations). Requires only the base classifier's
  predict_proba output -- no retraining.
- **Score**: not benchmarked on a directly comparable task; the 2019 follow-up
  reports consistent accuracy gains across label-shift benchmarks when the
  base classifier is reasonably calibrated.
- **Implementation difficulty**: Low -- ~20 lines, operates purely on
  existing XGBoost predict_proba output.
- **Mechanism hypothesis**: if the real unlabeled test set's true class
  distribution differs from train's (plausible given the small per-class test
  counts and the already-confirmed shift), correcting the posterior toward
  the test prior should recover accuracy the base classifier loses by
  implicitly assuming the train prior applies unchanged.
- **Applicability**: Medium-High. Cheap, orthogonal to every mechanism tried
  so far (not covariate-shift alignment, not train-time reweighting, not
  pseudo-labeling), and testable fold-safely by running EM only on each
  fold's real-test predict_proba (never on labels), mirroring the existing
  pseudo-labeling harness's fold-safety pattern.
- **Experiment**: exp_060 (scheduler id exp_059) -- queued via `scheduler.py
  add`. Probe: compute EM-estimated test-class-prior once (fold-safe, no
  labels touched) on the exp_042/sub_025 XGBoost+aug model's real-test
  predict_proba, compare to the train prior (KL divergence / max ratio) to
  check there's room for this to matter before spending a full CV run; if
  material, apply the correction inside the existing 3-fold CV harness and
  measure topq delta vs exp_056's current best (thresh=0.75 pseudo-labeling,
  +0.0069).
- **Predicted impact**: +0.01 topq (speculative -- first empirical check is
  whether the estimated test prior actually diverges from train's; if it
  doesn't, this closes immediately at near-zero cost).
- **Risk**: if the shift is purely covariate (loudness/noise-floor, per
  exp_003/exp_004) and not label-prior, EM will converge back to ~train prior
  and this is a fast, cheap null result, not a wasted full-CV run. Also:
  can compose with (not replace) the pseudo-labeling win, since it acts on
  posteriors at inference/pseudo-label-generation time.

### Rejected without queuing

- **Prototypical networks / few-shot metric learning** (revisited from block
  33's Finding 5): still rejected. No new evidence changes the prior verdict
  -- NN-encoder approaches remain the most shift-fragile model family
  measured on this dataset (sub_022/024 blend family), and the closed-set-vs-
  true-few-shot literature mismatch stands.
- **Self-distillation / soft pseudo-labels** (using predicted probabilities
  as soft targets instead of hard argmax labels for pseudo-labeling):
  considered as a variant of the already-closed pseudo-labeling family. Not
  queued separately -- exp_060's EM prior correction is more likely to
  compose with the existing hard-label pseudo-labeling win than to replace it
  with a different labeling mechanism, and the threshold-sweep axis (which
  soft-labeling would also interact with) is already closed per exp_056-059.

## Recommended priority order (block 35)

1. exp_060 (scheduler id exp_059, EM prior-shift correction) is the one
   fresh, orthogonal mechanism this pass found -- cheap probe first (check if
   the estimated test prior actually diverges from train's before spending
   full-CV budget).
2. If exp_060 is negative or near-zero (EM prior ~= train prior, i.e. the
   shift is purely covariate not label-prior), this closes the domain-
   adaptation family for good (3/3 negative: CORAL, diagonal alignment,
   EM prior-shift) and the next research pass needs to look outside standard
   ML domain-adaptation/imbalance/ensembling patterns entirely -- possibly
   revisit prototypical networks despite the shift-fragility concern, or
   accept exp_042/sub_025 (+ exp_056's thresh=0.75 pseudo-labeling once
   composed) as close to a local ceiling for this feature family.

## Result (block 36, 2026-07-07): exp_060/exp_059 landed decisively negative -- domain-adaptation family closed 3/3

Probe logs (`exp060_probe.log`, `exp060_topq_probe.log`, run prior to this
block, recorded into the scheduler this block): the EM-estimated real-test
class prior *does* diverge materially from the train prior (KL 0.30, L1 0.46,
one class with a 41x ratio) -- so the earlier "check if there's room for this
to matter" gate passed. But applying the EM correction to validation
posteriors made accuracy much worse, not better: plain 0.9442->0.8133,
**topq 0.9573->0.7949, delta -0.1624** (`scheduler.py record --id exp_059
--stage probe --delta -0.1624`, killed at probe cost per C4/C9). Reading: a
diverging EM-estimated prior is necessary but not sufficient for the
correction to help -- with only ~7 test rows/class average and many classes
in the 0-3 range, the EM prior estimate itself is too noisy to trust, so
"correcting" toward it moves posteriors in the wrong direction more often
than the right one. This is a distinct failure mode from CORAL/diagonal's
null results (those failed because tree monotonic-invariance made them no-ops
or measurement artifacts; this one is a real, harmful update).

**Domain-adaptation / distribution-correction mechanism family is now closed
3/3 negative** (CORAL, diagonal alignment, EM label-prior-shift) with three
distinct failure mechanisms, not one repeated one. Per the block-35 priority
list, the next research pass needs a genuinely different family --
candidates worth a fresh look: (a) prototypical/metric-learning networks
despite shift-fragility concerns (untested at the *grid-feature* input level,
only tested as raw-audio encoders so far -- distinct mechanism), or (b)
treat exp_042 + exp_056's thresh=0.75 pseudo-labeling composition as the
practical ceiling for this feature family and shift remaining budget to
finalizing/submitting that combination rather than continuing to search for
another lever. Recommend (b) for the next block unless exp_058's full CV
(thresh=0.60 pseudo-labeling, probe +0.0085, running as this block ends)
beats exp_056's +0.0069.

## Research pass, block 37 (2026-07-07): exp_058 full CV confirmed as new best; queued one fresh mechanism (cross-model agreement pseudo-labeling)

**exp_058 full CV landed while this block started** (`full_cv_exp058.log`,
recorded via `scheduler.py record --id exp_058 --stage full --delta 0.0137`):
thresh=0.60 pseudo-labeling beats thresh=0.75 (exp_056, +0.0069) by ~2x at
full 3-fold CV (plain +0.0009, weighted +0.0180, **topq +0.0137**), resolving
block 36's open question and reversing the probe-scale non-monotonicity
(exp_059's probe had thresh=0.75 probe +0.0171 > thresh=0.60 probe +0.0085 --
another instance of probe-vs-full-CV sign/rank flips this competition has
seen repeatedly, e.g. exp_010/016/035). This is now the best full-CV-confirmed
lever since sub_025/exp_042 and clears the coordinator's shift-aware gate
(PLAN.md note #2) against the sub_025-equivalent baseline. Full-train fit
launched this block (`fit_full_exp058_pseudolabel_thresh060.py`).

### Finding 7: cross-model agreement-gated pseudo-labeling (co-training pattern)

- **Source**: arxiv 2510.07509, "Efficient Generalization via Multimodal
  Co-Training under Data Scarcity and Distribution Shift" (logged via
  `writeup.py log --agent claude --url https://arxiv.org/abs/2510.07509
  --kind paper`).
- **Similarity**: exact match on both stressors -- data scarcity (2330
  train rows / 82 classes, 3-56/class) and confirmed train/test distribution
  shift (adversarial AUC 0.97+, exp_003/006). The paper's setting (limited
  labeled data + shift) is this competition's setting.
- **Approach**: dual-threshold pseudo-labeling gated by cross-view/cross-model
  AGREEMENT (not a single model's own confidence), with a PAC-style bound
  showing the benefit scales with inter-view agreement and view independence.
- **Score**: theoretical/synthetic-benchmark paper, no Kaggle-comparable score.
- **Implementation difficulty**: Low -- reuses the exact exp_051-059 harness,
  just trains a second stage-1 model (LGBM alongside XGBoost) and ANDs their
  agreement+confidence into the pseudo-label mask.
- **Mechanism hypothesis**: two independently-trained models with different
  inductive biases (leaf-wise vs level-wise GBDT growth, already established
  as meaningfully different on this dataset via exp_040) are less likely to
  agree on a WRONG label than either is individually confident-but-wrong, so
  agreement-gating should raise pseudo-label precision above single-model
  confidence thresholding at a comparable pseudo-label yield.
- **Applicability**: HIGH -- but note this is NOT the same thing as this
  competition's twice-failed "blend two models' final predictions" pattern
  (exp_017/044, closed 2/2 negative). Here agreement only filters WHICH real
  test rows get pseudo-labeled; the final stage-2 model that actually predicts
  submission labels stays XGBoost-alone, never a blend. Worth testing despite
  the blend family's track record because the failure mode is different
  (blending final probabilities dilutes toward the weaker model, per exp_031;
  gating pseudo-label inclusion by agreement doesn't touch final prediction
  at all).
- **Experiment**: exp_061 (queued, `scheduler.py add`, priority 0.0267) --
  `probe_exp061_agreement_pseudolabel.py`, sweeps agree_threshold
  0.50/0.60/0.70 against exp_058/059's single-model thresh=0.60 topq
  reference (0.9658, probe scale), same harness/split/seed as exp_051-059.
- **Predicted impact**: modest (+0.005 to +0.01 topq if it works, per the
  paper's reported gains being incremental, not transformative) -- but a
  genuinely new mechanism (label-selection agreement, not covariate/prior
  correction) after the domain-adaptation family closed 3/3 this block.
- **Risk**: LGBM's own predictions on this dataset are known to disagree with
  XGBoost meaningfully (exp_040's per-fold accuracy gap, exp_044's blend
  test) -- agreement-gating could end up with too few pseudo-labeled rows to
  matter (low yield), which the probe's percentile/count logging will surface
  directly.

## Recommended priority order (block 37)

1. Confirm exp_058's full-train fit + submission decision (topq +0.0137,
   above sub_025-equivalent baseline) -- this is a real, full-CV-confirmed
   gate-clearing candidate, prioritize banking/submitting it this block.
2. Run exp_061 (cross-model agreement pseudo-labeling) probe -- cheap,
   genuinely new mechanism, reuses existing harness.
3. If exp_061 is negative too, the pseudo-labeling family will have 2 live
   sub-axes closed (iterative multi-round, EM-prior) and confidence-threshold
   fully explored (0.60 peak) -- next fresh-mechanism search should look
   outside self-training entirely (e.g. prototypical/metric-learning at the
   grid-feature input level, per block 36's option (a)).

## Research pass, block 38 (2026-07-07): shrinkage-regularized label-shift correction (exp_062), reopening the domain-adaptation family on its own stated terms

Both cores were occupied at block start by exp_061 (cross-model agreement
pseudo-labeling probe, launched block 37, still running -- left in place,
not duplicated). Found one genuinely admissible fresh angle: PLAN_DRAFT.md's
own EM-prior-shift closure note explicitly left a reopening condition ("do
not revisit without a mechanism that specifically handles small-sample prior
estimation"), which a plain re-run of EM would not satisfy but a
regularized/shrinkage variant does.

### Finding 8: shrinkage-regularized (Dirichlet/James-Stein) label-shift correction

- **Source**: FMAPLS (arxiv 2511.18615, "Bayesian-based Online Label Shift
  Estimation with Dynamic Dirichlet Priors") and RLLS (Azizzadenesheli et al.,
  arxiv 1903.09734, "Regularized Learning for Domain Adaptation under Label
  Shifts") -- both logged via `writeup.py log --agent claude`.
- **Similarity**: exact match on the failure mode exp_059/060 hit -- both
  papers exist specifically because raw EM/MLE label-shift estimation is
  unstable under severe class imbalance and small per-class sample counts
  (this competition: 82 classes, ~7 test rows/class average, many at 0-3).
  FMAPLS reports up to 40% lower KL divergence vs unregularized baselines
  under exactly these conditions.
- **Approach**: regularize the noisy per-class prior estimate toward a
  trusted reference (train prior) via a Dirichlet/shrinkage prior, rather
  than trusting the raw EM update fully.
- **Score**: no Kaggle-comparable score (methods papers); FMAPLS reports
  relative KL-divergence and downstream-accuracy improvements over
  unregularized baselines in its own benchmarks, not on this task.
- **Implementation difficulty**: Low -- exp_059/060's exact harness, only the
  final correction step changes (convex combination instead of a full swap).
- **Mechanism hypothesis**: exp_059/060 already established the shift is real
  (KL 0.30 vs train prior) but that trusting the raw EM estimate per class is
  actively harmful at this sample scale. A per-class shrinkage weight
  `lambda_c = n_c / (n_c + alpha)` (n_c = soft test-assigned count under the
  current model) lets classes with real evidence (large n_c) keep most of the
  EM correction while classes with almost no test-assigned mass fall back
  toward the trusted train prior -- directly targeting the diagnosed noise
  source instead of a different shift axis.
- **Applicability**: HIGH as a reopening test, but bounded expectations --
  this is testing whether exp_059's diagnosed failure mode (noise, not wrong
  direction) is fixable with regularization, not a new shift-detection
  mechanism. If alpha-shrinkage still can't recover a non-negative delta at
  any tested alpha, that's a stronger closure of the whole family than
  exp_059/060 alone (rules out "the estimator was just too aggressive" as an
  explanation).
- **Experiment**: exp_062 (queued via `scheduler.py add`, priority 0.0333) --
  `probe_exp062_shrinkage_prior_shift.py`, sweeps alpha in {1, 5, 20, 50}
  against exp_059/060's exact reference numbers (uncorrected topq=0.9573),
  includes an alpha=0 sanity check that must reproduce exp_059/060's -0.1624
  raw-EM result. Launched this block on the second core (exp_061 occupies
  the first).
- **Predicted impact**: +0.01 topq if shrinkage recovers a positive delta at
  some alpha; a clean non-positive result at all alphas (including a
  reproduced alpha=0 sanity check) closes domain-adaptation for good with no
  remaining reopening condition.
- **Risk**: low compute risk (single stage-1 fit, closed-form corrections,
  same harness already validated 3x this competition). Main risk is
  over-interpreting a small positive delta as robust when n_topq=117
  (probe-scale) -- would need a full 3-fold CV confirmation before any
  submission decision, per this competition's repeated probe/full
  rank-flip pattern (exp_010/016/035/058).

## Recommended priority order (block 38)

1. Let exp_061 (cross-model agreement pseudo-labeling) and exp_062
   (shrinkage-regularized prior-shift) both land -- box is fully saturated,
   nothing else to launch until a core frees up.
2. exp_058's full-train fit (thresh=0.60 pseudo-labeling, topq +0.0137 full
   CV) remains the top-priority submission candidate, banked pending the
   daily cap reset (claude 3/3 informal cap hit today per STRATEGY.md note
   #6 and ~13 prior blocks' consistent discipline) -- submit it (or a
   composition with whichever of exp_061/062 also lands positive) first
   thing once the cap resets at 2026-07-08 00:00 UTC.
3. If exp_062 is negative across the whole alpha sweep including a
   reproduced alpha=0 sanity check, close the domain-adaptation family for
   good with no further reopening conditions -- do not search for a 4th
   variant.

## Research pass, block 39 (2026-07-07) -- queue-refill while exp_061/062 occupy both CPUs

Context: exp_060 (scheduler id, cross-model agreement-gated pseudo-labeling)
landed threshold 0.5 (topq flat, -0.0000 vs no-pl) and threshold 0.6 (topq
+0.0085, exactly tied with the already-closed single-model@0.60 result) while
this pass ran; threshold 0.7 still in flight. exp_061 (scheduler id,
shrinkage-regularized EM prior correction) still fitting its stage-1 XGBoost.
Both CPUs saturated, so used the window to look for a mechanism family that
hasn't been tried yet rather than another pseudo-labeling-threshold variant,
per block 33's note that the well is close to dry.

Queries used:
- `graph-based label propagation semi-supervised classification small
  labeled large unlabeled tabular distribution shift` (web search)
- `k-nearest neighbor classifier distribution shift robust vs gradient
  boosting small imbalanced dataset 2024 2025` (web search, checking whether
  a distance-based model family is worth adding for C7 diversity -- inconclusive,
  no direct comparison found, not pursued further as its own lever)

### Finding: graph-based label propagation (sklearn LabelSpreading) as a pseudo-labeling mechanism with a different inductive bias

- **Source**: [Label propagation algorithm](https://en.wikipedia.org/wiki/Label_propagation_algorithm);
  corroborating context from [Distribution Consistency based Self-Training
  for GNNs with Sparse Labels](https://arxiv.org/pdf/2401.10394) (arxiv
  2401.10394) on label propagation under sparse-label distribution shift.
- **Similarity**: same problem shape (few labeled train rows/class, larger
  unlabeled real-test pool, measured covariate shift) but a structurally
  different mechanism than anything tried this competition.
- **Approach**: build a kNN-affinity graph over train+real-unlabeled-test
  feature vectors jointly, diffuse train labels across the graph by local
  similarity (`sklearn.semi_supervised.LabelSpreading`/`LabelPropagation`)
  instead of using a classifier's own decision boundary at all.
- **Score**: n/a (general method, no competition-specific benchmark).
- **Implementation difficulty**: Low -- sklearn built-in, no custom training
  loop.
- **Mechanism hypothesis**: every pseudo-labeling/prior-correction mechanism
  tried so far (exp_051-060/061/062) still routes through a classifier
  trained only on train data, then asks "how much do I trust this
  classifier's output on test row X" (confidence threshold, cross-model
  agreement, or EM-reweighted prior). Label propagation instead asks "what
  do this row's nearest neighbors in the JOINT train+test feature manifold
  look like" -- no train-only decision boundary is involved in the
  propagation step at all. This sidesteps the specific failure modes already
  ruled out here: EM's noisy per-class prior estimate on thin classes
  (exp_059, topq -0.1624) and single/dual-classifier confidence
  miscalibration under shift (exp_051-060's whole threshold axis).
- **Applicability**: Medium -- genuinely untried mechanism family, but real
  risk: 14/82 classes have <10 train samples, so a kNN graph may not have
  enough same-class anchors nearby for those classes specifically (the same
  sparsity that has undermined every other lever tried on this dataset to
  some degree). Cheap enough (no retraining loop) that the risk is worth a
  probe rather than a pre-probe rejection.
- **Experiment**: queued as **exp_063** (scheduler id exp_062,
  `scheduler.py add`) -- fold-honest probe (same 80/20 holdout/split as
  exp_051-062), sweep `n_neighbors`/`gamma`, sanity-check per-class
  propagated-label agreement with train labels before trusting any topq
  delta, score against exp_042/sub_025's shift-aware topq reference.
- **Predicted impact**: +0.01 (uncertain sign given the rare-class-anchor
  risk above; genuinely new mechanism so no strong prior either way).
- **Risk**: rare-class kNN-anchor sparsity (see above); also graph
  construction cost scales with (train+test)^2 pairwise distances -- cheap
  at this dataset's scale (2330+583 rows) but worth confirming before a
  larger sweep.
- Second query (kNN vs GBDT shift-robustness) found no direct comparison
  specific enough to act on -- not pursued as a separate model-family-
  diversity lever; logged as an explicit non-finding, not a rejection of a
  concrete proposal.

## Recommended priority order (block 39)

1. Let exp_060 (agreement-gated pseudo-labeling, threshold=0.7 still running)
   and exp_061 (shrinkage-regularized EM prior correction) both land --
   record via scheduler.py regardless of sign.
2. exp_063 (graph-based label propagation, queued this pass) is the one
   genuinely new mechanism family found this block -- probe once a CPU
   frees up, with the rare-class-anchor sanity check before trusting any
   delta.
3. Submission cap: claude at 3/5 shared slots used today (2026-07-07) with
   exp_058's full-train fit (topq +0.0137) banked as the leading candidate
   for tomorrow's reset -- hold further submissions this calendar day per
   STRATEGY.md note #6 unless something clears exp_042/sub_025 by a wide
   margin.

## Research pass, block 40 (2026-07-07) -- exp_063 (label propagation) landed
negative, closing the pseudo-labeling mechanism-family search; one more
fresh-mechanism query before falling back to pure hyperparameter/ensemble
retreads

exp_063 landed both thresholds non-positive (thresh=0.5 topq -0.0086,
thresh=0.7 topq -0.0000) -- graph-based label propagation does not beat
confidence-threshold pseudo-labeling. Every inductive bias tried on the
pseudo-labeling axis (single-model confidence, cross-model agreement,
EM/shrinkage prior correction, graph propagation) is now closed except the
original confidence-threshold sweep itself (exp_058, thresh=0.60, full-CV
topq +0.0137). Searched for one more genuinely new mechanism before falling
back to retreading closed families.

### Finding 9: CMVN (cepstral mean/variance normalization) -- per-utterance
feature normalization for channel/recording mismatch

- **Source**: [Cepstral mean and variance normalization](https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization);
  [Modified Mean and Variance Normalization (IBM Research)](https://research.ibm.com/publications/modified-mean-and-variance-normalization-transforming-to-utterance-specific-estimates)
- **Similarity**: Standard speech-recognition fix for exactly the kind of
  shift exp_003/004 diagnosed here -- a per-recording channel/loudness/
  noise-floor mismatch between train and test recording conditions, not a
  labeling or class-distribution difference.
- **Approach**: normalize each utterance's own feature vector to zero mean,
  unit variance, using only that utterance's own statistics (no population
  stats from either train or test) -- removes recording-level channel
  differences per-clip instead of aligning dataset-level distributions.
- **Score**: n/a (technique reference, not a competition result).
- **Implementation difficulty**: Low -- pure row-wise transform, no fitting.
- **Mechanism hypothesis**: exp_003 found test clips have systematically
  ~1.5x higher spectral contrast and ~0.5x lower ZCR/chroma than train --
  consistent with a shared per-clip loudness/noise-floor offset spread
  across many feature dimensions. Row-normalizing each clip's own feature
  vector removes exactly this kind of shared per-clip level shift.
- **Applicability**: Medium-high on mechanism grounds, but with a real,
  pre-registered risk: our ~405 columns mix heterogeneous feature families
  (mfcc/chroma/contrast/zcr/tonnetz) with different natural scales and
  units, unlike CMVN's typical use on a single homogeneous cepstral-
  coefficient family. Z-scoring a row across all of them together is
  statistically unusual and could destroy real inter-family signal along
  with whatever it removes -- genuinely uncertain sign, worth a cheap probe
  rather than a prior-only judgment.
- **Why this isn't just CORAL/diagonal-alignment again**: exp_049's own
  notes proved column-wise affine rescaling (CORAL's diagonal special case)
  is mathematically neutral for tree models -- a monotonic per-column
  transform cannot change XGBoost/LightGBM histogram-split decisions. This
  is the opposite axis: a row-wise transform where each column's output
  depends on every other column's value in that row (via the row mean/std),
  which is not decomposable into independent per-column monotonic
  transforms and so is not covered by that invariance argument.
- **Experiment**: queued as **exp_064** (scheduler id exp_063,
  `scheduler.py add`) -- single 80/20 holdout probe (`probe_exp064_rownorm.py`,
  same split/seed/augmentation/model as exp_048-063), scored against the
  same plain/weighted/topq/rare_acc reference. Launched this block on the
  CPU core freed by exp_063's completion.
- **Predicted impact**: +0.01 (genuinely uncertain sign given the
  heterogeneous-feature-family risk above).
- **Risk**: heterogeneous feature-unit mixing (see above); also a clip with
  near-constant feature values (std near 0) could blow up under row
  normalization -- clipped denominator to 1e-8 in the implementation as a
  safety floor, not expected to bind in practice given ~405 real-valued
  spectral features per clip.

## 2026-07-07 block 38 -- ordinal-structure mechanism family (new)

### Source: Frank & Hall (2001) ordinal binary decomposition; CORAL/CORN
rank-consistent ordinal regression (Cao et al. 2019 / Raschka-research-group)

- **Similarity**: task type match at the label-structure level, not the
  domain level -- the competition's `Pitch_ID` target is not an arbitrary
  category id. Verified directly: `train.csv`'s `Pitch_ID` spans exactly the
  contiguous integer range 0..81 (82 values, all present), and exp_005's own
  grid-feature extractor (`extract_grid_features.py`) builds its 81 harmonic-
  evidence candidates from the MIDI-to-frequency formula
  `f0 = 440 * 2**((d-69)/12)` over a contiguous note range. Every mechanism
  family tried so far (model family, ensembling, feature-alignment,
  pseudo-labeling, hyperparameters, imbalance handling) has treated `Pitch_ID`
  as a nominal multiclass target where every pair of classes is equally
  "different" -- none has used the fact that class 40 and class 41 are one
  semitone apart while class 40 and class 80 are far apart.
- **Approach**: (1) CORAL/CORN -- deep-learning rank-consistent ordinal
  regression via K-1 weight-shared (CORAL) or conditionally-trained (CORN)
  binary "is y > k" classifiers, guarantees monotonic class probabilities.
  (2) Frank & Hall (2001) -- the classical, model-agnostic version of the same
  idea: train K-1 independent binary classifiers "is y > k" with *any* base
  learner (works directly with XGBoost/LGBM, no deep net needed), combine via
  P(y=k) = P(y>k-1) - P(y>k). (3) Simplest of all: plain regression on the
  integer-encoded label, round-and-clip to the valid class range.
- **Score**: not competition-specific; these are general ordinal-classification
  techniques, benchmarked on age-estimation/rating-prediction datasets in the
  source papers, not on pitch/audio tasks.
- **Implementation difficulty**: Low for (3) regression-and-round (drop-in
  objective swap on the existing harness). Medium for (2) Frank & Hall (K-1
  extra XGBoost fits, ~81x the single-fit cost -- expensive on this 2-core
  box, only worth it if (3) shows signal). High for (1) CORAL/CORN (needs a
  from-scratch small neural net) -- rejected pre-probe for the same
  small-data-overfit reasoning as the CREPE/HarmoF0 rejection above (2330
  clips / 82 classes is far below what these methods were validated on, and
  we have zero prior evidence a from-scratch net beats GBTs on this feature
  set at this scale).
- **Mechanism hypothesis**: severe class imbalance (14/82 classes <10
  samples) means many classes don't have enough rows to learn an independent
  nominal decision boundary; a regression/ordinal objective lets gradient
  signal from neighboring-pitch rows (which share correlated grid-harmonic
  evidence, since a true pitch's evidence leaks into adjacent MIDI-candidate
  columns by construction) reinforce each other, effectively borrowing
  statistical strength across nearby classes.
- **Applicability**: Medium-High for (3), untested; Medium for (2) contingent
  on (3); Low for (1), rejected pre-probe.
- **Experiment**: queued as **exp_065** (scheduler id, `scheduler.py add`) --
  `probe_exp066_ordinal_regression.py`, same 80/20 holdout split/seed/
  augmentation/model harness as exp_048-065 for direct topq comparability.
  Plain XGBoost regression (`reg:squarederror`) on the ordinal-encoded label,
  round+clip predictions to [0, 81]. Launched this block on the second CPU
  core while exp_065 (Sinkhorn pseudo-label, scheduler id exp_064) finishes
  its refit stage.
- **Predicted impact**: +0.01 (genuinely uncertain sign -- risk below).
- **Risk**: squared-error regression assumes a *linear* penalty in
  pitch-index units, which need not match the actual error cost. Octave
  confusions (a true pitch's evidence pattern repeating at +/-12 index
  positions) would be scored as a *large* regression error despite being a
  musically/acoustically natural confusion -- if octave-confusion is a
  meaningful share of the errors, plain regression could actively hurt vs
  nominal classification even though the ordinal *labeling* structure is
  real. If exp_065's raw MAE (logged pre-round) is low but rounded accuracy
  doesn't follow, that is exactly this failure mode and rules out (2)
  Frank & Hall too without a further probe.

## Recommended priority order (block 40)

1. exp_064 (row-wise CMVN-style normalization, queued this pass) is the one
   fresh mechanism found this block -- record via scheduler.py regardless of
   sign once the probe (launched this block) lands.
2. If exp_064 is also negative, the feature-transform/domain-adaptation and
   pseudo-labeling wells are both essentially dry for this competition's
   remaining CPU-bound search; next research pass should look specifically
   at whether any *combination* of already-confirmed-positive levers (aug +
   thresh=0.60 pseudo-label, already composing; XGBoost-alone) has an
   unexplored interaction, rather than a fifth mechanism family.
3. Submission cap unchanged: claude at 3/5 shared slots used today
   (2026-07-07), exp_058's full-train fit (topq +0.0137) remains banked as
   the leading candidate for tomorrow's first slot -- hold further
   submissions this calendar day per STRATEGY.md note #6.
4. exp_065 (ordinal regression on Pitch_ID, block 38 pass) is a genuinely new
   mechanism family (label structure, not feature/model/ensemble/domain-
   adaptation) -- record via scheduler.py once the probe lands. If positive,
   follow up with Frank & Hall binary decomposition (K-1 XGBoost fits); if
   MAE is low but rounded accuracy isn't, that specifically implicates
   octave-confusion as the failure mode and rules out both (2) and (3)
   without a separate probe.

## Research pass, block 41 (2026-07-07) -- ordinal-structure axis closed, exp_064/065 landed negative, fresh mechanism search

exp_064 (scheduler id, Confident Sinkhorn Allocation) landed: Sinkhorn-reassigned
labels are flat vs no-PL baseline but strictly worse than the established
argmax-threshold mechanism (-0.0171 at thresh=0.75, -0.0085 at thresh=0.60).
Pseudo-label assignment axis now closed 6/6 non-positive variants (confidence
threshold=argmax is still the only positive one, exp_058). exp_065 (ordinal
regression on Pitch_ID) landed catastrophically negative (topq -0.9060) --
closes the ordinal-structure axis opened last block; raw MAE=7.838 despite a
tight 82-class ordinal range confirms neighboring pitch classes do NOT share
enough gradient signal for regression smoothness to help (rules out Frank &
Hall binary decomposition too, per the pre-registered risk note -- no need for
a separate probe). All of feature-transform, model-family (nominal), ensembling,
hyperparameters (coarse grid), domain-adaptation, pseudo-labeling, and ordinal-
structure are now closed. Searched specifically for mechanisms matching this
dataset's defining trait -- severe few-shot imbalance (14/82 classes <10
samples, most classes far under LightGBM's own default split-blocking
threshold) -- rather than another generic retread.

### Finding 10: GBDT few-shot hyperparameter regime (min_data_in_leaf blocker)

- **Source**: [Gradient Boosting Trees and Large Language Models for Tabular Data Few-Shot Learning](https://arxiv.org/html/2411.04324v1)
- **Similarity**: Directly measures GBDT performance in the few-shot-per-class
  regime (few-shot tabular classification), the same regime this competition's
  82-class/3-56-samples-per-class label distribution sits in for its 14
  thinnest classes.
- **Approach**: identifies that LightGBM's default `min_data_in_leaf=20`
  silently blocks node splitting whenever fewer than 20 training rows would
  land in a child node -- exactly the case for any class with <20 samples.
  Their fix: `min_data_in_leaf=1`, `num_leaves=4`, `extra_trees=True`,
  `feature_fraction=0.5`, `bagging_fraction=0.5`. Reports 290% performance
  recovery vs LightGBM's stalled default in a matching few-shot regime.
- **Score**: not competition-specific (general FSL-tabular benchmark).
- **Implementation difficulty**: Low -- pure hyperparameter change, no new
  training loop, reuses the existing LightGBM path already in the codebase
  (LightGBM lost to XGBoost in exp_040, but with defaults untouched).
- **Mechanism hypothesis**: this is NOT a re-test of the closed model-family
  axis. exp_040's LGBM-vs-XGBoost comparison used LightGBM's stock defaults;
  nobody has tested whether LightGBM specifically stalls on this dataset's
  thin classes the way the paper describes, independent of XGBoost's own
  (already-permissive, min_child_weight=1 default) behavior. If the paper's
  diagnosis transfers, LightGBM under-performed XGBoost here for a mechanical
  reason (leaf-count blocking) rather than a genuine inductive-bias gap, and
  the few-shot-tuned config could reopen the model-family axis or produce a
  usefully-diverse blend component for the 14 rare classes specifically.
- **Applicability**: Medium-High -- directly matches the failure mode
  description, cheap to test, orthogonal to every closed axis (feature-
  transform, ensembling, XGBoost-only hyperparameter sweep, domain-adaptation,
  pseudo-labeling, ordinal-structure).
- **Experiment**: queued as **exp_067** (scheduler id exp_066) via
  `scheduler.py add`. Single 80/20 holdout probe, same harness/split/
  augmentation as exp_048-066, scored on plain/weighted/topq/rare_acc.
- **Predicted impact**: +0.015 (rare_acc specifically, since the mechanism
  targets exactly the thin-class leaf-splitting blocker).
- **Risk**: `num_leaves=4` is a large capacity cut for the non-rare classes
  (the LGBM capacity-tuning axis, exp_010/012, already found num_leaves
  tuning to be a dead lever in the *opposite* direction -- more capacity
  didn't help there, but that sweep never combined it with
  `min_data_in_leaf=1`, so the interaction is untested). Also risk of
  overfitting on the few-shot classes specifically (single-digit-sample
  leaves memorize noise) -- topq/rare_acc split will show if this trades off
  badly against the majority classes.

### Finding 11: Nearest-class-mean / prototypical classifiers for few-shot imbalance

- **Source**: [Prototypical Networks for Few-shot Learning, Snell et al. 2017](https://arxiv.org/abs/1703.05175);
  audio-domain corroboration: [Episodic fine-tuning prototypical networks for audio classification](https://arxiv.org/pdf/2410.05302)
  (PDF fetch failed to extract readable text -- used only as a pointer that
  prototypical/metric-learning approaches are applied to audio classification
  specifically, not for mechanism detail).
- **Similarity**: task-structure match, not domain match -- prototypical
  networks are the standard baseline for exactly the <10-samples-per-class
  regime 14/82 of this competition's classes are in. The core claim (a class
  centroid is a well-defined, low-variance target even from very few samples,
  unlike a tree-based decision boundary which needs enough rows to find a
  good split) is architecture-agnostic; the deep-learned-embedding version is
  overkill for 2330 rows (same small-data-overfit reasoning that already
  rejected CREPE/HarmoF0/CORAL-ordinal-net), but the classical nearest-class-
  mean classifier (no learned embedding, just distance to each class's mean
  feature vector) is a direct, cheap analog available off-the-shelf via
  `sklearn.neighbors.NearestCentroid`.
- **Approach**: classify each query by nearest Euclidean (or shrunk) distance
  to each class's centroid in (optionally PCA-reduced) feature space, instead
  of a GBDT's axis-aligned split-based decision boundary.
- **Score**: n/a (general technique reference).
- **Implementation difficulty**: Low -- `NearestCentroid` is a single
  scikit-learn call, no training loop, seconds to fit on 2330 rows.
- **Mechanism hypothesis**: every closed-axis model tried so far (LGBM,
  XGBoost, CatBoost, MLP) builds a decision boundary that needs enough
  same-class rows to distinguish a region of feature space -- weak for
  1-9-sample classes. A centroid is defined and stable even from a single
  sample, so this inductive bias should specifically help `rare_acc` (the
  probe harness's dedicated <10-sample-class metric) even if it's flat or
  negative on `plain`/majority-class accuracy.
- **Applicability**: Medium -- genuinely untested inductive bias family (no
  distance/metric-based classifier has been tried at all, only tree-split and
  MLP-decision-boundary models), but grid-harmonic features were engineered
  for tree-based split-finding (405 raw harmonic-evidence dims), not
  necessarily for a clean Euclidean-distance metric space -- PCA-reduction
  sweep in the probe will show if raw-dimensionality noise swamps the
  centroid signal.
- **Experiment**: queued as **exp_068** (scheduler id exp_067) via
  `scheduler.py add`. Single 80/20 holdout probe: `NearestCentroid` on
  standardized raw features vs PCA(n=20/50/100), same harness/split as
  exp_048-067.
- **Predicted impact**: +0.01, concentrated in `rare_acc`.
- **Risk**: if negative alone, cheap enough to still be worth testing as a
  rare-class-only blend component with XGBoost (route only the <10-sample
  classes' predictions through NearestCentroid) before closing the axis --
  do not spend a second full probe cycle on this without that composed
  variant first, per C9's 3-attempt budget.

## Recommended priority order (block 41)

1. exp_067 (LightGBM few-shot hyperparameters) -- highest predicted impact,
   lowest implementation risk, most directly targets a *named, paper-
   identified* mechanical blocker rather than a general "try a new model"
   guess.
2. exp_068 (NearestCentroid / prototypical) -- cheapest possible probe,
   run alongside exp_067 on the second core.
3. If both land negative, the "few-shot/thin-class-specific mechanism"
   family is closed too, and the next research pass should look at whether
   *any* remaining lever is purely a composition/interaction of
   already-positive levers (aug + pseudo-label thresh=0.60, currently the
   only two confirmed-positive full-CV levers) rather than a sixth fresh
   mechanism family -- the well is close to genuinely dry at that point.
4. Submission cap: claude at 3/3 daily slots used 2026-07-07 (resets
   2026-07-08 00:00 UTC) -- exp_058's full-train fit
   (`submission_exp058_xgb_augmented_pseudolabel060.csv`, topq +0.0137) is
   still the top-priority banked candidate for the first slot after reset.

## Research pass (block 38, 2026-07-07 ~19:3x UTC)

STRATEGY.md's own assessment: feature-transform, model-family, ensembling,
hyperparameter (LGBM+XGBoost), domain-adaptation, imbalance-reweighting
(class_weight/focal-loss/EM-prior-shift), pseudo-labeling acceptance-criterion,
ordinal-regression, and prototypical/centroid-classifier axes are all closed.
Searched for a genuinely fresh mechanism given this.

### Finding: SMOTE/mixup-style feature-space interpolation for extreme imbalance
- **Source**: [MixBoost (arxiv 2009.01571)](https://arxiv.org/pdf/2009.01571),
  general SMOTE-vs-mixup literature survey via WebSearch.
- **Claim**: for extreme per-class imbalance, synthesizing new samples by
  interpolating between same-class real samples IN FEATURE SPACE (SMOTE) or
  between different-class samples with soft labels (mixup) is a standard,
  distinct lever from noise-based augmentation.
- **Mechanism hypothesis**: this competition's rare-class problem (14/82
  classes <10 samples) has so far only been attacked via waveform-level
  noise+time-stretch augmentation (exp_018/033/037, positive, full-CV
  confirmed +0.0094-0.0137 stacked) and pseudo-labeling (exp_051-058,
  positive). SMOTE-style interpolation is mechanistically distinct: it
  densifies the CONVEX HULL of the existing feature manifold for rare
  classes rather than perturbing individual real samples with noise, and
  costs almost nothing (pure numpy on already-extracted 405-dim grid
  features, no audio re-processing).
- **Applicability check**: grep of `EXPERIMENTS.jsonl` for
  smote/adasyn/interpolat confirms zero prior attempts this competition --
  genuinely untried axis, not a re-test.
- **Experiment**: queued as **exp_070** (scheduler id, idea exp_071) via
  `scheduler.py add`. `probe_exp071_smote_feature_interp.py`: convex
  interpolation (t~U(0.2,0.8)) between same-class TRAIN-split pairs only
  (fold-safe), bringing each rare class up to ~12 rows. Three variants on
  the same 80/20 holdout harness as exp_048-070: SMOTE-only (replaces
  more_oversample), more_oversample-only (reference reproduction check),
  and stacked (more_oversample + SMOTE on top).
- **Predicted impact**: +0.015 topq, standalone or stacked.
- **Risk**: interpolating in a 405-dim engineered-feature space (harmonic-
  evidence scores per MIDI candidate, not a smooth learned embedding) may
  produce synthetic rows that don't correspond to any physically plausible
  audio clip -- unlike SMOTE's usual justification in cleaner/lower-dim
  feature spaces. If negative, this closes the last standard imbalance-
  handling lever and STRATEGY.md's dry-well assessment becomes final:
  future blocks should focus on composing/tuning already-positive levers
  (augmentation params, pseudo-label threshold) rather than searching for
  a 7th fresh mechanism family.

### Rejected: further pseudo-labeling / consistency-regularization variants
- Sources: arxiv 2408.07221 (pseudo-labeling review), arxiv 2401.04435
  (uncertainty-aware long-tailed SSL), arxiv 2403.12986 (BaCon contrastive
  balancing). All assume either much larger unlabeled pools or a
  deep-learning backbone with intermediate feature layers to apply
  consistency/contrastive losses to -- this competition's unlabeled pool is
  only 583 real test rows (already exploited via exp_051-058's threshold
  sweep) and the model family is GBDT (XGBoost), which has no intermediate
  representation to apply feature-level contrastive balancing to. Not
  queued; would require a from-scratch deep model, already rejected for
  this dataset size (CREPE/HarmoF0/PANNs research pass, block 3).

## 2026-07-08 (new block, dry-well research pass)

Trigger: STRATEGY.md's block-38 note flagged the axis well as close to dry --
exp_070 (SMOTE) and exp_068 (LGBM-leafsize reconciliation) were the only two
open threads, and both closed negative this block (exp_068 full CV -0.0051,
confirms exp_066's +0.0342 was single-holdout noise; exp_070/SMOTE already
closed exp071_probe.log block 38, -0.0086 all 3 variants). Per the mandate,
this pass looks for a genuinely new mechanism family rather than another
variant of a closed axis.

Queries used:
- "wavelet scattering transform audio classification small dataset pitch
  2025" (web search)
- "gammatone cepstral coefficients GFCC pitch classification missing
  fundamental" (web search -- no direct hits on missing-fundamental
  specifically; GFCC is a generic MFCC variant with a biologically-inspired
  filterbank, same family as the already-closed generic-spectral-feature
  axis (exp_002/004), not a new mechanism. Not queued.)
- "kymatio Scattering1D bioacoustic small dataset classification accuracy
  results" (web search)

### Source: Wavelet Scattering on the Pitch Spiral (arxiv 1601.00287, logged)
- **Similarity**: Directly about extracting pitch-class information via
  scattering coefficients, separating pitch dynamics from spectral-envelope/
  timbre -- same problem framing as this competition's harmonic-grid
  features, different mechanism (translation-invariant multiscale wavelet
  cascade vs explicit per-candidate-f0 harmonic/ACF/cepstrum scoring).
- **Approach**: Cascade wavelet convolutions + modulus operators
  successively over time, log-frequency, and octave index; theoretical paper
  (closed-form approximation for a nonstationary harmonic source-filter
  model), no empirical benchmark reported in the abstract.
- **Score**: N/A (theoretical, no reported task/dataset accuracy).
- **Implementation difficulty**: High for the full pitch-spiral-specific
  construction (custom octave/log-frequency convolution cascade). Low-Medium
  for a plain off-the-shelf 1D scattering transform (kymatio's
  `Scattering1D`) applied directly to the raw waveform, which captures the
  same translation-invariant multiscale mechanism without the paper's
  custom spiral geometry -- this is what got implemented, not the full
  paper.
- **Mechanism hypothesis**: Scattering coefficients are stable to small time
  warps/deformations and summarize energy across dyadic frequency/time
  scales, potentially capturing periodicity structure (and therefore missing-
  fundamental pitch cues) in a way that's robust to the recording-level
  loudness/noise-floor shift already characterized (exp_003/004) without
  needing explicit per-candidate harmonic templates.
- **Applicability**: Untested for this specific task; kymatio's own docs
  (`kymat.io/gallery_1d`) claim state-of-the-art results for supervised
  musical-instrument classification "in the setting of limited annotated
  training data" -- directly relevant given this competition's 82 classes /
  ~28 samples/class average.
- **Experiment**: queued as **exp_073** (scheduler id exp_072, own/research)
  via `scheduler.py add`. `extract_scattering_features.py`: kymatio
  `Scattering1D(J=8, Q=8)` on 8kHz-resampled, zero-padded-to-2**16-sample
  audio, mean+std pooled over the time axis -> 468 features/file (comparable
  scale to the 405-dim grid feature set). Full-dataset extraction (2913
  files, measured ~1.08s/file serial on this box) launched in background
  this block (`extract_scattering_full.log`), running concurrently with
  exp_071's full CV job (both single-threaded, box has 2 CPUs). Next block:
  probe via single 80/20 holdout XGBoost (same split/seed as exp_035-071) --
  scattering-alone vs grid-alone (topq baseline 0.9573), and scattering+grid
  concatenated (additive test, same pattern as exp_007's rejected generic+
  grid combination).
- **Predicted impact**: +0.02 topq (optimistic given "no monoculture" C7 and
  a genuinely different invariance mechanism than every feature variant
  tried so far; wide uncertainty since the paper itself has no empirical
  benchmark to anchor the estimate).
- **Risk**: (1) 8kHz downsampling implies a 4kHz Nyquist -- MIDI 110
  (~3951Hz) is near the edge, so higher harmonics of the highest-pitch
  classes get truncated; may need a higher SR (more compute) if initial
  results look promising but capped by this. (2) kymatio 0.3.0 has a
  scipy>=1.15 incompatibility (`scipy.special.sph_harm` removed) requiring a
  monkeypatch (`sph_harm_y` shim) to import -- narrow and load-bearing, but
  flagging in case a kymatio/scipy upgrade later removes the need or breaks
  it differently. (3) New package dependency (`kymatio`, pulled via `~/ml/bin/pip
  install kymatio`, not previously in the venv).

### Rejected: GFCC / gammatone-filterbank features
- Same family as the already-closed generic-spectral-feature axis
  (exp_002/004: MFCC/chroma/contrast -> LGBM, badly lost to grid-harmonic
  features, exp_007 confirmed combining generic+grid features hurts vs
  grid-only). GFCC is a biologically-motivated filterbank swap for MFCC, not
  a mechanistically different feature -- would very likely repeat exp_002's
  result. Not queued.

### 2026-07-08 (block, /day): revisiting exp_073's own flagged-but-untested risk
- **Source**: own re-read of `extract_scattering_features.py`'s docstring
  (exp_073), not a new external source -- this block's `/day` research pass
  found the scheduler queue empty with a free CPU core (exp_076 occupying
  the other), and rather than searching for a brand-new mechanism family
  (STRATEGY.md already assesses the well as close to dry), converted an
  already-flagged-but-never-tested risk into an actual experiment, per C3.
- **Similarity**: n/a (methodology check on our own existing feature, not an
  external pattern).
- **Approach**: exp_073's extraction script downsamples to 8kHz (4kHz
  Nyquist) purely for compute reasons, flagging "MIDI 110 (~3951Hz) is near
  the edge" without checking against real per-class frequency data or
  testing an alternative SR.
- **Mechanism hypothesis**: cross-checked gemini's `pitch_yin_mapping.csv`
  (YIN-detected median_freq per Pitch_ID) -- max is ~2005Hz (Pitch_ID 18),
  14/82 classes >1000Hz. A class with a ~1500-2000Hz perceived fundamental
  has 2nd/3rd harmonics at 3000-6000Hz, exactly the band 8kHz-SR scattering
  discards. If scattering's signal partly comes from harmonic energy in that
  band, raising SR to 16kHz (8kHz Nyquist) should disproportionately help
  the high-median-freq class tertile.
- **Applicability**: direct, uses this competition's own data/labels, not
  transferred from elsewhere.
- **Experiment**: queued as **exp_077** (scheduler id exp_075) via
  `scheduler.py add`. `probe_exp077_scattering_higher_sr.py`: re-extracts
  scattering at 16kHz (SC_N doubled to 2**17 to preserve window length) for
  a cheap stratified subsample (500 train/200 test, not a full
  re-extraction), compares grid+scattering-8kHz vs grid+scattering-16kHz on
  overall topq AND topq restricted to the high-median_freq tertile
  specifically (the subset the hypothesis predicts should move). Launched in
  background on the free core (exp_076 occupying the other) --
  `probe_exp077.log`.
- **Predicted impact**: +0.01 topq overall if real (larger on the
  high-freq-tertile-only metric); could also land at ~0 if scattering's
  useful signal is concentrated in the sub-4kHz band regardless (most
  classes' fundamentals are well under 2kHz per the mapping above).
- **Risk**: (1) `pitch_yin_mapping.csv` is YIN-detected and known-unreliable
  per its own std_midi column (some classes are 8-16+ semitones off) --
  using it only for a coarse tertile *ranking*, not as ground truth. (2) a
  500/200-row subsample at single-holdout fidelity is probe-only; per this
  competition's repeated probe/full reversal pattern (exp_010/012/016 etc.),
  any positive signal here needs a full-CV re-extraction+confirmation before
  being trusted, not a submission-worthy result on its own.

## 2026-07-08 (/day, new-mechanism search per STRATEGY.md block-113 mandate)

Trigger: STRATEGY.md flagged the scattering axis as 5/5 non-positive-beyond-
standalone (SR, log-compression, augmentation, pseudo-labeling all fail to
add on top of the standalone +0.0223) and said the next block needs a
genuinely new mechanism family. exp_075 (16kHz full-CV) was launched in the
background this block; this pass looked for what to queue next rather than
wait idle.

Queries used:
- "missing fundamental pitch perception virtual pitch algorithm feature
  machine learning classification small dataset 2025" (web search)
- "supervised contrastive metric learning small audio classification few
  samples per class 2025" (web search)

### Rejected: Terhardt/Goldstein virtual-pitch (psychoacoustic template) algorithms
- **Similarity**: Directly targets the missing-fundamental phenomenon this
  competition is named for.
- **Approach**: Weighted harmonic-template matching against a bank of
  candidate fundamentals, biologically motivated (auditory-filterbank
  weighting curves) rather than a plain harmonic-energy/ACF/cepstrum score.
- **Mechanism hypothesis**: could differ from the existing grid features by
  weighting harmonics non-uniformly (perceptual salience curves) instead of
  raw energy/ACF/cepstrum-lag scoring.
- **Applicability rejected**: this is the same mechanism family as the
  existing grid-harmonic features (exp_005: "81 candidate f0s x harmonic-
  energy+ACF+cepstrum-lag") -- a per-candidate-f0 harmonic template score,
  differing only in the weighting curve. Same class of idea as the already-
  rejected GFCC (a reweighted-filterbank variant of MFCC): a parametrization
  change within an already-dominant feature family, not a new invariance
  mechanism. Per C7, would very likely just perturb exp_005's grid features
  rather than add new signal. Not queued.

### Rejected: supervised contrastive / prototypical few-shot embedding learning
- **Similarity**: 2025 papers (arxiv 2509.10074, 2409.09647) target exactly
  this competition's regime -- small-N, few-shot, per-class audio
  classification.
- **Approach**: train a deep encoder with supervised-contrastive or
  prototypical loss, classify via nearest-centroid in the learned embedding
  space.
- **Score**: state-of-the-art on MetaAudio 5-way/5-shot benchmark (no
  absolute accuracy reported for a directly comparable 82-way task).
- **Applicability rejected**: this is a metric-learning/nearest-centroid
  classification mechanism -- exp_067/068 (Nearest-class-mean / prototypical
  classifiers on the existing grid features) already tested this decision
  rule and it failed catastrophically (topq -0.1881, killed at probe). The
  papers add a *learned embedding* on top of centroid classification, which
  is a bigger investment (train a deep net from scratch on ~2900 rows/82
  classes, CPU-only, no GPU on this box) for a decision rule already shown to
  underperform the existing GBDT+grid-features pipeline here. Frozen
  pretrained deep-audio embeddings (CREPE exp_028, PANNs exp_029) were also
  already rejected (negative) -- the domain mismatch that sank those likely
  also caps a from-scratch small-data contrastive encoder. Not queued.

**Conclusion**: no new mechanism family surfaced this pass beyond
reparametrizations of already-closed axes (harmonic-template weighting,
metric-learning/centroid classification). This confirms rather than
contradicts STRATEGY.md's dry-well read. Recommendation for next block:
finish evaluating exp_075 (16kHz scattering, full CV running in background)
and, if that also closes flat/negative, treat the feature/model-mechanism
search as exhausted for this competition and shift remaining effort to (a)
the flagged grid+scattering hedge-submission decision (mechanistically
diverse vs the banked XGBoost+augment+pseudolabel pipeline, per the
final-selection-by-public-lb memory card) and (b) tightening the existing
best pipeline's own hyperparameters/composition rather than new families.

## 2026-07-08 (/day, block 126) -- independent re-check after blend axis closed 6/6, no new lead

Trigger: PLAN.md's block-124/125 coordinator notes report the engineered-
feature + GBDT/ExtraTrees + ensembling paradigm closed 4-5x independently
(feature families 4/4, model families, hyperparameters, domain adaptation
5/5, imbalance handling, pseudo-labeling all sub-variants, ordinal/label-
propagation/centroid, pretrained embeddings 2/2, prediction-level blending
6/6 by every weight-selection method, from-scratch CNN closed on data-shape
grounds), and recommend either maintenance or waiting for direction absent
new gemini/codex activity or a genuinely new mechanism. Confirmed via
`scheduler.py next` (queue empty), `memory_cli.py retrieve --anchor stuck`
(no new candidate), and registry inspection (no submissions past sub_029,
no gemini/codex activity since sub_021) before running this pass.

Queries used (independent of the block-113 pass, checking for anything
missed):
- "Kaggle missing-fundamental-puzzle competition" (web search -- still no
  competition-specific hits, reconfirms block 2/27/113's finding a 4th time)
- "missing fundamental pitch perception classification deep learning
  few-shot small dataset" (web search)

### Result: no new mechanism family found; reconfirms exhaustion

Both hits (few-shot audio classification via prototypical networks /
transfer learning, and general few-shot-with-limited-data surveys) restate
mechanisms already tested and closed here: prototypical/metric-learning
classification (exp_067/068, topq -0.1881) and pretrained-embedding
transfer (CREPE exp_028 -0.0206, PANNs exp_029 -0.0137). No qualitatively
different lever surfaced. This is now the 4th independent research pass
(blocks 113, the block-113-followup, and this one) to search explicitly for
a new mechanism family and come back empty -- explicit rejection per C3,
not a silent drop.

**Conclusion**: no experiment queued this block. Forcing a new variant into
an axis already closed 4-6x (per C4/C9, that is not the smallest next test)
would be lower expected value than the maintenance work this block instead
did: cross-review check (no new gemini/codex submissions since sub_021),
sign-agreement/calibration check (unchanged, pearson=0.976, 62%/9, still
below trust bar since no new submission since sub_029), and lint (clean).
Current banked position (sub_027 CV-best public 0.95402 + sub_028
diverse-hedge public 0.94827) stands as the reasonable stopping point per
the block-124/125 coordinator recommendation. Next genuinely new lever
would need either (a) explicit human/coordinator direction, (b) new
gemini/codex activity to cross-review, or (c) a materially different data
regime (e.g. more rows/class) that isn't available in this competition.
