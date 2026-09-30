# Revision analyses (manuscript revision, 2026-09)

**Status:** PROTOCOL. Written on 2026-09-29, before any revision outcome was computed.
**Scope:** additional and re-run analyses for `00_manuscript/pannormal_scanner_batch_effects.tex`.
**Rule for results:** outcomes are recorded under `results/<analysis>/` and discussed first.
The manuscript, its tables, figures and `analysis/paper/` locks are **not** updated until
the results have been discussed.

IDs use the prefix `RV` so they cannot be confused with the historical E0–E9 experiments.

## Why these analyses

The manuscript argues that scanner correction has to be judged on three axes at once:
fidelity of the corrected image to the real paired target, alignment of PFM representations
with the real target, and preservation of tissue information. The current evidence covers
these axes unevenly: the third axis is measured only for a few methods, representation
alignment is summarized mainly as a distance, sample sizes differ between analyses, and
the external validation is confounded by acquisition magnification. The analyses below
close these gaps without changing the study question.

## Rules

1. **One evaluation set.** Every comparative evaluation and every perturbation probe uses
   the same 20 locations per held-out PanNormal slide (`results/location_sets/set20.csv`)
   and the same 2,387 PLISM locations. Fitting, selection and training use the 40-location
   set on training-fold slides (`set40.csv`), of which `set20` is a strict nested subset.
   Scanner and tissue characterization (RQ1) keeps using all matched locations.
2. **Pre-specified endpoints.** Each analysis names its primary endpoint and how each
   outcome will be read. All outcomes are reported whatever their direction.
3. **Deviations are logged**, with date and reason, in the Deviations section below.
4. **Layout** follows `analysis/paper/`: `<name>.py` with a matching `<name>.sbatch`
   (submitted from the repository root with `JOB_CONDA_PREFIX` and `JOB_WORKDIR`),
   outputs and logs under `results/<name>/`. Model weights, folds and cohort are the
   locked ones listed below.

## Shared definitions

| Item | Definition |
| --- | --- |
| Cohort | 103 PanNormal slides, 37 tissue types: `outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv` |
| Folds | five physical-slide folds: `outputs/augmentation_ood_v1/00_contract/slide_folds.csv` |
| `set40` | 40 locations per slide, first 4 by `source_index` in each of 10 spatial replicate groups: `outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/00_contract/locked_image_evaluation_index.csv.gz` |
| `set20` | first 2 per replicate group within `set40` (`uni_location_indices` in `src/scanner_batch_extensions.py`); 2,060 locations |
| Directions | AT2 (source) to VERSA, AKOYA, GT450, S360, S60 in PanNormal; AT2 to GT450, S360, S60 in PLISM |
| PLISM | 13 sections, 2,387 locations (at most 4 per section and core), corrections fitted in PanNormal and applied without refitting |
| PFMs | UNI v1 (primary), UNI2-h, Virchow2, H-optimus-1; frozen, 256×256 field of view resized as in the paper |
| Image methods | raw, Reinhard, Macenko, Vahadane, Pix2Pix, CycleGAN, frequency, color + frequency (Reinhard then frequency) |
| Feature methods | ridge affine (primary), ComBat, affine OLS (secondary); fitted on `set40` of training folds |
| Target distance | cosine distance between the (corrected) AT2 embedding and the real target-scanner embedding at the same location |
| Target gain | raw target distance minus corrected target distance; positive means closer to the real target |
| Statistics | means over slides with the five directions equally weighted; 95% CIs from 2,000 slide bootstrap resamples (PLISM: section bootstrap); paired differences use the same resamples |

## Overview

| ID | Question | Importance | Effort (work / compute) | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RV-P0a | Are the location sets fixed as files? | ★★★ | 0.5 d / minutes | — | done |
| RV-P0b | Do published image metrics change on `set20`? | ★★★ | 1 d / CPU minutes | P0a | done |
| RV-P0c | Are the two UNI extraction paths identical? | ★★★ | 0.5 d / CPU minutes | P0a | done |
| RV-P0d | Can every paper asset be regenerated? | ★★★ | 1 d / CPU | — | done |
| RV01 | Do the blur/sharpen findings hold on `set20`? | ★★ | 0.5–1 d / ~1.5 GPU-h | P0a | done |
| RV02 | Do band sensitivities hold on `set20` in four PFMs? | ★★ | 1 d / ~13 GPU-h | P0a | done |
| RV03 | Corrected-image embeddings in all four PFMs | ★★★ (enabling) | 1 d / ~5 GPU-h | P0a | done |
| RV04 | Three-axis evaluation of every method in every PFM | ★★★ | 2–3 d / CPU | RV03 | done |
| RV04b | Is the target scanner still detectable after correction when the classifier is independent of correction fitting, linearly and nonlinearly? | ★★★ | 0.5 d / CPU | RV03 | done |
| RV05 | Does representation alignment recover a downstream tissue task? | ★★★ | 1.5–2 d / CPU | RV03 | done |
| RV06 | Existing results consolidated with provenance | ★★★ | 0.5–1 d / none | — | done |
| RV07 | One summary figure of the three axes | ★★★ | 1 d / none | RV04, RV05 | done |
| RV08 | How large is the registration noise floor? | ★★ | 1 d / <1 GPU-h | P0a | done |
| RV09 | Do correction gains depend on tissue type? | ★★ | 1–1.5 d / CPU hours | RV03 | done |
| RV10 | Is the high-frequency phenotype robust to resampling? | ★★ (★★★ if the pipeline did not anti-alias) | 2–3 d / CPU | Stage A feasibility | done |
| RV11 | Does the scanner effect differ by tissue under the simplest model with scanner × tissue as a direct term? | ★★★ | 0.5 d / CPU minutes | — | done |
| RV12 | Can the frequency phenotype and its tissue dependence be shown directly (transfer curves, tissue × scanner map)? | ★★ | 0.5 d / CPU minutes | RV11 | done |
| RV13 | Why are PFMs scanner-sensitive: image-only pretraining, frequency reliance, or narrow data sources? Does stain-normalized pretraining (EXAONEPath) remove only the colour part? | ★★ | 1–1.5 d / ~8 GPU-h | RV02, RV03 | done |
| RV14 | Does a model's sensitivity to small spatial-frequency perturbations predict its robustness to real scanner differences? | ★★★ | 0.5 d / CPU | RV13 | done |

★★★ required for the revision · ★★ strongly recommended · ★ optional.

---

## RV-P0: consistency of the existing analyses

### RV-P0a Location sets

- **What:** write `set40.csv` and `set20.csv` (slide, tissue, fold, `location_index`,
  `source_index`, replicate group) and a summary with counts and hashes.
- **Checks:** 40 and 20 per slide; `set20` ⊂ `set40`; `set20` equals the locations stored in
  `outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards`.
- **Output:** `results/location_sets/`.

### RV-P0b Re-aggregation on `set20`

- **Why:** Table 2 mixes 40 locations (Pix2Pix/CycleGAN UNI distance, image residual,
  coverage and NCC) with 20 (everything else), and the augmentation-oracle percentages
  49.1% and 93.9% are on 40 locations.
- **What:** recompute on `set20` only: image residual, coverage, joint coverage, SSIM,
  LPIPS, target- and source-gradient NCC for every image method; UNI target distance for
  Pix2Pix and CycleGAN; augmentation-oracle residual, joint coverage and real-vs-oracle
  classifier accuracy.
- **Primary endpoint:** difference between the `set20` value and the published value for
  each number in Table 2 and in the augmentation text.
- **Expectation:** differences at or below the rounding shown in the paper; the conclusions
  unchanged. Any difference that changes a sign or a ranking is reported for discussion.
- **Output:** `results/set20_reaggregation/`.

### RV-P0c UNI extraction parity

- **Why:** feature correction is fitted on 40-location UNI embeddings from a separate
  extraction than the evaluation embeddings.
- **What:** cosine similarity between the two extractions for the raw six-scanner images at
  the shared `set20` locations.
- **Expectation:** ≥ 0.999 for every location. Anything lower is traced to its cause
  (resize, colour conversion, precision) before feature results are used.
- **Output:** `results/uni_extraction_parity/`.

### RV-P0d Provenance restoration

- **What:** recover or rewrite the missing producers of (i)
  `analysis/paper/results/scanner_color_frequency_space/pannormal_tissue_positions.csv`
  (Fig. 2), (ii) the PLISM GAN cross-PFM features, (iii) the GT450 edge-constraint
  structure ablation; show that each reproduces its frozen output.
- **Output:** restored code beside the related `analysis/paper/` code, after discussion;
  verification record in `results/provenance_restoration/`.

---

## RV01 Blur and sharpening trajectories on `set20`

- **Question:** do UNI blur/sharpen trajectories, the convergence of scanners under strong
  blur, and the accompanying loss of tissue information hold on the full evaluation set?
- **Why it matters:** Section 3.2 and Fig. 3 rest on three locations per slide.
- **Design:** the paper's operations and strengths (blur σ = 0, 0.5, 1, 2, 3, 6; unsharp
  α = 0, 0.25, 0.5, 1, 2), all six scanners, `set20`; PCA fitted to the 60 mean states as
  before, σ = 6 projected.
- **Primary endpoints:** median cosine similarity of displacement directions across scanner
  pairs (blur σ = 3, sharpen α = 2); median interscanner distance ratio at σ = 6; median
  retained gradient energy at σ = 6; same-tissue nearest-neighbour proportion at raw and
  σ = 6 (slide means over 20 locations).
- **Secondary endpoint:** six-way scanner linear probe on σ = 6 embeddings (slide-grouped
  cross-validation), to test whether blurred scanners remain decodable while distances
  shrink.
- **Expectation:** shared directions (median cosine > 0.5); distance ratio well below 1;
  retained gradient near 1%; lower tissue retrieval under strong blur. Retrieval percentages
  will differ from the paper because slide means now use 20 locations.
- **How we read it:** if any of the three qualitative statements fails, Section 3.2 is
  rewritten around the new result. A scanner probe that stays high under σ = 6 strengthens
  the statement that a shorter distance is not correction.
- **Output:** `results/blur_sharpen_set20/`.

## RV02 Frequency-band manipulation on `set20` in four PFMs

- **Question:** do UNI's stronger response to the high band, the absence of target gain from
  high-band manipulation, and the between-PFM differences in band weighting hold on the full
  evaluation set?
- **Design:** the paper's procedure (Reinhard colour match to each target; low–mid, mid and
  high bands; increase and decrease; doses 0.25 and 0.50; target direction from training
  folds) on `set20`, for UNI v1, UNI2-h, Virchow2 and H-optimus-1.
- **Primary endpoints:** high-minus-low–mid representation shift at dose 0.25 per PFM;
  mean additional target gain from high-band manipulation in the target direction.
- **Expectation:** positive high-minus-low–mid difference for UNI v1 and UNI2-h, negative for
  Virchow2 and H-optimus-1; high-band target gain near zero or negative.
- **How we read it:** a sign change with a CI excluding zero for any PFM changes the
  statement that PFMs weight bands differently for that PFM.
- **Output:** `results/band_manipulation_set20/`.

## RV03 Corrected-image embeddings in four PFMs

- **Question:** enabling step. Per-location embeddings for every image correction in every
  PFM, which RV04, RV05 and RV09 need.
- **Design:** PanNormal `set20` × five directions: regenerate Reinhard, frequency,
  colour + frequency, Macenko and Vahadane deterministically from the stored parameters and
  stain references; subset the stored Pix2Pix and CycleGAN images to `set20`. PLISM 2,387
  locations × three directions: regenerate the conventional corrections with the
  PanNormal-fitted parameters; use the stored GAN features. Embed with UNI v1, UNI2-h,
  Virchow2 and H-optimus-1. Raw source and target embeddings on `set20` come from the stored
  40-location raw embeddings.
- **Quality checks:** UNI v1 embeddings of regenerated images match the stored ones
  (cosine ≥ 0.999); target distances reproduce Table 3 within rounding.
- **Output:** `results/corrected_embeddings/` (embeddings kept as shards; not locked).

## RV04 Three-axis evaluation of every method in every PFM

- **Question:** does any correction improve image fidelity, representation alignment and
  tissue preservation together, and does a smaller target distance mean that the target
  scanner is no longer detectable?
- **Design:** every image and feature method, four PFMs, PanNormal (`set20`, five
  directions) and PLISM (three directions).
  - *Image fidelity:* from RV-P0b (PanNormal) and the existing PLISM benchmark.
  - *Representation alignment:* target distance and gain; **normalized target distance**
    (target distance divided by the mean between-tissue distance among the same condition's
    embeddings); **spread ratio** (between-tissue distance of the condition over that of the
    real targets), which detects shrinkage of the representation space.
  - *Target detectability:* balanced accuracy of a logistic-regression classifier separating
    real target embeddings from corrected embeddings, five-fold cross-validation grouped by
    slide, pooled over target scanners. This is the same estimand as the existing UNI v1
    analysis in `03_uni`; chance is 0.5.
  - *Tissue preservation:* slide means over `set20`; the query is the corrected slide mean;
    the reference bank is the real target-scanner slide means of other slides; same-tissue
    top-1 retrieval, macro recall over 36 tissue types (102 slides), scanner-equal mean.
    PLISM uses core and section means, with other sections as the bank.
- **Primary endpoints:** detectability per method × PFM (PanNormal pooled); retrieval macro
  recall minus the real-target value; normalized target distance.
- **Expectations (hypotheses):**
  - H4a: image corrections leave detectability ≥ 0.95 in every PFM.
  - H4b: feature corrections lower detectability markedly within PanNormal but not in PLISM.
  - H4c: ridge affine has a spread ratio below 1, so its normalized gain is smaller than its
    raw gain.
  - H4d: tissue retrieval falls after some image corrections in some PFMs (for example
    Virchow2 after Pix2Pix).
- **How we read it:** if a method lowers target distance but not detectability, it moved
  embeddings without removing scanner identity. If detectability falls to near chance with
  retrieval preserved, that method is the internal upper bound and is presented as such.
  If spread shrinks, distance gains are partly compression and are reported normalized.
- **Output:** `results/three_axis_evaluation/`.

## RV04b Target detectability: correction-independent and nonlinear probes (amendment)

*Added on 2026-09-29, after RV04 and before any RV04b outcome was computed.*

- **Why:** in RV04 the classifier's cross-validation folds are the correction folds. The
  corrected training embeddings of fold j come from a map fitted on folds that include the
  classifier's test fold, so the classifier learns a shift that reverses on the test fold;
  ComBat and OLS fell below chance. Separately, a linear probe near chance after an affine
  or moment-matching correction does not show that scanner identity is gone (the author notes
  that FEATMAP reported the same linear-probe behaviour, and historical E8 found Procrustes at
  0.12–0.16 linear but 0.61–0.78 with an MLP).
- **Design:**
  - *PanNormal:* for each outer fold k, use only the slides held out in fold k, whose corrected
    embeddings come from map k and whose real targets were not used to fit map k. Split those
    slides into five inner groups (sorted slide IDs, index modulo 5); train on four, test on
    one. Accuracy per location → per slide (classes balanced within slide) → mean over 103
    slides.
  - *PLISM:* corrections were fitted on PanNormal, so the RV04 section folds (sorted
    sections, index modulo 5) are kept; each fold-fitted variant is evaluated separately and
    averaged within section.
  - Classes: corrected AT2 vs real target, pooled over directions; inputs L2-normalized.
  - Methods: raw and every image and feature method; four PFMs.
- **Classifiers (fixed before running):**
  - linear: StandardScaler + LogisticRegression(C = 0.1, liblinear), identical to RV04;
  - MLP: StandardScaler + MLPClassifier(hidden 256, ReLU, alpha = 1e-3, early stopping on a
    10% validation split, patience 10, max 300 iterations, random_state 0);
  - k-NN: k = 15, cosine distance, uniform weights.
- **Primary endpoints:** detectability per classifier × method × PFM × dataset; paired
  differences MLP − linear and k-NN − linear (slide or section bootstrap, 2,000 resamples,
  shared with RV04).
- **Hypotheses and read-outs:**
  - H4e (leakage): within-fold linear detectability of ComBat and OLS is not below chance
    (CI upper bound ≥ 0.5). If it is still below chance, leakage is not the whole explanation.
  - H4f (nonlinear residue, author's hypothesis): after feature corrections in PanNormal, MLP
    detectability exceeds linear detectability (CI of the difference above 0) in each PFM.
  - Design check: raw within-fold detectability stays near the RV04 raw value (≈ 0.99).
- **How we read it:** if nonlinear probes detect the target scanner well above chance where the
  linear probe does not, feature correction removes the linearly decodable part of the scanner
  signal but not scanner identity, and any claim about "removing" scanner effects is
  probe-dependent.
- **Output:** `results/detectability_within_fold/`.

## RV11 Scanner × tissue interaction as a direct fixed effect (amendment)

*Added on 2026-09-29, before any RV11 outcome was computed.*

- **Question:** is the scanner effect on each image measure different between tissue types,
  tested with the simplest model that keeps scanner × tissue as an explicit term?
- **Why:** the RQ1 mixed model has four random components, REML fits and bootstrap
  likelihood-ratio tests, and it reports variance fractions of what remains after removing
  scanner means. The same question can be asked with one fixed-effects model whose
  interaction estimates are read directly in the measure's units.
- **Data:** the RQ1 input, `analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv`
  (slide × target-scanner contrasts to AT2; 13 measures; 103 slides × 5 scanners; balanced).
- **Model (per measure):** d(s, r) = α_r + β_s + γ_(s, t(r)) + ε. α_r is the slide offset
  (absorbs the tissue main effect and slide differences), β_s the mean scanner effect,
  γ the scanner × tissue term, ε the scanner × slide variation within a tissue.
- **Test:** F = [SS_γ / (S−1)(T−1)] / [SS_ε / (S−1)(R−T)], df 144 and 264. Primary p value from
  9,999 permutations of tissue labels across slides (each slide keeps its five contrasts);
  the F-distribution p value is reported alongside; Benjamini–Hochberg across the 13 measures.
- **Effect sizes, in measure units:** tissue-specific scanner effects m(t, s) = mean contrast of
  scanner s over the slides of tissue t; per scanner, their between-tissue SD and range; the
  tissue share SS_γ / (SS_γ + SS_ε); for high-frequency transfer, the number of tissues in which
  AKOYA is lowest and GT450 highest.
- **Sensitivity:** excluding the one tissue represented by a single slide.
- **Secondary:** the same model on the RV09 slide × scanner target gains (Reinhard, colour +
  frequency, frequency increment, ridge affine, ComBat; UNI v1 primary, other PFMs secondary).
- **Expectation:** scanner × tissue interaction for most measures, as in the mixed model.
- **How we read it:** if conclusions agree with the mixed model, the fixed-effects model is
  proposed for the main text and the mixed model moves to the supplement.
- **Output:** `results/scanner_tissue_interaction/`.

## RV12 Frequency transfer curves and tissue × scanner map (figures)

*Added on 2026-09-29, before any RV12 figure was drawn.*

- **Question:** can the frequency phenotype and its tissue dependence be shown directly rather
  than as single band summaries?
- **Data:** `outputs/final_image_study_v1/03_frequency/spectra.csv` (103 slides × 6 scanners ×
  72 radial bins, all matched locations per slide; log2 amplitude transfer and coherence to
  AT2), and RV11's m(t, s).
- **Panels:**
  - A. log2 transfer to AT2 against spatial frequency (cycles/µm), one curve per scanner:
    mean of tissue-type means (tissue-equal, as in Fig. 2), ribbon = interquartile range across
    tissues; analysis bands shaded; AT2 Nyquist (0.99 cycles/µm) marked.
  - B. every tissue's curve for AKOYA and GT450, with the two tissues quoted in the text
    (aorta, liver) highlighted.
  - C. tissue × scanner heatmap of m(t, s) for selected measures (a*, OD contrast, low–mid and
    high-frequency transfer), diverging scale centred at zero.
  - Supplementary: coherence to AT2 against frequency.
- **Check:** band averages of the curves reproduce the slide-level band summaries
  (`band_summary.csv`).
- **Style:** the paper's scanner colours, with direct labels and line styles as secondary
  encoding. No new statistics.
- **Output:** `results/frequency_transfer_figures/`.

## RV13 Anchor, frequency or data diversity (amendment)

*Added on 2026-09-29, before any RV13 outcome was computed.*

- **Question:** is the scanner signal of PFMs explained by
  - H1: image-only pretraining, which a non-image anchor (text or spatial transcriptomics) would reduce;
  - H2: reliance on high-frequency content;
  - H3: training data from narrow acquisition sources (curated WSI tiles), as opposed to
    heterogeneous figure or web images;
  - H4: and does stain normalization in pretraining and inference (EXAONEPath) remove only the
    colour-driven part?
- **Why:** anchored models (CONCH, SEAL) looked less or differently sensitive in an earlier
  three-location review measured with absolute cosine distances, which are not comparable
  across models. This analysis uses scale-free metrics on the evaluation set.
- **Models (cells):**

  | | Narrow sources (curated WSI) | Heterogeneous sources |
  | --- | --- | --- |
  | Image-only | UNI v1, UNI2-h, Virchow2, H-optimus-1; EXAONEPath (+ stain normalization) | DINOv2 ViT-L/14 (natural images; reference) |
  | Anchored | SEAL-UNI2, SEAL-CONCH (spatial transcriptomics) | CONCH (figure–caption; pre-projection primary, projected secondary), PLIP (image–text) |

  CLIP is deferred until the CONCH pattern is known.
- **Preprocessing:** each model's released preprocessing applied to the same 256 × 256 field
  (0.5052 µm/px). EXAONEPath as released: its Macenko normalizer (fixed reference), resize 256,
  centre crop 224, ImageNet normalization; the same without Macenko is an off-label comparison.
- **Conditions (PanNormal `set20`):** raw six scanners; Reinhard, frequency and colour +
  frequency (AT2 → five targets, regenerated as in RV03); band manipulations as in RV02 (three
  bands, increase and decrease, doses 0.25 and 0.50, scanner target direction). The four
  original PFMs reuse the RV02 and RV03 embeddings.
- **Metrics (scale-free):**
  1. normalized scanner distance: mean AT2–target cosine distance / mean between-tissue distance
     among raw AT2 embeddings;
  2. detectability of AT2 vs target (raw): linear, MLP and k-NN probes as in RV04b, locked
     slide-fold cross-validation; six-way linear scanner probe;
  3. colour / frequency decomposition: fraction of the raw target distance removed by Reinhard
     (colour share), by frequency matching alone, and additionally by colour + frequency over
     Reinhard (frequency share);
  4. band sensitivity: representation shift per band at doses 0.25 and 0.50 divided by the
     between-tissue distance; high / low–mid ratio;
  5. tissue information: same-tissue retrieval macro recall among raw AT2 slide means (36
     tissues / 102 slides), so that low scanner sensitivity is not mistaken for robustness when
     a model is simply uninformative.
  6. PathoROB robustness index (added 2026-09-29 before any RV13 outcome; Kömen et al., Nature
     Communications 2026; definition as in `pathorob/robustness_index/robustness_index_utils.py`):
     raw embeddings of all six scanners at `set20` pooled (biological class = tissue type,
     confounder = scanner); L2-normalized, k nearest neighbours with neighbours from the same
     physical slide excluded (this removes the same location on other scanners); SO = same tissue,
     other scanner; OS = other tissue, same scanner; RI_k = ΣSO / Σ(SO + OS), accumulated over
     neighbour ranks 1..k and all anchors. Anchors: the 36 tissue types with at least two slides.
     Primary k: PathoROB's rule (the k in {5, 10, 15, 20, 25, 30, 40, 50} that maximizes kNN
     balanced accuracy for tissue type); the RI curve over k = 1–50 is reported alongside.
     Because tissues outnumber scanners here, a random embedding gives RI ≈ 0.12, not 0.5; values
     are compared across models on the same data, with this chance level stated.
- **Comparisons and read-outs:**
  - H1 (controlled): SEAL-UNI2 − UNI2-h and SEAL-CONCH − CONCH, paired slide bootstrap, on
    normalized scanner distance, best-probe detectability and normalized high-band
    sensitivity. Supported if both contrasts lower normalized distance and detectability.
  - H2: image-only WSI models show a larger frequency share and higher normalized high-band
    sensitivity than heterogeneous-source models.
  - H3: heterogeneous-source models (CONCH, PLIP, DINOv2) are lower on normalized distance and
    detectability than narrow-source models (image-only WSI and SEAL), without lower tissue
    retrieval. If anchored models are lower regardless of source instead, the anchor
    explanation is favoured.
  - H4: EXAONEPath as released has a lower colour share than the other image-only WSI models,
    a frequency share that is not lower, and high detectability.
- **Caveat:** cross-model comparisons are confounded by architecture, input size and pretraining
  data; within-backbone contrasts (H1) and EXAONEPath with vs without Macenko are the only
  controlled comparisons. DINOv2 is a non-pathology reference. Results go to the supplement.
- **Output:** `results/anchor_frequency_diversity/`.

## RV14 Perturbation sensitivity and scanner robustness across PFMs

*Defined on 2026-09-29 after the RV13 results (internal record); reported descriptively.*

- **Question:** across PFMs, does the size of the representation shift caused by a small,
  controlled spatial-frequency perturbation (RV02/RV13 band manipulation, equal OD change,
  divided by the model's between-tissue distance) predict robustness to real scanner
  differences, and does the band preference (high / low–mid ratio) matter?
- **Models:** the eight WSI-pretrained variants (UNI v1, UNI2-h, Virchow2, H-optimus-1,
  EXAONEPath as released, SEAL-UNI2, CONCH and SEAL-CONCH pre-projection). PLIP, DINOv2 and
  EXAONEPath with raw input are shown for reference and not included in the correlations.
- **Measures (from RV13 `model_draws`, shared 2,000 slide resamples):** normalized shift per
  band (low–mid, mid, high; doses 0.25 and 0.50), high / low–mid ratio; robustness: PathoROB RI,
  normalized scanner distance, best-probe detectability.
- **Statistics:** Spearman ρ across the eight models; 95% CI from recomputing ρ in every slide
  resample (measurement uncertainty); exact permutation p over the 8! model orderings of the
  point estimates; partial Spearman controlling for between-tissue distance.
- **Figures:** main — sensitivity vs RI and vs normalized scanner distance with model labels,
  and ρ by band and for the ratio; supplement — tissue information vs scanner sensitivity for
  all variants (PLIP and CONCH contrast).
- **Output:** `results/sensitivity_robustness/`.

## RV05 Downstream proxy: cross-scanner tissue classification

- **Question:** when a classifier is built on one scanner, does correcting images or
  features from another scanner recover its performance, and does representation gain
  predict that recovery better than image fidelity does?
- **Design:** for each PFM and target scanner, leave one slide out. Build tissue centroids
  from the real target-scanner location embeddings of all other slides (36 tissue types
  with at least two slides). Classify the held-out slide's locations under the real target
  (reference), raw AT2, and every corrected AT2 condition. PLISM leaves one section out;
  every section contains all cores.
- **Primary endpoint:** tissue macro recall (location accuracy averaged within slide, then
  within tissue, scanner-equal); **recovery** = (corrected − raw) / (real target − raw),
  computed only where real target − raw ≥ 2 percentage points.
- **Secondary:** multinomial logistic regression instead of nearest centroid; Spearman
  correlation across method × scanner × PFM cells between recovery and (a) target gain,
  (b) SSIM or image-residual improvement.
- **Expectation:** recovery is positively but imperfectly associated with target gain and
  weakly with image fidelity.
- **How we read it:** a strong association supports target distance as a proxy; a weak one
  says distance alone is insufficient even for a coarse task. If the raw-to-target gap is
  below 2 points for most cells, the result is reported as "scanner shift has little
  effect on coarse tissue identity", itself informative, and recovery is not computed.
- **Output:** `results/downstream_tissue_proxy/`.

## RV06 Existing results consolidated

- **What:** verify provenance and design for results already computed but absent from the
  paper, and collect them in one table without recomputation:
  (a) symmetric low-pass filtering (scanner probe vs tissue accuracy),
  `outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/04_frequency_mechanism/02_matched_information_removal/`;
  (b) PLISM Pix2Pix/CycleGAN in four PFMs, `outputs/plism_gan_crossencoder_2026-09-25/`;
  (c) feature correction in the reverse direction, `outputs/discussion_followup_2026-09-25/feature40_reverse/`
  and `feature40_external_reverse/` (not the 20-location `feature_reverse/`).
- **Output:** `results/existing_evidence/`.

## RV07 Summary figure

Drawn after RV04 and RV05 are discussed: for each method × scanner × PFM, image-fidelity
improvement against representation gain, with detectability and tissue preservation encoded
on the same panel. No new computation.

*Drawn on 2026-09-29 after the framing discussion (image alignment vs representation alignment).*
Script `summary_figure.py` (+ `.sbatch`); output `results/summary_figure/`.

## RV08 Registration noise floor

- **Question:** how much target distance remains when two images differ only by
  misregistration of the size allowed by quality control?
- **Design:** AT2 images at `set20`; shifts of 0.25 and 0.5 pixel (Fourier shift of a padded
  crop) and 1 and 2 pixels (read from the WSI), in four directions. Verify first that reading
  the AT2 WSI at the stored coordinates reproduces the cached patch exactly.
- **Primary endpoint:** mean PFM cosine distance between original and shifted image, per PFM
  and shift, as a fraction of the raw AT2–target distance; SSIM and LPIPS as image
  counterparts.
- **How we read it:** report the fraction of achievable reduction,
  (raw − corrected) / (raw − floor), next to the percentage reduction. A floor that is a large
  share of the best corrected distance means that part of the residual is irreducible under
  this design.
- **Output:** `results/registration_floor/`.

## RV09 Tissue dependence of correction gains

- **Question:** do correction gains depend on tissue type, as the scanner image phenotypes
  do (RQ1)?
- **Design:** slide × scanner target gains for Reinhard, colour + frequency (and the
  frequency increment over Reinhard), ridge affine and ComBat; UNI v1 primary, other PFMs
  secondary. The RQ1 mixed model: fixed scanner means; random shared tissue,
  scanner × tissue and slide effects; restricted likelihood-ratio tests with 500 parametric
  bootstrap replicates; Benjamini–Hochberg across methods.
- **Secondary:** Spearman correlation across tissue types, for AKOYA and GT450, between
  the tissue-mean high-frequency log2 ratio and the tissue-mean frequency increment.
- **Expectation:** tissue dependence present for most methods; for AKOYA, larger frequency
  increments in tissues with a stronger high-frequency deficit.
- **Output:** `results/gain_tissue_dependence/`.

## RV10 Robustness of the high-frequency phenotype to resampling

- **Question:** is the high-frequency ordering of scanners (AKOYA lowest, GT450 highest) a
  property of acquisition or of how images were resampled to 0.5052 µm/px?
- **Background:** GT450 (0.2624 µm/px, 40×) and VERSA (0.2742 µm/px, 20× optics) were
  downsampled about 1.9× and 1.8×; AKOYA (0.4999) almost not at all. Resampling happened in
  the VALIS registration outside this repository, and its interpolation and anti-aliasing
  settings are not recorded.
- **Stage A (feasibility):** obtain the AT2-to-native mapping for GT450, VERSA, S360, S60 and
  AKOYA (VALIS outputs or the E0 native-geometry transforms) and confirm by local matching
  that native crops correspond to the registered patches.
- **Stage B:** for `set20`, extract native crops with a margin and resample to 0.5052 µm/px by
  (i) bicubic without anti-aliasing, (ii) anti-aliased Lanczos, (iii) box integration over the
  AT2 pixel footprint. Identify which variant the registered patch matches best.
- **Stage C:** high-band log2 ratio to AT2 and UNI v1 distance to AT2 per variant.
- **Primary endpoint:** sign and ordering of the scanners' high-band ratios under each
  variant.
- **How we read it:** if the GT450 excess disappears under anti-aliased resampling, the
  high-frequency axis of RQ1 partly reflects processing for downsampled scanners and is
  reframed. If it persists, this becomes a supplementary robustness result.
- **Output:** `results/resampling_robustness/`.

---

## Deviations

| Date | Analysis | Deviation | Reason |
| --- | --- | --- | --- |
| 2026-09-29 | RV-P0b | Pix2Pix/CycleGAN UNI distances in the long table use the GAN `04_uni` feature shards restricted to `set20` (path B); the published values come from the `gan_encoder_review_2026-09-25` re-embedding (path C), which is also on `set20` and kept in `summary_methods.csv`. | Path B matches the other methods' extraction (RV-P0c); path C clamps after resizing and differs by up to 0.009 per location. |
| 2026-09-29 | RV-P0b | Oracle classifier: primary estimate fits on `set40` of training folds and evaluates on `set20` (rule 1); a `set20`-only refit is reported alongside. | Rule 1 separates fitting from evaluation. |
| 2026-09-29 | RV-P0b | Two gradient-NCC definitions are kept: the image-phenotype OD-gradient NCC in `image_metrics_slide.csv`, the benchmark grayscale NCC in `image_metrics_slide_supplementary.csv`. | The paper uses the first for AKOYA and the second for the GT450 edge constraint. |
| 2026-09-29 | RV-P0b | Scope extended to the GT450 edge-constraint and modal-candidate numbers in the text; PLISM, blur, three-location high-band ratios and tissue retrieval left to RV01, RV08/RV10 and RV04. | Completeness of the comparison without overlapping other analyses. |
| 2026-09-29 | RV04 | PLISM detectability uses five folds grouped by section (sorted sections, index mod 5). | The protocol names slide-grouped folds only for PanNormal. |
| 2026-09-29 | RV04, RV05 | Classifier and retrieval inputs are L2-normalized. | Feature-corrected embeddings are not unit-norm; without normalization a classifier could separate conditions by vector length. No effect on stored unit-norm embeddings. |
| 2026-09-29 | RV04 | Between-tissue distance = mean over all location pairs from different tissue types across the 103 held-out slides (PLISM: different cores, any section), recomputed in every bootstrap resample. | Makes the protocol's definition explicit. |
| 2026-09-29 | RV04, RV05 | Retrieval and classification CIs resample query slides with predictions fixed; the reference bank is not resampled. | Keeps the bank identical across conditions. |
| 2026-09-29 | RV05 | Secondary logistic regression: StandardScaler, C = 0.1, lbfgs, same leave-one-slide-out design. | Protocol left the configuration open. |
| 2026-09-29 | RV09 | Primary test is any tissue dependence (shared tissue and scanner × tissue variances jointly); the scanner × tissue test is also reported; Benjamini–Hochberg across the five methods within each PFM. | Mirrors the RQ1 testing sequence. |
| 2026-09-29 | RV04 | Hypothesis read-outs made mechanical: H4a estimate ≥ 0.95; H4c spread-ratio CI upper bound < 1; H4d CI upper bound of retrieval minus raw < 0; H4b reported without a threshold. | "Markedly" in H4b has no numeric definition; stated before outcomes. |
| 2026-09-29 | RV05 | Feature methods are excluded from the image-fidelity Spearman association. | They do not change the image. |
| 2026-09-29 | RV01 | The paper's additional strong-blur (σ = 12, 24, 48) and gamma sweeps were not re-run. | Not listed in the protocol. |
| 2026-09-29 | RV01 | Scanner probe reuses the six-way probe of `frequency_aggregate.py` with the locked folds, on location embeddings (primary) and slide means (sensitivity); σ = 3 added. | Protocol did not name the classifier. |
| 2026-09-29 | RV01 | Bootstrap CIs added for every endpoint; retrieval CIs resample query slides with the bank fixed. | The paper reported point estimates only. |
| 2026-09-29 | RV02 | Each slide's manipulated images are rendered once and embedded by all four PFMs through their own paper embedding paths. | Identical inputs across PFMs. |
| 2026-09-29 | RV02 | For UNI2-h, Virchow2 and H-optimus-1, stored Reinhard embeddings exist only at the old three locations; the other 17 locations are parity-checked on raw target embeddings only. | No stored reference for them. |
| 2026-09-29 | RV02 | 2,000 bootstrap resamples for all PFMs (the paper's cross-PFM script used 3,000). | Protocol statistics rule. |
| 2026-09-29 | RV10 | Stage A gate set to ECC ≥ 0.95 at ≥ 95% of locations per scanner, fixed after a one-slide pilot and before the full run. | Protocol named no numeric gate. |
| 2026-09-29 | RV10 | E0 native-geometry transforms used only as a starting point; geometry is a per-location refined affine (NCC then ECC). | Current AKOYA registration is non-rigid and not in the E0 frame. |
| 2026-09-29 | RV10 | Kernels: bicubic = Catmull-Rom (a = −0.5); box = 16×16 supersampled average; Lanczos anti-aliased. | Makes the variants explicit. |
| 2026-09-29 | RV10 | 1–2 unconfirmed locations per scanner dropped from Stages B and C. | Failed the Stage A match. |
| 2026-09-29 | RV10 | UNI v1 ran in fp32 on CPU. | Per-user GPU limit reached. |
| 2026-09-29 | RV-P0d | GAN training for the structure ablation was not re-run; restoration verified from the frozen checkpoints onward (predict, image, UNI, compare). | GPU training is not bit-reproducible, so a retrain would be a new replicate, not a restoration. Recovered training code is ready (~5 A100-h) if wanted. |
| 2026-09-29 | RV-P0d | Restore scripts are verbatim recovered code with output redirected to `results/provenance_restoration/`; code identity was tied to the runs through byte-code caches left in `.Trash`. | Original sources were deleted during the refactor. |
| 2026-09-29 | RV06 | Existing results keep their original bootstrap counts (20,000 for low-pass and GAN reverse, 4,000 for PLISM GAN, 5,000 for feature reverse) and are recorded, not recomputed. | Consolidation only; recomputation was not required by any inconsistency. |
| 2026-09-29 | RV08 | Three stage-specific sbatch files (identity gate on CPU, GPU embedding, CPU aggregation) instead of one. | Stages need different partitions and the identity check gates the rest. |
| 2026-09-29 | RV08 | UNI v1 embedded in full precision via `load_encoder("uni_v1")`; stored `03_uni` used half precision (parity ≥ 0.99994). Fourier window padded 128 px, shifted images rounded to 8 bit; fractions are ratios of means bootstrapped jointly; the SSIM ratio uses 1 − SSIM; each method's fraction uses the raw distance from its own evidence file. | Choices left open by the protocol. |
| 2026-09-29 | RV11 | Partial ω² = df_γ(F − 1) / [df_γ(F − 1) + N] added as the chance-corrected effect size; the tissue share SS_γ / (SS_γ + SS_ε) has an expectation of 0.353 under no interaction with these degrees of freedom. | The protocol's tissue share is not interpretable without its chance level. |
| 2026-09-29 | RV13 | CONCH, SEAL-CONCH and SEAL-UNI2 use the earlier review loader (bicubic upsampling on float tensors, clamp, half precision) rather than CONCH's PIL transform (parity 0.9996); PLIP and DINOv2 processors reproduced on float tensors so sub-intensity band renders are not quantized (parity ≥ 0.9993); DINOv2's released preprocessing centre-crops 224 of the 256 field. | Band renders perturb by ~1 intensity level; PIL quantization would erase them. |
| 2026-09-29 | RV13 | EXAONEPath: timm ViT-B/16 with the HF config; `macenko.py` needs Python ≥ 3.10, so a line-by-line port is used (reference max abs diff 5e-5; parity with the official path ≥ 0.99999); torchstain 1.4.1 vendored under `third_party/torchstain` (MIT). The released `macenko_normalizer()` fits its reference to the bundled TCGA target image; 'as released' uses that fitted reference. Macenko failed on 147 of 166,860 images (32 locations, 7 slides); the released fallback (input image) is kept and flagged; results unchanged without them. | Faithful to the released pipeline. |
| 2026-09-29 | RV13 | Six-way probe = RV01's class-balanced multinomial probe; group means weight models equally with pre-projection CONCH features; RI chance level computed exactly = 0.089 (the amendment's ~0.12 was an approximation). | Implementation choices left open. |
| 2026-09-29 | RV13 | PLIP weights loaded with `torch.load(weights_only=True)` (the installed transformers refuses the `.bin` under torch < 2.6); best probe = maximum of the three probes within each resample; band sensitivity primary dose 0.25; H2–H4 read as difference with CI (H4 'high detectability' ≥ 0.95); RI computed inside the metrics script. Final aggregation job 24240149. | Implementation choices; no effect on estimates. |
| 2026-09-29 | RV03 | PLISM Pix2Pix/CycleGAN for UNI2-h, Virchow2 and H-optimus-1 regenerated from the frozen checkpoints (only distances were stored); they agree with the stored distances to a mean of ≤ 3e-5. | Per-location features did not exist. |
| 2026-09-29 | RV03 | PanNormal Macenko/Vahadane regenerated with the seed convention of the stored UNI embeddings (no fold in the seed), not the fold-seeded `table2_benchmark` convention; differences ≤ 3 intensity levels. | Parity with the stored embeddings. |
| 2026-09-29 | RV03 | UNI v1 embedded through the paper path (`prenorm.embedding`), not the review script, which clamps after resizing. | The smoke test showed per-location differences up to 9e-3 with the clamping path. |
| 2026-09-29 | RV07 | Two panels: (A) image-residual reduction vs representation gain per image correction × scanner × PFM (PanNormal); (B) representation gain vs best-probe detectability (RV04b) per correction × PFM, both datasets. Tissue preservation is not encoded (it is in Tables 2–3); detectability is pooled over directions because RV04b has no per-scanner estimate. Representation gain = target gain / raw between-tissue distance of the PFM; Spearman ρ across the 35 cells per PFM added as a descriptive summary. | One panel with every encoding was unreadable; the scale-free gain avoids the blow-up of relative gains where raw distances are small (Virchow2). |
| 2026-09-29 | RV03 | Affine OLS fitted for UNI2-h, Virchow2 and H-optimus-1 with the same code. | No paper counterpart; protocol lists OLS as secondary. |

## Open issues

| Date | Analysis | Issue | Proposed resolution |
| --- | --- | --- | --- |
| 2026-09-29 | RV04 | Detectability for fold-fitted corrections (feature corrections, and potentially Reinhard, CycleGAN and other fold-fitted image corrections) uses corrected embeddings from maps whose fitting data include the real targets of the classifier's test fold. Sub-chance accuracy for ComBat and OLS shows the resulting bias. | Recompute detectability within each outer fold: for the slides held out in fold k, train and test the classifier by inner slide-grouped cross-validation on those slides only, so neither class touches the data used to fit fold k's map; average over folds. Apply to every method, raw included, for comparability. **Resolved by RV04b (2026-09-29).** |

## Results index

Filled in as analyses finish; one line per analysis with the headline result and the path
of its summary.

| ID | Result | Summary |
| --- | --- | --- |
| RV-P0a | `set40` (4,120) and `set20` (2,060) written; `set20` equals the stored UNI locations; the paper's blur and band probes share 206 of 309 locations. | `results/location_sets/summary.json` |
| RV-P0b | 112/114 published numbers reproduced from stored data; on `set20` no sign or ranking changes; 25 numbers move at printed precision (largest: AKOYA colour + frequency residual 0.927 → 0.934; oracle classifier 93.9% → 93.7%; joint coverage 49.1% unchanged). Pix2Pix/CycleGAN image residual computed for the first time (0.404 and 0.550 vs raw 1.313). | `results/set20_reaggregation/summary.md` |
| RV-P0c | Pass: minimum cosine 0.99999 over 20,600 comparisons; input pixels identical. | `results/uni_extraction_parity/summary.md` |
| RV01 | All three qualitative statements hold on `set20`: displacement cosine 0.772 (blur σ = 3) and 0.697 (sharpen α = 2); distance ratio at σ = 6 0.551; retained gradient 0.98%; same-tissue NN 64.2% → 45.1%. Six-way scanner probe 0.994 raw vs 0.967 at σ = 6. | `results/blur_sharpen_set20/summary.md` |
| RV02 | High-minus-low–mid signs unchanged in all four PFMs (UNI v1 +0.0012, UNI2-h +0.0031, Virchow2 −0.0006, H-optimus-1 −0.0019 at dose 0.25). Expectation not met for UNI2-h: its high-band target-direction gain is positive (+0.0053, 0.0044–0.0062); the same sign is present in the paper's stored outputs but was not reported. UNI v1 −0.0009, Virchow2 +0.0001, H-optimus-1 −0.0010. | `results/band_manipulation_set20/summary.md` |
| RV10 | Stage A passed (2058–2059/2060 locations per scanner). No tested resampler reproduces the registered patches (box closest). GT450 highest and AKOYA lowest under every resampler in all 2,000 bootstrap resamples; GT450's ratio stays positive (+0.53 to +0.96 log2 vs +0.68 for the pipeline). VERSA and S60 change sign with the kernel; magnitudes shift by −0.18 to +0.38 log2. UNI v1 distances to AT2 change by ≤ 0.007 and keep their ranking. | `results/resampling_robustness/summary.md` |
| RV-P0d | All three restorations reproduce their frozen outputs byte-identically: Fig. 2 inputs (true producer `scanner_tissue_space.py`, recovered from `.Trash`, not named in `analysis/paper/README.md`), PLISM GAN cross-PFM distances (214,830 values), structure ablation from frozen checkpoints. | `results/provenance_restoration/` |
| RV06 | 40 checks: 32 pass, 8 notes, 0 fail. Low-pass run code identity very likely but not provable (source files edited after the run). PLISM GAN: pooled gains negative in all four PFMs, but UNI2-h GT450 gains are positive (Pix2Pix +0.006, CycleGAN +0.027; CIs include 0). GAN reverse runs use 40 locations; several checkpoints were selected early (Pix2Pix GT450→AT2 fold 4 at pass 5; CycleGAN folds 1–3 at passes 20–25) by the pre-set rule. | `results/existing_evidence/summary.md` |
| RV08 | Identity and shift checks pass. The 1-px floor is 2–4% of the raw AT2–target distance (UNI v1 0.028, UNI2-h 0.035, Virchow2 0.019, H-optimus-1 0.036) and 3–6% of the best corrected distance; fraction of achievable reduction ≈ percentage reduction (UNI v1 ridge 0.353 vs 34.3%). Image metrics are far more sensitive: SSIM 0.729 at 1 px and 0.384 at 2 px, the latter below the raw AT2–target SSIM of 0.572. | `results/registration_floor/summary.md` |
| RV03 | Complete: 103 PanNormal slides × 56 conditions and 13 PLISM sections × 154 conditions, four PFMs. All embedding parity checks ≥ 0.999; all 100 ridge penalties match; 45/48 cells of the cross-PFM table reproduced at four decimals (three UNI v1 PanNormal cells off by 0.0001, explained by the embedding path). | `results/corrected_embeddings/summary.md` |
| RV04 | All parity checks pass. Detectability: every image correction ≥ 0.98 except CycleGAN (0.91–0.95) in PanNormal; everything ≥ 0.99 in PLISM, feature corrections included. PanNormal feature-correction detectability is **not interpretable as run**: ComBat (0.29–0.36) and OLS (0.35–0.73) fall below chance, the signature of cross-fitting leakage (correction maps for one fold are fitted on targets that serve as the classifier's test fold); see Open issues. Ridge spread ratio 0.93–0.95 in all PFMs (H4c 24/24); normalized target distance: UNI raw 0.243, Reinhard 0.211, colour + frequency 0.214, ridge 0.180, ComBat 0.184. Retrieval within ±3 pp of the real target except UNI Macenko (−6.1) and Virchow2 Pix2Pix (−7.5); PLISM retrieval is near ceiling (94–99%). | `results/three_axis_evaluation/summary.md` |
| RV04b | Leakage confirmed and removed: within-fold linear detectability of ComBat is 0.55–0.60 and of OLS 0.58–0.74 (H4e: 8/8 not below chance; RV04 cross-fitted values 0.29–0.73). No feature correction removes scanner identity under every probe in PanNormal: ComBat is near chance linearly but detectable by k-NN (0.62–0.80) and MLP (MLP − linear +0.04 to +0.07, CI > 0 in all four PFMs); OLS MLP − linear +0.02 to +0.03 (all > 0); ridge keeps a linear signature (0.67–0.87) while k-NN is near chance (0.54–0.58). H4f 9/12 (ComBat 4/4, OLS 4/4, ridge 1/4). Image corrections stay ≥ 0.91 under every probe (CycleGAN lowest). PLISM: every method ≥ 0.99 linear, including feature corrections. Raw within-fold linear 0.96–0.99 (design check). | `results/detectability_within_fold/summary.md` |
| RV11 | Scanner × tissue interaction for all 13 measures (permutation q ≤ 0.0047; also 13/13 without the single-slide tissue). Partial ω² agrees with the mixed-model interaction fraction for frequency measures (low–mid 0.53 vs 0.57, mid 0.38 vs 0.39, high 0.21 vs 0.20) and is 0.19–0.42 for the others. AKOYA lowest and GT450 highest high-frequency transfer in 37/37 tissues. Correction gains: interaction in 19/20 method × PFM cells (H-optimus-1 Reinhard p = 0.099). | `results/scanner_tissue_interaction/summary.md` |
| RV12 | Transfer curves: AKOYA falls from ~0.1 cycles/µm to −1.8 log2 at high frequency (IQR across tissues ±0.3); GT450 rises to +0.7; VERSA, S60 and S360 separate only above ~0.6 cycles/µm. AKOYA tissue curves span −1.2 (liver) to −2.5 (aorta). Band averages reproduce `band_summary.csv` exactly. Coherence with AT2 decays to near 0 at high frequency for every scanner and by ~0.4 cycles/µm for VERSA, so paired high-frequency content is not phase-aligned (misregistration and/or incoherent noise): amplitude-based phenotypes are unaffected, pixel-wise metrics such as SSIM are. | `results/frequency_transfer_figures/` |
| RV13 | No pretraining strategy removed scanner identity: all 11 model variants detect the target scanner at ≥ 0.95 (best probe) and six-way probes are ≥ 0.95. H1 not supported (SEAL − base: detectability unchanged; normalized distance −0.013 for UNI2-h, +0.026 for CONCH). H3 not supported: heterogeneous-source models are not lower on normalized distance (+0.005, n.s.), and have lower RI (−0.13) and tissue retrieval (−0.19). H4: EXAONEPath has the largest normalized distance (0.476; raw input 0.378), a larger colour share than other WSI models (0.246 vs 0.125; prediction not met) and a low RI (0.18). H2: image-only WSI models slightly higher frequency share and normalized high-band shift (small, CI > 0). PathoROB RI (chance 0.089): Virchow2 0.53, H-optimus-1 0.51, CONCH 0.41, SEAL-CONCH 0.41, SEAL-UNI2 0.39, UNI 0.34, UNI2-h 0.30, DINOv2 0.27, EXAONEPath 0.18, PLIP 0.06. Exploratory (not pre-specified): across variants, normalized high-band sensitivity correlates with RI (Spearman −0.74) and normalized distance (+0.87; shared denominator). | `results/anchor_frequency_diversity/summary.md` |
| RV14 | Across the eight WSI-pretrained PFMs, the representation shift per equal-OD frequency perturbation (normalized by between-tissue distance) tracks robustness to real scanner differences: high band vs PathoROB RI ρ = −0.90 [−0.98, −0.74], permutation p = 0.005; vs normalized scanner distance ρ = +0.90; mid −0.88, low–mid −0.76; dose 0.50 high −0.95. Band preference (high / low–mid ratio) does not (ρ = −0.19, p = 0.66). Unchanged when controlling between-tissue distance (partial −0.90) or using raw shifts; detectability is at ceiling and uncorrelated. Main figure `fig_sensitivity_robustness`; supplement `fig_tissue_vs_scanner_supp` (PLIP: 63% same-scanner / 4% same-tissue neighbours vs CONCH 30% / 21%). | `results/sensitivity_robustness/summary.md` |
| RV05 | No raw-to-target gap: a classifier built on real target-scanner embeddings classifies raw AT2 as well as the targets (0/20 PanNormal and 0/12 PLISM cells reach 2 pp; median gap −0.9 and +0.2 pp), so recovery is not computed. Several corrections lower tissue accuracy instead: Pix2Pix in Virchow2 52.0 → 43.8%, Macenko in UNI 57.5 → 51.9%; in PLISM Vahadane, Pix2Pix and CycleGAN lower Virchow2 by 8–10 pp. | `results/downstream_tissue_proxy/summary.md` |
| RV07 | Image alignment explains part of representation alignment: Spearman ρ between image-residual reduction and representation gain across 35 correction × scanner cells 0.55 (UNI v1), 0.52 (UNI2-h), 0.17 (Virchow2, n.s.), 0.68 (H-optimus-1). In 17/35 cells the same images move at least one PFM toward and another away from the target; median range across PFMs 8.6 points of between-tissue distance. Best-probe detectability: PanNormal image 0.92–1.00, feature 0.61–0.87; PLISM ≥ 0.99 for every method. | `results/summary_figure/summary.md` |
| RV09 | Gains depend on tissue for every method in every PFM (all BH q ≤ 0.034; scanner × tissue 8–43% of variance). For AKOYA, the frequency increment is larger in tissues with a stronger high-frequency deficit: Spearman −0.57 (UNI v1), −0.35 (Virchow2, H-optimus-1), −0.16 n.s. (UNI2-h). GT450: no association. | `results/gain_tissue_dependence/summary.md` |

## Manuscript assets

Written for the full revision of `00_manuscript/pannormal_scanner_batch_effects.tex` (2026-09-29).
Nothing is typed by hand; every asset is regenerated by the script named.

| Asset (in `00_manuscript/`) | Producer | Source results |
| --- | --- | --- |
| `tables/table_scanner_tissue_interaction.tex` (Table 1), `tables/table_scanner_tissue_interaction_full_supp.tex` | `manuscript_tables.py` | RV11; `analysis/paper/results/direct_slide_lmm/endpoint_summary.csv` |
| `tables/table_image_correction_three_axis.tex` (Table 2) | `manuscript_tables.py` | RV04, RV04b |
| `tables/table_cross_pfm_detectability.tex` (Table 3) | `manuscript_tables.py` | RV04, RV04b |
| `tables/table_detectability_probes_supp.tex`, `tables/table_cross_pfm_distance_supp.tex` | `manuscript_tables.py` | RV04b, RV04 |
| `tables/table_pfm_robustness_supp.tex` | `manuscript_tables.py` | RV13, RV14 |
| `figures/fig_scanner_phenotype_transfer.{png,pdf}` (Fig. 2) | `manuscript_figure_phenotype_transfer.py` | RV-P0d positions, RV12 |
| `figures/fig_augmentation_uni_trajectories_set20.{png,pdf}` (Fig. 3) | `manuscript_figure_augmentation_trajectories.py` | RV01 |
| `figures/fig_uni_band_sensitivity_set20.{png,pdf}` (Fig. 5), `figures/fig_{uni2,virchow2,hoptimus1}_band_sensitivity_set20_supp.{png,pdf}` | `manuscript_figure_band_sensitivity.py` | RV02 |
| `figures/fig_sensitivity_robustness.png` (Fig. 6), `figures/fig_tissue_vs_scanner_supp.png` | copied from `results/sensitivity_robustness/` | RV14 |
| `figures/fig_image_vs_representation.png` (Fig. 5) | copied from `results/summary_figure/` (`summary_figure.py`) | RV07 |
| `figures/fig_tissue_scanner_heatmap_supp.png`, `figures/fig_coherence_supp.png` | copied from `results/frequency_transfer_figures/` | RV12 |

The paper asset audit (`scripts/checks/audit_paper_assets.py`, `analysis/paper/artifact_map.json`) does not
yet list these assets; it is updated once the revised text and assets are final.
