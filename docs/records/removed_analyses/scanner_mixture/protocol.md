# RV17 AT2/GT450 mixtures in pretraining, without blur augmentation (ViT-Tiny pilot)

**Status:** PROTOCOL. Written on 2026-10-01, before any RV17 model was trained or any outcome computed.
**Follows:** RV16 (`scanner_composition_pilot_protocol.md`). This is a feasibility pilot: four runs in parallel, one seed each.

## Why

RV16 (ViT-S/14, single-scanner pretraining, standard DINOv2 recipe) was NO-GO on its
band-weighting endpoints. Every arm responded to the high band about ten times less than to
the low–mid band, so those endpoints sat at a floor. The standard augmentation blurs 90% of
second global crops and 50% of local crops, so the objective rewards invariance to losing high
frequencies. Three scanner-response endpoints did differ reproducibly between AT2- and
GT450-pretrained models (exploratory).

Single-scanner pretraining also cannot test the mechanism most likely to put scanner signals
into real PFMs. When scanners are mixed, scanner cues can become a shortcut for instance
discrimination.

The goal here is to see quickly whether effects of the scanner distribution reproduce, not to
reproduce PFM-scale training. With the standard recipe, high-frequency use would only appear
after training long enough for the model to become robust to the blur augmentation; that is not
needed for this question. The blur augmentation is therefore switched off (investigators'
decision), and the comparison with RV16 is descriptive.

## Questions

* **Q1 (descriptive).** Without Gaussian-blur augmentation, do the models use high frequencies,
  so that band endpoints become testable?
* **Q2 (primary).** Do scanner-response endpoints change with the AT2 share of the pretraining
  data? Do mixtures follow the pure sets (dose–response) or depart from both (shortcut)?

## Design

| Item | Definition |
| --- | --- |
| Data, held-out set, evaluation | as RV16: pretraining folds 1–4 (75,891 locations); held-out fold 0 `set20` (420 locations); the 81 RV13 conditions; raw six-scanner bank of the pretraining slides |
| Composition (4 arms) | AT2 share p ∈ {1.0, 0.8, 0.2, 0.0}, i.e. AT2:GT450 = 100:0, 80:20, 20:80, 0:100. Each pretraining location contributes one image: AT2 if u < p, else GT450, with u ~ U(0, 1) drawn once per location (seed 17). The sets are nested and the tissue content is identical in every arm |
| Augmentation | `DataAugmentationDINO` with the Gaussian blur replaced by an identity that consumes the same random draws; every other augmentation is unchanged. No blur-on arm in RV17 |
| Seeds | init seed 0 and data-order seed 0 in every arm, so within the experiment all arms see the same location at the same step with the same crops, jitter, solarization, masks and drop-path draws. There is no replicate run |
| Model | ViT-Tiny/14: DINOv2 `DinoVisionTransformer` with embedding 192, depth 12, 3 heads, MLP ratio 4, otherwise as `vit_small` |
| Recipe | as RV16 (official DINOv2 code, Sinkhorn–Knopp, separate iBOT head, KoLeo, batch 256, bf16), except: DINO and iBOT heads with 16,384 prototypes (compute and 40 GB GPUs; same in every arm); **25 epochs with full schedules**, warm-ups scaled to the length (lr 3 epochs, teacher temperature 8 epochs, last layer frozen 1 epoch) |
| Checkpoints | teacher at epochs 0, 10 and 25; **epoch 25 is primary** |

Normalization (RV16 amendment D1) and the statistics (21 held-out slides, 2,000 slide resamples, seed 20260924) are unchanged.

## Endpoints

Per model, at epoch 25:

* **S1:** home-scanner tissue kNN, AT2 accuracy − GT450 accuracy.
* **S2:** normalized AT2–GT450 distance of raw pairs.
* **S3:** AKOYA colour + frequency increment over Reinhard (normalized).
* **P3:** the same increment in the GT450 direction.
* **S5:** scanner detectability, the balanced accuracy of a linear probe separating raw AT2 from
  raw GT450 held-out embeddings. Standardized logistic regression, C = 0.1, liblinear, five-fold
  cross-validation grouped by slide, averaged per slide.
* **Band ratio R:** high-band ÷ low–mid-band normalized shift, dose 0.25.

**Q1 (descriptive).** R in every arm. Band endpoints are *testable* if mean R ≥ 0.5 (RV16
standard recipe: ≈ 0.1; released PFMs 0.7–2.4). RV16 differs in model size, so the comparison
is not a controlled test of the blur augmentation.

**Q2 (primary).** For each E ∈ {S1, S2, S3, P3, S5}:

* **Trend:** least-squares slope of E on p across the four arms, with a slide-bootstrap 95% CI.
  A trend is *present* if the CI excludes zero and E(p = 1.0) and E(p = 0.0) differ in the same
  direction as the slope. Expected signs from RV16: S1 > 0, S2 < 0, S3 > 0, P3 > 0.
* **Mixture contrast:** M = mean E(p = 0.8, 0.2) − mean E(p = 1.0, 0.0), with its CI. The
  shortcut account predicts M > 0 for S2 and S5 (mixtures separate the scanners more than either
  pure set).

There is no replicate, so training noise is not estimated within RV17. For reference, the RV16
seed-to-seed differences of the same endpoints at RV16's primary epoch 25 are reported next to
every effect. The noise reference of an endpoint is the larger of |A′ − A| and |B′ − B|. For S5,
which RV16 did not compute, it is computed here from the RV16 embeddings in the same way. An
effect (|slope| over the full range of p, or |M|) smaller than its noise reference is read as not
distinguishable from training noise.

**GO** for a replicated sweep (more seeds and shares, and a blur-on control) if at least one Q2
trend or mixture contrast is present and exceeds the RV16 seed noise of that endpoint.

## Files

`scanner_mixture_pretrain.py`/`.sbatch` (array 0–3), `scanner_mixture_embed.py`/`.sbatch`,
`scanner_mixture_metrics.py`/`.sbatch`; outputs in `results/scanner_mixture/`.

## Deviations

(none yet)

## Results

**Epoch 25 (primary), 2026-10-01.** QC: step-0 teachers identical in the four arms, band renders
identical to RV02 on 21/21 slides; realized AT2 shares 1.000, 0.802, 0.201, 0.000.

* **Decision: GO.** Two trends exceeded their RV16 noise reference with the sign expected from RV16:
  * S2 (normalized AT2–GT450 distance; 0.471, 0.484, 0.554, 0.596 for p = 1.0 → 0.0): slope
    −0.122 [−0.138, −0.106], noise reference 0.052.
  * S3 (AKOYA increment; 0.072, 0.054, 0.049, 0.041): slope +0.025 [+0.015, +0.034], noise
    reference 0.0012.
* **Other endpoints:**
  * P3: trend present (−0.0041 [−0.0070, −0.0011]) but with the sign opposite to the RV16
    expectation, and not present at epoch 10. It is not consistent across experiments.
  * S1: same direction as RV16, CI includes zero (+0.031 [−0.017, +0.102]).
  * S5: mixture contrast +0.022 [+0.011, +0.034], just above its noise reference (0.018); it is
    the only result in the direction of the shortcut account, and it is absent at epoch 10.
  * S2 mixture contrast: −0.015 [−0.032, +0.003]; mixtures follow the dose–response, not the
    shortcut account.
* **Q1:** band ratio R 0.137, 0.204, 0.134, 0.129 (mean 0.151). Below 0.5, so band endpoints
  remain untestable even without blur augmentation at this model size and length.
* **Descriptive, epoch 10:** S2 slope −0.119 and S3 slope +0.015 (both CIs exclude zero); S1
  slope +0.040 [+0.013, +0.071].
* Single seed; no within-RV17 noise estimate. Files: `results/scanner_mixture/metrics/`.
