# RV18 Continued pretraining of UNI on AT2/GT450 mixtures (overnight pilot)

**Status:** PROTOCOL for round 1. Written on 2026-10-01 (23:30 EDT), before any RV18 model was
trained or any outcome computed. Round 2 is chosen from the round-1 outcome and gets its own
dated section here before it is run.

## Why

RV16/RV17 showed, in small from-scratch models, that the AT2 share of the pretraining data
changes scanner-response endpoints in a dose-dependent way:
* S2: normalized AT2–GT450 distance, lower with more AT2.
* S3: AKOYA colour + frequency increment, higher with more AT2.

Those models barely use high frequencies (band ratio R ≈ 0.1–0.15) and carry little tissue
information, so they cannot speak for real PFMs or for band weighting. UNI uses high
frequencies (R ≈ 1.3 in RV02) and carries tissue information. Continuing its self-supervised
training on AT2/GT450 mixtures asks whether the same composition effects appear in a real PFM.
It also asks whether band weighting moves, and whether a within-model change of sensitivity is
accompanied by a change of scanner distance. RQ3 shows that relation only across models.

## Design (round 1)

| Item | Definition |
| --- | --- |
| Initialization | UNI v1 (`MahmoodLab/uni`, ViT-L/16) loaded into DINOv2 `vit_large` (patch 16, img 224, LayerScale, no registers). The state-dict keys match one to one; `mask_token` (absent in UNI) starts at zero. Parity with the timm UNI used in the paper: cosine ≥ 0.999 |
| Data | as RV17: pretraining folds 1–4 (75,891 locations); held-out fold 0 |
| Composition (4 arms) | AT2 share p ∈ {1.0, 0.8, 0.2, 0.0}, with the RV17 per-location assignment (seed 17) |
| Augmentation | as RV17: `DataAugmentationDINO` with Gaussian blur replaced by a draw-only identity |
| Seeds | init (heads) 0, data-order 0; all arms paired as in RV16/RV17 |
| Objective | DINOv2 (DINO + iBOT + KoLeo, Sinkhorn–Knopp, separate iBOT head), heads newly initialized, 16,384 prototypes, global crops 224, local crops 96 (patch 16) |
| Schedule | 3 epochs at batch 256 (888 iterations). Backbone frozen for the first 0.5 epoch while the heads train. Its lr then ramps up over 0.25 epoch to a peak of 1e-4 at the top block (layer-wise decay 0.9; patch embedding × 0.2) and follows the cosine schedule to 1e-6. Head lr peaks at 2e-3 after a 0.5-epoch warm-up. Last head layer frozen for 0.5 epoch. Teacher temperature 0.04 → 0.07 over 1 epoch. Teacher momentum 0.992 → 1. Weight decay 0.04 → 0.4. Drop path 0.1. bf16 with activation checkpointing |
| Checkpoints | teacher backbone at epochs 0 (= UNI), 1, 2, 3; **epoch 3 primary**, epoch 1 for consistency |

The learning rate is the one free choice. It is set for moderate adaptation without losing
tissue information, and tissue information is checked (below).

## Evaluation

As RV17: the `set20` locations of the 21 held-out slides, the 81 RV13 conditions, and the
pretraining-slide bank. The models are embedded with the paper's UNI path: bicubic 256 → 224
without antialiasing, ImageNet normalization, teacher layer-normed CLS, fp32 with TF32 allowed.

Endpoints: the RV17 set, plus
* normalized band shifts (low–mid, mid, high; dose 0.25);
* S6, the mean normalized AT2→target distance over the five target scanners (the manuscript's
  normalized scanner distance on held-out slides);
* tissue information: between-tissue distance and kNN on unseen scanners.

The RV17 endpoints are:
* S1: home-scanner tissue kNN, AT2 accuracy − GT450 accuracy.
* S2: normalized AT2–GT450 distance of raw pairs.
* S3: AKOYA colour + frequency increment over Reinhard.
* P3: the same increment in the GT450 direction.
* S5: AT2-vs-GT450 linear-probe detectability.
* R: band ratio, high ÷ low–mid normalized shift.

Statistics as RV16/RV17.

**Primary reading (epoch 3).** For S2, S3 and R: the trend over p (slope and 95% CI), its sign
relative to RV17, and its consistency at epoch 1. Also each arm minus UNI, for the effect of
continued training as such. Tissue retention is required. If kNN on unseen scanners or
between-tissue distance falls well below UNI's, an endpoint change is not interpreted as a
composition effect. There is no replicate in round 1, so a pattern counts as "interesting" only
if the trend CI excludes zero at epoch 3, has the same sign at epoch 1, and tissue is retained.
Round 2 then replicates it (data-order seed 1) and widens the conditions.

## Files

`uni_continued_pretrain.py`/`.sbatch`, `uni_continued_embed.py`/`.sbatch`,
`uni_continued_metrics.py`/`.sbatch`; outputs in `results/uni_continued/`.

## Deviations

**2026-10-02 00:30 EDT (operational).** Arm p080 completed all 888 iterations but was killed for
host memory (96 GB) while writing its epoch-3 checkpoint. The DataLoader prefetch buffers (22
workers × 4 batches) were the largest consumer. p080 is re-run with the same seeds and code,
prefetch factor 2 and 128 GB. The other arms are unaffected.

**2026-10-02 00:55 EDT (bug; round 1 invalid).** Every round-1 checkpoint equalled UNI: endpoint
differences were about 1e-5. The training script froze the backbone for the head warm-up
*before* building the optimizer. `dinov2.utils.param_groups.get_params_groups_with_decay` skips
parameters that do not require grad, so the backbone was never in the optimizer. Only the heads
trained, and the teacher backbone stayed UNI. `metrics_prelim_three_arms/` is therefore not an
outcome of the design. The script now builds the optimizer first, asserts that every backbone
parameter is in it, logs the relative change of the student and teacher backbones from UNI at
every log step, and stops if the student backbone has not moved 25 iterations after unfreezing.
All four arms are re-run with the same design, seeds, prefetch factor 2 and 128 GB.

While p080 re-runs, the three finished arms (p100, p020, p000) are embedded and summarized as a
preliminary, descriptive look (`metrics_prelim_three_arms/`). The round-1 reading uses all four arms.

## Round 2 (written 2026-10-02 02:15 EDT, after the corrected round-1 outcome, before round 2 is run)

**What round 1 showed (single seed).**

* Tissue retention: kNN on unseen scanners fell from UNI's 0.569 to 0.544–0.556 at epoch 1 and
  to 0.483–0.501 at epoch 3. The backbones moved ~1.3% (student) and ~0.9% (teacher).
* Trends over the AT2 share p whose CIs excluded zero at epoch 3 with the same sign at epoch 1:
  * band ratio R: slope −0.104 at epoch 1, −0.086 at epoch 3. More AT2 means relatively less
    high-frequency weighting.
  * S2: +0.018 / +0.032.
  * S6: +0.011 / +0.023.
  * S3: −0.004 / −0.007.
* P3: slope −0.006 at epoch 3, same sign at epoch 1 but with a CI that includes zero.
* S2 and S3 have signs opposite to RV16/RV17.

**Round 2: data-order replicate.** The same four arms, design and code are re-run with
data-order seed 1 (`train/<arm>_s1`); the head init seed is unchanged. Each arm's endpoint is
embedded at epochs 1 and 3.

**Reading.** A round-1 trend counts as *replicated* if the seed-1 slope has the same sign and a
95% CI excluding zero at epoch 3, and the same sign at epoch 1. Training noise is the mean over
the four arms of |E(seed 1) − E(seed 0)|. The trend must exceed it, with both measured over the
full range of p. Endpoints for this reading: R, S2, S6, S3, P3. Tissue retention is reported
alongside; epoch 1 is preferred for interpretation where epoch-3 tissue loss is large.

## Results (rounds 1 and 2), 2026-10-02

**QC.**
* Epoch-0 teachers are identical in every arm and equal UNI (conversion parity 0.9999998).
* Band renders are identical to RV02 (21/21 slides).
* The student backbones moved 1.32–1.33% from UNI and the teachers 0.88–0.89%, in both seeds.

**Tissue retention.** kNN on unseen scanners fell from 0.569 (UNI) to 0.54–0.56 at epoch 1 and to
0.47–0.50 at epoch 3 in every arm. Epoch 1 is therefore preferred for interpretation, as
pre-specified.

**Round-2 reading** (`results/uni_continued/replicate/summary.md`; slopes over the full range of
p, seed 0 / seed 1):

* **Replicated by the rule at epoch 3, and in agreement at epoch 1:**
  * S2: +0.032 / +0.033; at epoch 1 +0.018 / +0.018; noise 0.001. More AT2 gives a larger
    AT2–GT450 distance.
  * S6: +0.023 / +0.021; at epoch 1 +0.011 / +0.011. More AT2 gives a larger distance from AT2 to
    every other scanner.
  * S3: −0.007 / −0.005; at epoch 1 −0.004 / −0.005. More AT2 gives a smaller AKOYA
    colour + frequency increment.
  * P3: −0.006 / −0.004; at epoch 1 −0.003 / −0.003, seed-0 CI including zero. More GT450 makes
    GT450-direction frequency matching less harmful.
  * Low–mid band sensitivity: +0.0004 / +0.0004; at epoch 1 +0.0003 / +0.0003.
* **R, not replicated by the epoch-3 rule:** −0.086 / −0.025, seed-1 CI including zero; noise
  0.031. At epoch 1, where tissue is retained, both seeds agree: −0.104 / −0.083, both CIs
  excluding zero, noise 0.009. More AT2 shifts UNI's weighting toward the low–mid band, through
  higher low–mid sensitivity rather than lower high-band sensitivity.
* **Continued training as such**, in every arm: S6 rose (+0.014 to +0.025 at epoch 1; +0.02 to
  +0.05 at epoch 3) and tissue kNN fell.
* **Signs relative to RV16/RV17:** S2 and S3 have the opposite sign. P3 has the sign originally
  expected in RV16.

Round 3 (blur-on standard recipe, p = 0.5) was planned but not run: the session ended after round 2.

## Round 3: dose–response at epoch 1 (written 2026-10-02 08:30 EDT, before round 3 is run)

**Why.** The band-ratio trend replicated at epoch 1 with two seeds, where tissue information is
retained. A finer grid of AT2 shares tests whether R follows the share smoothly, which is a
dose–response rather than a four-point contrast. The investigators' aim is to show the
direction reproducibly, not to estimate it precisely.

**Design.** AT2 share p ∈ {0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9} (arms `p010` … `p090`; nested RV17
assignment, seed 17), data-order seeds 0 and 1. That is 14 runs. Everything else is as rounds 1
and 2, including the 3-epoch schedule. Each run stops after its epoch-1 checkpoint
(`--stop-epoch 1`), so these checkpoints come from the same procedure as the round-1/2 epoch-1
checkpoints. The existing epoch-1 checkpoints at p = 1.0, 0.8, 0.2 and 0.0 (both seeds) complete
an 11-share grid.

**Primary endpoint.** Band ratio R at epoch 1, against p.

**Secondary endpoints.** Low–mid and high normalized shifts, S2, S6, S3, P3 and tissue kNN.

**Reading.**
* Per seed: the least-squares slope of R on p over the 11 shares, with a slide-bootstrap 95% CI,
  and Spearman ρ.
* Pooled: the slope over the 22 points.
* *Reproduced:* both per-seed slopes are negative (the round-1/2 direction) and their CIs
  exclude zero.
* The scatter of R against p (both seeds, UNI as reference) is the figure.

**Round 3 results (2026-10-02 09:40 EDT; `results/uni_continued/dose_e001/`).**

* **QC:** epoch-0 teachers identical across arms; band renders identical to RV02; 14 new runs.
* **R — reproduced:** R falls monotonically with the AT2 share in both seeds.
  * From 1.383 → 1.282 (seed 0) and 1.370 → 1.285 (seed 1); UNI 1.316.
  * Slope −0.106 [−0.128, −0.087] and −0.081 [−0.105, −0.061]; Spearman ρ −1.00 in each seed.
  * Pooled −0.093 [−0.116, −0.075], ρ −0.99 over 22 models.
* **Driven by low–mid sensitivity:** low–mid sensitivity rises with p (ρ +0.91 / +0.95); high-band sensitivity shows no trend (ρ −0.40 / +0.05).
* **Secondary endpoints are dose-dependent in both seeds:**
  * S2: slope +0.018, ρ ≈ +0.9. Flat up to p ≈ 0.4, then rising.
  * S6: +0.011, ρ ≈ +0.95. Flat up to p ≈ 0.3, then rising.
  * S3: −0.004, ρ ≈ −0.9.
  * P3: −0.0045, ρ −0.74. Monotone from p = 0 to 0.9; p = 1.0 is back near the p = 0.2 level in both seeds.
* **Tissue kNN** declines slightly with p (−0.011 over the full range).

## Round 3 results (2026-10-02 09:40 EDT)

**QC.** 14 new runs, all stopped at iteration 296 (epoch 1). Student backbones moved 0.76–0.77%
from UNI, the same as the round-1/2 epoch-1 checkpoints. Epoch-0 teachers are identical.
Band renders are identical to RV02.

**Primary: reproduced.** R fell monotonically with the AT2 share in both seeds:
* seed 0: slope −0.106 [−0.128, −0.087], Spearman ρ −1.00;
* seed 1: slope −0.081 [−0.105, −0.061], ρ −1.00;
* pooled: −0.093 [−0.116, −0.075], ρ −0.99 over 22 models.

R went from 1.38/1.37 at p = 0 to 1.28/1.29 at p = 1, crossing UNI's 1.32 near p ≈ 0.65. The
change came from low–mid sensitivity, which rose monotonically with p (pooled slope +0.0003,
ρ +0.92). High-band sensitivity showed no trend (ρ −0.18); it was raised by continued training
in every arm.

**Secondary.**
* **S2 and S6:** flat for p ≤ 0.4 and rising for p ≥ 0.6, so scanner separation grows only when
  AT2 dominates (pooled slopes +0.018 and +0.011).
* **S3:** decreasing with p (−0.004).
* **P3:** decreasing with p up to 0.9 (−0.005); p = 1.0 was off-trend in both seeds.
* **Tissue kNN:** slightly lower with more AT2 (−0.011), and below UNI in every arm.

Files: `results/uni_continued/dose_e001/` (`summary.md`, `band_ratio_vs_share.png`,
`secondary_vs_share.png`).
