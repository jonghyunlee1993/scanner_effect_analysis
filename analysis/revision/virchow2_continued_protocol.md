# RV19 Continued pretraining of Virchow2 on AT2/GT450 mixtures: band ratio over the AT2 share

**Status:** PROTOCOL. Written on 2026-10-02 (09:45 EDT), before any RV19 model was trained or any outcome computed.
**Follows:** RV18 (`uni_continued_protocol.md`).

## Why

In RV18, continuing UNI's self-supervised training on AT2/GT450 mixtures shifted its band ratio
monotonically with the AT2 share. This held over 11 shares and two seeds at epoch 1: R went from
1.38 to 1.28, with UNI at 1.32. More AT2, the scanner with less high-frequency content, raised
low–mid sensitivity relative to high-band sensitivity.

UNI starts high-frequency-weighted (R ≈ 1.3). Virchow2 starts low-frequency-weighted: in the
manuscript's RQ3 it responds most to the low–mid band. RV19 asks whether the scanner composition
of the pretraining data moves the band weighting in the same direction for a PFM with the
opposite starting profile. A same-direction result supports a model-general claim; an
opposite or flat result limits the claim to UNI.

## Design

| Item | Definition |
| --- | --- |
| Initialization | Virchow2 (`paige-ai/Virchow2`, ViT-H/14, 4 register tokens, SwiGLU, LayerScale), the checkpoint used in the manuscript. It is loaded into the vendored DINOv2 `DinoVisionTransformer` (embedding 1280, depth 32, 16 heads, SwiGLU with hidden 3416 per SwiGLU branch, 4 registers). Keys are renamed `mlp.fc1` → `mlp.w12` and `mlp.fc2` → `mlp.w3`; the register positions in timm's `pos_embed` are folded into `register_tokens`, which is exact at 224 px; `mask_token` starts at zero. Parity with the timm Virchow2 of the manuscript (CLS ⊕ mean patch token): cosine ≥ 0.999 |
| Data, assignment, augmentation, pairing | as RV18: pretraining folds 1–4, nested per-location AT2/GT450 assignment (seed 17), blur-off augmentation, shared draws within a data-order seed |
| Shares | AT2 share p ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} |
| Seeds | data-order seeds 0 and 1 (12 runs) |
| Objective and schedule | as RV18: DINO + iBOT + KoLeo, Sinkhorn–Knopp, new heads (16,384 prototypes); 3-epoch schedule with backbone frozen for the first half epoch, backbone peak lr 1e-4 (layer-wise decay 0.9), head peak lr 2e-3; global crops 224, local crops 98 (patch 14). Each run stops after its epoch-1 checkpoint, as in RV18 round 3 |
| Evaluation | as RV18 (21 held-out slides, 81 conditions, shared slide resamples). The embedding is the manuscript's Virchow2 path: bicubic 256 → 224 without antialiasing, ImageNet normalization, bf16 autocast, CLS ⊕ mean of patch tokens (registers excluded), unit norm |

## Endpoints and reading

* **Primary:** the band ratio R (high ÷ low–mid normalized shift, dose 0.25) at epoch 1, against p.
* **Reproduced in Virchow2:**
  * Both per-seed least-squares slopes of R on p are negative (the RV18 direction) with 95%
    slide-bootstrap CIs excluding zero.
  * The opposite sign with CIs excluding zero is reported as a reversal.
  * Anything else is reported as no detectable effect.
* **Secondary:** low–mid and high normalized shifts (which band moves), S2, S6, S3, P3, and
  tissue kNN on unseen scanners.
* **QC (fatal):**
  * conversion parity;
  * epoch-0 teachers identical across arms;
  * the student backbone moves after unfreezing;
  * band renders identical to RV02.

## Files

Training, embedding and analysis reuse the RV18 scripts with `--base virchow2`. Outputs go to
`results/virchow2_continued/`.

## Deviations

**2026-10-02 10:20 EDT (operational).** The seed-1 array waited for node CPUs, so its request was
reduced from 24 CPUs and 128 GB to 8 CPUs and 80 GB per task. Fewer DataLoader workers do not
change the training data: every sample's augmentation is seeded by its key and the sampler order
is fixed, so only throughput changes.

## Contingency: epoch 3 (written 2026-10-02 11:38 EDT, before any RV19 outcome was seen)

If the epoch-1 reading is not *reproduced*, that is, if not both per-seed R slopes are negative
with CIs excluding zero, the 12 runs are repeated with the same seeds and code without the epoch-1
stop. They complete the 3-epoch schedule, and the same reading is applied at epoch 3 (primary)
and epoch 2.

The epoch-1 runs saved no optimizer state, so the runs restart from Virchow2. Their first 296
iterations reproduce the epoch-1 runs up to GPU nondeterminism.

**Why:** at epoch 1 the Virchow2 teacher moved about four times less than UNI's (relative change
0.07% vs 0.27%). With 32 blocks and layer-wise decay 0.9, the lower blocks receive smaller
updates. A null at epoch 1 could therefore reflect too little adaptation rather than no effect.
The epoch-3 reading reports tissue kNN alongside, as in RV18.


## Results, epoch 1 (2026-10-02 12:07 EDT; `results/virchow2_continued/dose_e001/`)

**QC:** conversion parity 0.9999998; epoch-0 teachers identical; band renders identical to RV02.

**Not reproduced.** The R slopes over p are +0.001 [−0.017, +0.015] and +0.011 [−0.001, +0.023]
(pooled +0.006 [−0.005, +0.015], ρ +0.14). Continued training raised R in every arm, from
Virchow2's 0.655 to 0.69–0.73.

With more AT2, Virchow2's low–mid and high-band sensitivities both rose (slopes +0.0005 and
+0.0004, both seeds), so the ratio did not move. UNI behaved differently: only its low–mid
sensitivity rose.

Secondary endpoints:
* S6 rose with p, as in UNI (+0.010).
* S3 (+0.006) and P3 (+0.002) have signs opposite to UNI.
* S2 is flat.
* Tissue kNN fell slightly with p.

As pre-registered (contingency above), the 12 runs are repeated over the full 3-epoch schedule
in `results/virchow2_continued_e3/`.

**Interim look (written 2026-10-02 14:05 EDT, before any epoch-2/3 outcome).** Seed 0 finishes about
two hours before the last two seed-1 runs. Its epoch-3 checkpoints are embedded and summarized as
soon as they exist (`dose_e003_seed0_interim/`). The look is descriptive only; the reading uses
both seeds.


**Interim look, seed 0, epoch 3 (2026-10-02 15:10 EDT; `results/virchow2_continued_e3/dose_e003_seed0_interim/`).**
Descriptive only; the reading waits for seed 1.
* R slope +0.018 [−0.008, +0.041], ρ +0.09. R is 0.79–0.86 in every arm, up from Virchow2's 0.655.
* Both band sensitivities rose with p again (low–mid +0.0021, high +0.0019; ρ +0.37, with p = 0.4
  high in both).
* S6 +0.024 (ρ +0.94), S3 +0.026 (ρ +1.00), P3 +0.005 (ρ +0.94); S2 flat.
* Continued training itself moved Virchow2 far more than the share did:
  * band sensitivities rose about 5-fold over Virchow2 in every arm;
  * S2 and S6 about doubled;
  * tissue kNN on unseen scanners fell from 0.55 to 0.42–0.47.


## Results, epoch 3 (2026-10-02 18:05 EDT; `results/virchow2_continued_e3/dose_e003/`)

**Not reproduced, and no reversal: no detectable effect.**
* The R slopes over p are +0.018 [−0.008, +0.041] (seed 0) and −0.017 [−0.044, +0.010] (seed 1).
  Pooled, the slope is +0.000 [−0.018, +0.017], ρ −0.10.
* R is 0.79–0.86 in every arm, up from Virchow2's 0.655, with no order over p.

The bands do not move consistently with p:
* In seed 0, both band sensitivities rose with p (low–mid +0.0021, high +0.0019).
* In seed 1, both fell (−0.0004 and −0.0006; only the high-band CI excludes zero).
* Within a seed the two bands moved together, so R stayed flat in both seeds.

Secondary endpoints that agree across seeds:
* S6 rises with p: +0.024 and +0.045 (pooled +0.035 [+0.029, +0.041], ρ +0.90).
* S3 (+0.022) and P3 (+0.006) also rise with p.
* S2 has opposite signs in the two seeds (−0.007, +0.016).
* Tissue kNN on unseen scanners falls with p: −0.022 and −0.055 (pooled −0.039 [−0.063, −0.013]).

Continued training as such dominates, as in the seed-0 interim. Against Virchow2, in every arm:
* low–mid sensitivity is 5–7-fold higher (0.0023 → 0.011–0.015);
* high-band sensitivity is 6–9-fold higher (0.0015 → 0.009–0.013);
* S6 is about twice as high (0.128 → 0.25–0.30);
* tissue kNN on unseen scanners is lower (0.55 → 0.38–0.47).

Epoch 2 (secondary reading) follows in `dose_e002/`.


## Results, epoch 2 (2026-10-02 18:40 EDT; `results/virchow2_continued_e3/dose_e002/`)

**Not reproduced, and no reversal: no detectable effect.**
* The R slopes over p are +0.024 [+0.004, +0.045] (seed 0) and −0.021 [−0.053, +0.016] (seed 1).
  Pooled, the slope is +0.002 [−0.018, +0.022], ρ −0.04.
* Only seed 0's CI excludes zero, and its sign is the opposite of seed 1's. The reversal reading
  needs both seeds, as the reproduction reading does.

Epoch 2 repeats epoch 3 in every endpoint:
* Both bands rose with p in seed 0 and fell in seed 1.
* S6 rose with p (pooled +0.032 [+0.027, +0.037]), as did S3 (+0.021) and P3 (+0.004).
* Tissue kNN on unseen scanners fell with p (pooled −0.038 [−0.060, −0.011]).
* In every arm, against Virchow2:
  * low–mid sensitivity is 0.010–0.014;
  * high-band sensitivity is 0.008–0.012;
  * S6 is 0.24–0.29;
  * tissue kNN is 0.39–0.46.

## Reading (2026-10-02 18:40 EDT)

**Virchow2's band ratio did not move with the AT2 share at epoch 1, 2 or 3.**

The contingency rules out too little adaptation as the reason. By epoch 3, continued training had
raised both band sensitivities 5–9-fold and moved the band ratio from 0.655 to about 0.8 in every
arm; the share of AT2 images still left the ratio flat.

In both PFMs, more AT2 raised S6 (the normalized distance from AT2 to the other scanners), and in
both, more AT2 lowered tissue kNN on unseen scanners. What differs is the band ratio:
* In UNI, low–mid sensitivity rose with the AT2 share and high-band sensitivity did not.
* In Virchow2, the two bands moved together.
