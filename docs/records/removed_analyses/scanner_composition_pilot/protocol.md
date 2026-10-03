# RV16 Scanner composition of pretraining data (pilot)

**Status:** PROTOCOL. Written on 2026-10-01, before any model was trained or any outcome computed.
**Scope:** pilot for a possible RQ3 extension of `00_manuscript/pannormal_scanner_batch_effects.tex`.
Kept in its own file while RV15 is being added to `README.md`; a row and pointer will be added
to the README overview once both are in place.

## Question

Does the scanner composition of the pretraining data, by itself, change how a model responds to
spatial-frequency changes and to frequency correction, when tissue content, initialization and
training procedure are held fixed?

RQ3 of the manuscript shows that PFMs weight frequency bands differently and that the same
frequency correction moves them in opposite directions, but the comparison across released
models is observational: models differ in architecture, data, scale and objective. PanNormal
allows an intervention that no single-scanner dataset allows: the same registered tissue
locations exist on every scanner, so two pretraining sets can contain exactly the same tissue
and differ only in the scanner that imaged it.

## Design

| Item | Definition |
| --- | --- |
| Pretraining slides | PanNormal folds 1–4 (82 slides), every cached matched location (about 75,900) |
| Held-out slides | fold 0 (21 slides, 15 tissue types, all represented in the pretraining slides); fixed in advance as the first fold |
| Arms | **A**: AT2 image of every pretraining location. **B**: GT450 image of the same locations. Each location contributes exactly one image in each arm |
| Replicates | **A′**, **B′**: as A and B with data-order seed 1 instead of 0 (noise floor) |
| Initialization | init seed 0 in every arm; the step-0 state dict hash must be identical across arms |
| Pairing | within one data-order seed, A and B see the same location at the same step with the same crop coordinates, flips, colour-jitter and blur draws, iBOT masks and drop-path draws (per-sample RNG seeded from data-order seed, epoch and location; per-batch mask RNG seeded from the batch keys) |
| Model | ViT-S/14 (`dinov2.models.vision_transformer.vit_small`, LayerScale 1e-5, no registers), from scratch |

### Training recipe

Official DINOv2 code, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, vendored unmodified in
`third_party/dinov2_7764ea0f` except for an annotations-only Python 3.9 patch (see `SOURCE.txt`).
Configuration: `ssl_default_config.yaml` with the method settings of the released ViT-L/14
config (`train/vitl14.yaml`): Sinkhorn–Knopp centering, separate iBOT head, patch size 14,
local crops 98 px. Losses: DINO CLS loss, iBOT masked-patch loss and KoLeo (weight 0.1); heads
with 65,536 prototypes, bottleneck 256, hidden 2,048, three layers. Augmentation:
`DataAugmentationDINO` unchanged (2 global crops 224 px, scale 0.32–1; 8 local crops 98 px,
scale 0.05–0.32; colour jitter, grayscale, Gaussian blur and solarization exactly as the
released code applies them); iBOT masks on half of the global crops, ratio 0.1–0.5. Optimizer:
AdamW (0.9, 0.999), base lr 0.004 per 1,024 with the square-root rule (0.002 at batch 256),
10 warm-up epochs, cosine decay to 1e-6; weight decay 0.04 → 0.4; teacher momentum 0.992 → 1;
teacher temperature 0.04 → 0.07 over 30 epochs; layer-wise lr decay 0.9; patch-embedding lr
× 0.2; gradient clipping 3.0; last head layer frozen for the first epoch. Drop path 0.1
(uniform), the usual ViT-S value instead of the ViT-L default 0.3. 100 epochs of one pass over
the pretraining locations at batch 256. Input normalization uses the ImageNet statistics of the
released code in every arm (no per-arm statistics).

Implementation deviations from the released training loop, none of which changes the
objective: one GPU without FSDP; bf16 autocast instead of fp16 with a gradient scaler; the
student backbone is run separately on global and local crops instead of xFormers nested
tensors, and each head on each input separately instead of one block-diagonal call; a
world-size-1 `torch.distributed` group so that the Sinkhorn–Knopp reductions are identities.

Teacher backbones are saved at step 0 and after epochs 10, 25, 50 and 100. ~~Epoch 100 is
primary~~ **Epoch 25 is primary and epoch 50 is the confirmation look** (amendment D2); epochs
10 and 100 are descriptive only.

## Evaluation

* **Images:** `set20` locations of the 21 held-out slides (420 locations), the 81 RV13
  conditions rendered with the RV02/RV03 functions unchanged and fold-0 correction
  parameters: raw six scanners; Reinhard, frequency and colour + frequency from AT2 to each
  target; the 60 band manipulations (low-mid, mid, high; increase and decrease; doses 0.25 and
  0.50; colour-matched AT2 base for each target). For the tissue gate, the raw six-scanner
  images at the `set20` locations of the 82 pretraining slides.
* **Embedding:** teacher backbone, layer-normed CLS token; the 256-px field resized to 224 px
  with antialiased bicubic interpolation on float tensors, ImageNet normalization, fp32.
* **Definitions** (as RV02/RV03): representation shift = cosine distance between the
  colour-matched base and the band-changed image; target distance = cosine distance to the real
  target-scanner image at the same location; gain = reduction of target distance.
* **Statistics:** location means within slide and direction, the five directions equally
  weighted within slide, mean over the 21 held-out slides; 95% CIs from 2,000 bootstrap
  resamples of held-out slides (seed 20260924), one resample matrix shared by all statistics.

## Endpoints

**Gate (G), competence.** Tissue classification of held-out locations by 15-nearest-neighbour
cosine voting against the pretraining-slide `set20` locations of the same scanner, balanced
accuracy over the held-out tissue types, averaged over VERSA, AKOYA, S360 and S60 (scanners no
arm saw). Passes if, for every arm at epoch 100, the improvement over the step-0 model has a
95% CI above zero. If the gate fails, the primary endpoints are not interpreted.

**Primary (epoch 25; amendment D2).** Arm effect = B − A, estimated twice (B − A and B′ − A′).
Each endpoint is divided by the model's between-tissue distance (amendment D1 below).

* **P1, band weighting:** high-minus-low-mid representation shift at dose 0.25 (both signs, all
  five directions), the RV02 primary endpoint.
* **P2, direction asymmetry:** high-band shift after an increase minus after a decrease, dose
  0.25, all five directions. Expectation stated in advance: B − A < 0 (a GT450-trained model is
  displaced relatively less when high-band amplitude is raised toward its pretraining spectrum).
* **P3, frequency correction:** in the GT450 direction, additional target gain of colour +
  frequency over Reinhard alone (the manuscript's Fig. 4 quantity). Expectation: B − A > 0.

**Decision rule.** A primary endpoint is *reproduced* if (a) B − A and B′ − A′ have the same sign
and both 95% CIs exclude zero, and (b) the smaller of |B − A| and |B′ − A′| exceeds the larger of
the two noise differences |A′ − A| and |B′ − B|. Slide bootstrap CIs describe evaluation
sampling only; condition (b) is the test against training noise. Directions that disagree with
the stated expectation are reported as reproduced effects of the opposite sign, not discarded.
The pilot is **GO** for a continuous composition sweep if the gate passes and at least one
primary endpoint is reproduced; otherwise **NO-GO**, reported as a null result.

**Secondary (descriptive).** Shifts per band and sign, raw and divided by each model's
between-tissue distance (mean cosine distance between raw AT2 embeddings of different tissue
types on held-out slides); high-band target-direction gain per direction (RV02 definition);
P3 in the AKOYA direction and for frequency alone; normalized AT2–GT450 distance of raw pairs;
home-scanner tissue accuracy (queries and bank on AT2, then on GT450) as an arm × scanner
interaction; every endpoint at epochs 10, 25 and 50 and for the step-0 model.

## Files

| File | Role |
| --- | --- |
| `scanner_composition_pretrain.py` / `.sbatch` | one array task per arm (A, B, A′, B′) |
| `scanner_composition_embed.py` / `.sbatch` | render held-out conditions once, embed with every checkpoint |
| `scanner_composition_metrics.py` / `.sbatch` | gate, endpoints, decision, `summary.md` |
| `results/scanner_composition_pilot/` | checkpoints, logs, embeddings, metrics |

## Deviations

**2026-10-01, before any full-run outcome (amendment D1: primary endpoints normalized).** The
end-to-end smoke run (one pretraining slide, 100 short epochs; `results/.../smoke/`, not an
outcome) showed that (i) the step-0 teacher maps every image to almost the same CLS token
(LayerScale 1e-5 makes the residual branches negligible at initialization; between-tissue
distance ≈ 0), and (ii) the overall spread of the embedding can differ several-fold between
arms (smoke: 0.046 vs 0.008). A raw cosine shift then mixes frequency weighting with global
embedding scale, so B − A on raw shifts would be confounded. As in the manuscript's RQ3
(sensitivity divided by between-tissue distance) and Fig. 5A (gain as a share of between-tissue
distance), P1, P2 and P3 are therefore **divided by each model's between-tissue distance**:
the mean cosine distance between raw held-out locations of different tissue types, computed on
AT2 and on GT450 embeddings and averaged (symmetric for the two arms), recomputed in every
bootstrap resample from slide-pair means. Arm and noise differences, the decision rule and the
expectations apply to the normalized endpoints; raw values are reported as secondary. The
step-0 model remains the reference of the gate only; its normalized endpoints are undefined.
Also recorded: the profile run measured ~1.0 s per iteration (GPU-bound, 76 GB peak), about
8.5 h per arm.

**2026-10-01 14:10 EDT, before any full-run outcome (amendment D2: primary epoch 25).** The
pilot asks whether the phenomenon appears at all after short pretraining, not how it evolves.
On the investigators' decision, the primary analysis moves from epoch 100 to the epoch-25
checkpoint of the same runs (the runs themselves are unchanged; at epoch 25 the cosine
schedules are mid-way, lr still high and teacher momentum about 0.993). Epoch 50 is a
confirmation look: a primary endpoint reproduced at epoch 25 is reported as *confirmed* if it
also satisfies the decision rule at epoch 50 with the same sign. Epoch 10 is descriptive. The
runs are stopped after the epoch-50 checkpoint to free GPUs for further experiments, so no
epoch-100 checkpoint is produced (the pilot only tests feasibility). GO/NO-GO is decided at
epoch 25 and does not change with the epoch-50 look. At the time
of this amendment no full-run embedding or metric had been computed (training at epoch ≈ 12 in
three arms and ≈ 7 in A′); the epoch-10 checkpoint was judged too early to be informative.
Operational: arm A′ ran at ~55% of the speed of the other arms on its node; it is cancelled after
its epoch-10 resumable state is written and resumed on another node (same code, same state,
same sample keys).

## Results

**Epoch 25 (primary), 2026-10-01.** Gate passed in every arm (tissue kNN on unseen scanners
+0.15 to +0.23 over step 0). **NO-GO**: no primary endpoint met the decision rule. P1 differed in
the seed-0 pair only (B − A −0.0052, B′ − A′ −0.0001) and the GT450 replicates differed by as much
(B′ − B +0.0056); P2 and P3 were near zero in every arm. All four models responded to the high
band about ten times less than to the low–mid band (normalized shift ≈ 0.0005 vs 0.005–0.011),
so the high-frequency hypotheses were not testable at this stage of training. Exploratory
(`scanner_composition_secondary.py`, not part of the decision): three pre-specified secondary
endpoints met the decision rule — home-scanner tissue kNN crossover (B − A −0.067, B′ − A′
−0.080), larger normalized AT2–GT450 distance in GT450-trained models (+0.137, +0.074) and a
smaller AKOYA colour + frequency increment in GT450-trained models (−0.012, −0.013). Files:
`results/scanner_composition_pilot/metrics_e025/`, `secondary_e025/`.

**Epoch 50 (confirmation look), 2026-10-01.** The decision stays NO-GO (amendment D2). P1 and
P2 did not meet the rule. P3 met the rule at epoch 50 but not at epoch 25 and with the sign
opposite to the stated expectation: adding frequency matching to Reinhard in the GT450
direction moved the GT450-trained models away from the real GT450 image (levels B −0.0036,
B′ −0.0038; A −0.0002, A′ −0.0005; B − A −0.0034, B′ − A′ −0.0033; noise ≤ 0.0003). High-band
sensitivity was still about a tenth of the low–mid sensitivity in every arm. Exploratory: S1
(−0.068, −0.123), S2 (+0.056, +0.047) and S3 (−0.019, −0.020) again met the rule. Training
stopped after epoch 50 in every arm; no epoch-100 checkpoint exists. Files:
`metrics_e050/`, `secondary_e050/`.
