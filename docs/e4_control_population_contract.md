# E4 control-population execution contract

**Status:** frozen before E4 outcome access on 2026-08-03  
**Parent decision:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md)

## Input population

- 109 physical slides, six scanners and 100 matched locations per scanner--slide.
- Native-WSI-only, explicit-AA RGB grid at 0.5052 µm/px.
- ResNet50, UNI v1, CONCH v1 and Virchow2 with the already frozen TRIDENT encoder,
  checkpoint and evaluation-transform contracts.
- Raw embeddings are reused from the audited E0 population rather than recomputed.

## Frozen image controls

Every control uses the existing four-level `FixedLaplacianPyramid`, retains the source
coarsest coefficient and performs one explicit RGB-range clamp after reconstruction.

| Condition | Detail-band operation | Role |
|---|---:|---|
| `hf_retention_0p75` | source × 0.75 | partial attenuation |
| `hf_retention_0p50` | source × 0.50 | partial attenuation |
| `hf_retention_0p25` | source × 0.25 | partial attenuation |
| `hf_retention_0p00` | source × 0.00 | complete-blur collapse control |
| `hf_boost_1p25` | source × 1.25 | sharpening |
| `hf_boost_1p50` | source × 1.50 | sharpening |
| `hf_boost_2p00` | source × 2.00 | sharpening/noise amplification |
| `registered_loo_hf_mean_0p25` | 0.75 source + 0.25 mean of the other five scanners at the same location | paired image-space oracle |
| `global_train_hf_mean_1p00` | complete replacement by the training-slide global coefficient mean | phase-cancellation negative control |

The global coefficient mean is fitted separately at the 224, 256 and 512 px model FOVs.
For a held-out physical slide it is computed exactly as `(all-slide sufficient statistic −
held-out-slide statistic) / 64,800`; no patch or scanner from the held-out slide contributes.

## Extraction and audit gates

- Each condition preserves scanner, location, replicate, canonical-center, native-grid hash,
  global-reference hash, encoder and checkpoint identity.
- Per-patch clipping fraction, clipping MAE, requested HF RMS and post-clamp operational HF
  RMS are stored with the embedding.
- Expected population: 436 model--slide shards and 2,354,400 control embeddings.
- Feature outcome aggregation cannot start until every shard passes hash, schema, finiteness,
  positive-norm and identity checks.

## Frozen E4 inference

- Primary scanner dispersion: L2-normalized six-scanner centroid RMS, location→slide
  aggregation and 109-slide equal weighting.
- Content endpoint: same-location raw-AT2 cosine minus the within-slide unmatched-AT2 q95;
  pooled Δmargin lower 95% slide-bootstrap CI must be at least −0.02.
- Collapse ratios: covariance trace, entropy effective rank and median pairwise distance;
  point estimate ≥0.90 and lower CI ≥0.85 for every non-AT2 source scanner.
- Uncertainty: the same 5,000 physical-slide bootstrap resamples with seed `20260803`.
- A condition is safe for one PFM only after content and every-scanner collapse gates pass.
  `Common safe across core PFMs` requires four-of-four PFM passage.

