# E5 comparator-population execution contract

**Status:** PRE-OUTCOME FROZEN; numerical stability constants are selected only by the
input-domain audit described below  
**Frozen:** 2026-08-03, after approval of Decisions 1--7 and before E5 PFM endpoint access  
**Parent decision:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md)

This contract turns the approved E5 method names into executable transforms. It does not
change the parent decision record. The E5 population cannot be scored until the input-only
stability manifest, the image and feature populations and their identity audits all pass.

## Population and cross-fitting

- The population is the same 109 physical slides, six scanners and 100 registered locations
  used in E0--E4. AT2 is scanner index zero and remains the unmodified target in every
  comparator condition.
- ResNet50, UNI v1, CONCH v1 and Virchow2 use their frozen model-specific native FOV and the
  already audited raw embedding as the feature-space input and as the AT2 output.
- Primary predictions use 109-fold leave-one-physical-slide-out (LOSO). For held-out slide
  `l`, every sufficient statistic is exactly `all slides - slide l`; all six scans and all
  100 locations of `l` are absent from fitting.
- Cross-fitted held-out predictions are materialized before endpoint aggregation. The 5,000
  replicate physical-slide bootstrap never refits a transform.

## Primary image-space comparators

All RGB inputs come from the audited native-WSI explicit-AA grid at 0.5052 µm/px. Model FOVs
are centered crops of 224, 256 or 512 px. Training color statistics use a fixed 8-pixel
lattice (offset four pixels on each axis) within every train patch; the held-out transform is
then applied to every pixel. This deterministic sampling rule is part of the method, not an
outcome-tuned choice.

| Condition | Frozen transform |
|---|---|
| `reinhard_lab` | Per-scanner pooled CIE L*a*b* train mean and SD are mapped affinely to the pooled AT2 train mean and SD. Conversion uses sRGB/D65, SD is floored at `1e-3`, and the inverse sRGB is clipped once to `[0,1]`. No held-out image statistic is used. |
| `paired_od_affine` | A 3-channel optical-density affine with bias is fit from registered source/AT2 train pixels by sufficient statistics. OD is `-log((RGB8+1)/256)`. A relative ridge of `1e-6 × mean(diag(X'X)[0:3])` is applied to the three OD coefficients; bias is unpenalized. The inverse OD image is clipped once to the RGB8 range. |
| `frequency_calibration` | Train patches estimate the 72-bin Hann-windowed radial power of patch-mean-removed mean OD. The phase-preserving correction gain is `sqrt(P_AT2/P_source)`, geometrically anchored to one over 0.03--0.10 cycles/µm, smoothed in log space by the fixed kernel `[1,2,3,2,1]/9`, set to unity below 0.10 cycles/µm, and symmetrically capped. The same scalar radial gain is applied to each centered OD channel with 25% reflected padding; the inverse is cropped and RGB-clipped once. |

Every image method records per-patch pre-clip channel fraction and clipping MAE. The AT2
image and embedding are never re-encoded or altered: the audited raw AT2 embedding is copied
into each condition.

Macenko remains Supplement-only as approved. It is not allowed to replace or delay the three
primary image comparators. Its LOSO population, if materialized, is reported outside the
primary common-safe claim.

## Primary feature-space comparators

Both methods operate on the frozen, unnormalized raw PFM embeddings. They are fit separately
for each PFM and each of the five source scanners. For `X` (source) and `Y` (paired AT2), train
means are subtracted and the target train mean is restored after transformation.

| Condition | Frozen transform |
|---|---|
| `coral` | Source covariance is whitened and AT2 covariance is recolored. Both covariance matrices use the same panel-wide isotropic shrinkage `C_a=(1-a)C+a tr(C)/D I`; symmetric eigendecomposition is used and eigenvalues are floored at `1e-7 × tr(C_a)/D` only as a floating-point safeguard. |
| `orthogonal_procrustes` | From paired train embeddings, `R=UV'` for `U,S,V'=SVD(X_c'Y_c)` and held-out output is `(x-mean_X)R+mean_Y`. There is no scale term, ridge term or reflection override. |

The full train-fold matrices are derived from all-slide sums, Gram matrices and paired
cross-products after subtracting the held-out slide contributions. Neither method uses tissue
labels, held-out scanner centroids or any E4/E5 endpoint.

## Input-only numerical stability gate

The two constants not fixed in the parent decision are selected before any E5 PFM endpoint is
computed.

- CORAL shrinkage candidates are `0.0001, 0.001, 0.01, 0.05, 0.10, 0.20`. Select the smallest
  panel-wide value for which every model/scanner covariance in the deterministic calibration
  fold has condition number at most `10,000`, every transformed value is finite and held-out
  transformed L2-norm q01/q99 stay within 0.1--10 times the corresponding raw quantiles.
- Frequency gain-cap candidates are `1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20,
  1.25, 1.50, 2.00, 3.00, 4.00`.
  Select the largest
  panel-wide cap for which every FOV/source-scanner cell in the deterministic calibration
  slide is finite, mean materially clipped-channel fraction is at most `0.005`, patch q99
  materially clipped-channel fraction is at most `0.05`, and mean clipping MAE on `[0,1]`
  RGB is at most `0.002`. A material excursion is below `-1/255` or above `1+1/255`; the
  zero-tolerance out-of-range fraction is still recorded as a diagnostic.
- The deterministic calibration slide is the lexicographically first audited slide. It is
  excluded when its train-fold parameters are calculated. This audit may inspect raw RGB,
  raw feature scale, covariance conditioning and output numerical range only. It must not
  calculate scanner radius, content margin, collapse, retrieval, PFM ranking or tissue result.
- The selected constants, candidate table, source hashes and pass/fail are written to
  `outputs/e5_input_stability/summary.json`. All downstream E5 programs require its hash and
  `stability_gate_pass=true`.

If no candidate passes, execution stops before outcome access and this document receives a
dated pre-outcome amendment. No threshold is relaxed after an E5 endpoint is opened.

### Pre-outcome amendment 1 — 2026-08-03

The first input-only run selected CORAL shrinkage `0.05` and passed Procrustes stability, but
no frequency cap passed because the zero-tolerance fraction counted sub-quantization floating
excursions at RGB boundaries. Across candidates the mean clipping MAE was only
`0.000003--0.000929` on the `[0,1]` scale, while the zero-tolerance fraction reached `0.0394`.
No scanner-radius, content, collapse, retrieval, tissue or model-ranking endpoint had been
computed or opened. The frequency gate above was therefore amended to apply its fraction
thresholds to excursions larger than one RGB8 code while retaining zero-tolerance fraction
and clipping MAE as diagnostics. Candidate caps and every numerical threshold are unchanged.

### Pre-outcome amendment 2 — 2026-08-03

With the material-excursion definition, AKOYA remained above the unchanged mean-fraction
limit at the smallest original cap 1.25 (`0.0109--0.0115` across FOVs); the other
scanner/FOV cells passed. Mean clipping MAE remained at most `0.000187`. No E5 endpoint had
been computed or opened. Conservative cap candidates `1.05, 1.10, 1.15, 1.20` were added
below the original grid so that the largest cap satisfying the already frozen thresholds can
be selected. The transform, calibration slide and all stability thresholds are unchanged.

### Pre-outcome amendment 3 — 2026-08-03

The expanded audit found that S360 narrowly exceeded the unchanged mean material-clipping
limit at caps 1.05 (`0.00506--0.00532`) and 1.10 (`0.00519--0.00523` for 224/256 px), while
AKOYA passed at 1.10. The worst corresponding clipping MAE was `0.000048`. No E5 endpoint had
been computed or opened. Candidates `1.01, 1.02, 1.03, 1.04` were added below the grid; the
largest value satisfying the same every-cell thresholds remains the selection rule.

## Population and result gates

- Expected image population: 436 model--slide shards and 784,800 embeddings across the three
  primary image conditions.
- Expected feature population: 436 model--slide shards and 523,200 embeddings across the two
  primary feature conditions.
- Expected total E5 comparator embeddings: 1,308,000. AT2 copies are counted in each condition
  because each materialized method tensor must be independently complete.
- Audits require exact scanner/location identity, raw-AT2 equality, source and checkpoint
  hashes, fit-statistic and stability-manifest hashes, finite values and positive norms.
- The frozen E4 primary radius, content-margin non-inferiority, every-source-scanner collapse
  gate, four-of-four PFM claim rule, bootstrap seed and 5,000 replicates are reused without
  alteration.
