# E5-RF1 Reinhard residual-frequency execution contract

**Status:** PRE-OUTCOME FROZEN for the E5-RF1 post-core extension  
**Frozen:** 2026-08-03, after PanNormal E0--E7 result lock and before E5-RF1 PFM endpoint
access  
**Parent evidence:** [`e5_comparator_population_results.md`](e5_comparator_population_results.md)  
**Scientific rationale:** [`reinhard_frequency_extension.md`](reinhard_frequency_extension.md)

E5-RF1 is a method-development extension. It does not reopen the frozen E4--E7 decision
record, change the five-method E5 primary ranking, or alter any locked result artifact.

## Population and outer cross-fitting

- Use the same 109 physical slides, six scanners, 100 locations, native-WSI explicit-AA grid
  and four frozen PFMs as E5.
- AT2 remains scanner index zero and is never rendered or re-encoded; its already audited raw
  embedding is copied into the extension condition.
- Use five balanced, outcome-blind physical-slide folds. Sort slides by
  `SHA256("e5_rf1_fold_v1:" + slide_id)` and assign consecutive sorted slides round-robin to
  folds 0--4. This produces folds of 22, 22, 22, 22 and 21 slides. Tissue labels, images and
  all PFM endpoints are forbidden from fold construction.
- For a held-out fold, all statistics are fit only from the other four folds. Every scanner
  and all 100 locations of each held-out slide are absent from fitting. Cross-fitted outputs
  for all 109 slides are materialized before endpoint aggregation.
- Model-native centered FOVs remain 224, 256 and 512 px. All statistics and transforms are
  fit separately for each FOV and source scanner.

## Frozen transform

The single extension condition is `reinhard_residual_frequency`.

### Step 1: training-fold Reinhard

For each scanner, pool the same fixed 8-pixel lattice used by E5 across training slides. Map
its CIE L*a*b* mean and SD to the pooled training AT2 mean and SD, with SD floor `1e-3`.
Apply that fixed transform to every training and held-out pixel. The Stage-A base image is the
same `[0,1]`-clipped sRGB output used by the locked E5 Reinhard implementation; its range
diagnostics are retained.

### Step 2: post-Reinhard residual spectrum fit

On every transformed training source patch and raw training AT2 patch, compute the 72-bin
Hann-windowed radial power of patch-mean-removed mean OD at 0.5052 µm/px. For source scanner
`s`, fit

`g_s(f) = sqrt(P_AT2(f) / P_Reinhard(s)(f))`.

Geometrically normalize the gain to one over 0.03--0.10 cycles/µm, smooth log gain with the
fixed kernel `[1,2,3,2,1]/9`, set it to one below 0.10 cycles/µm and symmetrically cap it by
the selected panel-wide cap. No tissue label, held-out image, PFM feature or downstream
endpoint enters this fit.

### Step 3: shared-OD residual and gamut projection

Convert the held-out Reinhard image to OD and let `m` be its channel mean. With 25% reflected
padding, apply `g_s(f)` to the centered Fourier coefficients of `m`, reconstruct `m'`, and set
the proposed scalar residual `d = m' - m`. Add the same `d` to all OD channels; this preserves
the two within-pixel chromatic OD differences.

For each pixel with original Reinhard OD vector `o`, project only the scalar residual into

`[-min(o), log(256) - max(o)]`.

The resulting three OD channels lie in `[0, log(256)]` and map to RGB8 support under the
frozen E5 inverse `RGB=(256 exp(-OD)-1)/255`. A final `[0,1]` clamp is permitted only as a
floating-point safeguard. Record proposed out-of-range channel fraction, projection fraction,
RGB change induced by projection and the final numerical clamp magnitude for every patch.

## Input-only cap selection

Candidate symmetric gain caps are `1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20, 1.25,
1.50, 2.00, 3.00, 4.00`. Evaluate all candidates on every cross-fitted held-out slide before
any PFM endpoint is computed. Select the largest panel-wide cap for which every
FOV/source-scanner cell satisfies:

- finite output and spectrum diagnostics;
- mean materially out-of-range channel fraction no greater than `0.005`;
- patch q99 materially out-of-range channel fraction no greater than `0.05`;
- mean RGB MAE induced by the shared-OD projection no greater than `0.002`;
- no final numerical-clamp MAE greater than `1e-6`.

A material excursion is proposed RGB below `-1/255` or above `1+1/255`. Zero-tolerance range
and projection fractions remain diagnostics. Target-spectrum log-RMSE before and after the
residual correction is reported for every cell but is not used to relax the gamut gate. If no
candidate passes, execution stops before PFM outcome access and a dated pre-outcome amendment
is required.

## Population and endpoint gates

- Expected feature population: 436 model--slide shards and 261,600 embeddings for the one
  condition, counting the raw AT2 copy in each shard.
- Audit exact scanner/location identity, raw-AT2 equality, source/checkpoint/statistic/cap and
  contract hashes, finite features and positive norms.
- Reuse without alteration the frozen E5 scanner-radius, content-margin non-inferiority,
  every-source-scanner collapse metrics, physical-slide bootstrap seed `20260803`, 5,000
  bootstrap replicates and four-PFM common-safe rule.
- Report the extension beside raw, locked Reinhard and locked frequency calibration. Do not
  recompute or rewrite the locked E5 table.
- Regardless of direction, report paired image-domain spectrum change and gamut projection.
  If the feature population passes, run the existing grouped tissue-type secondary endpoints
  with the same tissue-balanced inference and minimum-three sensitivity. Tissue results do
  not revise the frozen primary E5 decision.

## Learned-generator boundary

E5-RF1 is the required non-generative analytic baseline. A deterministic residual network,
weak adversarial extension or diffusion model is outside this contract. Such a model may be
started only under a new document that fixes outer folds, image-only model selection,
frequency/structure losses, hallucination checks and external PLISM validation before its PFM
or tissue endpoints are opened.

