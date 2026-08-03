# E0 native geometry cohort contract

**Status:** native geometry and native-AA gates passed; final E0--E3 result lock complete

**Manifest version:** `e0_native_geometry_v1`

## Purpose

The frozen `e0_integer_512_v1` manifest identifies 109 slides × 100 canonical
biological centers, but its RGB pixels live in historical registered TIFFs. Those
TIFFs are unsuitable as the primary pixel source because GT450/VERSA were downsampled
without explicit anti-aliasing and moving-scanner MPP tags were overwritten
incorrectly.

This stage maps every frozen center back to the scanner-native WSI. Historical
registered pixels are used only for same-scanner geometry recovery and reconstruction
QC. Final image and PFM inputs must be generated from native WSI pixels.

## Frozen recovery route

For each physical slide and moving scanner:

1. Recover a native→historical-target global similarity from 4,096-pixel thumbnails
   using 12,000-feature SIFT, deterministic FLANN matching and RANSAC.
2. Start each local match at the already frozen scanner offset in
   `e0_integer_512_v1`.
3. Match a 256 px low-pass patch within ±120 target pixels and store the residual
   integer displacement, NCC and RGB reconstruction error.
4. Recenter the maximum 512 px model FOV using `legacy offset + native residual`.
5. Map all four target-FOV corners through the inverse native→target matrix and verify
   native bounds with a four-pixel interpolation margin.
6. Store the complete affine matrix and explicit-AA pre-scale in every long-manifest
   row. The later DataLoader applies `Lanczos3 reduction → residual bicubic affine`
   directly to the native WSI.

AT2 uses an identity native→target matrix only if raw and historical target dimensions
match and all 100 deterministic 256 px checks have RGB MAE ≤0.5. Its final RGB also
comes from the native WSI.

The ±120 local range was fixed after the five-moving-scanner `12.5_11` sentinel. It is
larger than the Akoya residual q95 observed with the earlier ±64 pilot and remains
inside the ±128 geometric support implied by the already frozen 512 px FOV around a
256 px center.

## Frozen gates

Global transform gates, applied to every moving scanner–slide cell:

- SIFT RANSAC inliers ≥80;
- thumbnail inlier reprojection q95 ≤4 px;
- affine anisotropy ratio ≤1.02;
- recovered native-px/target-px scale within 5% of scanner MPP expectation.

Location gates:

- low-pass same-scanner NCC ≥0.75;
- residual optimum does not touch the ±120 search boundary;
- shifted 512 px FOV is inside the target canvas;
- all mapped FOV corners are inside the native WSI with interpolation margin.

A scanner–slide cell passes only when its global transform passes and all 100 frozen
locations pass. No outcome, PFM feature or tissue label is used for this decision.

## Cohort completeness gate

The merged output must contain exactly:

- 109 slide shards;
- 654 scanner–slide cells (`109 × 6`);
- 65,400 unique scanner-location rows (`109 × 100 × 6`);
- 10,900 complete six-scanner tuples;
- zero duplicate, missing or unexpected keys;
- zero failed cells and locations after any targeted fallback.

The first merge may fail this promotion gate. Its failure list defines the only
scanner–slide cells eligible for the following prespecified fallback hierarchy:

1. Audit the preserved VALIS rigid branch at the same frozen 100 centers with the
   unchanged global, local, boundary and padding gates.
2. If all 100 centers pass, recover a same-scanner native→rigid transform and use the
   reconstructed-native↔rigid patches to estimate the same ±120 local integer
   residual used by the primary route. Require same-scanner NCC ≥0.75 and a non-boundary
   optimum, then use the composed geometry to read pixels directly from the native WSI.
   The preserved rigid image remains geometry/QC only.
3. If the preserved rigid output is missing or either audit fails, rerun VALIS
   rigid/affine from the native WSI for that scanner–slide cell only and apply the same
   gates.
4. A location that fails only the strict 512 px native-FOV bound is replaced by the
   next eligible location in the already ranked common-coordinate candidate pool. The
   ranking never uses registration outcome, PFM features or tissue labels.

Reserve candidates continue the original `SELECTION_SEED=20260802` permutation after
the last candidate needed by `e0_integer_512_v1`; they do not restart or rerank the pool.
Failed location IDs are processed in ascending order and receive the first reserve rank
that passes the complete six-scanner route gate. A rejected reserve remains recorded and
is never reconsidered for another slot.
Candidate trials reuse only fingerprint-matched global transform matrices from the frozen
cohort shard; all location residual, NCC, boundary and FOV checks are recomputed.
The same read-only fingerprint rule applies to preserved/from-scratch rigid candidate
routes through `--transform-cache-root`; a cache miss or mismatch is an error, never a
trigger to overwrite the frozen transform.
`audit_e0_native_candidate_trial.py` then composes an explicit scanner-to-route map and
accepts a trial only when all 600 scanner-location keys, all six cell gates and all 100
six-scanner tuples pass with native WSI marked as the pixel source.
`select_e0_native_fallback_actions.py` freezes the next action per failed cell: strict
target/native bounds failures go directly to deterministic candidates, complete preserved
rigid cells are promoted, and only missing or non-bounds geometry failures request a
from-scratch pairwise native VALIS rigid run.
An absent audit shard is not treated as an absent preserved route: if the expected rigid WSI
exists, the cell is first labeled `audit_preserved_rigid`; only a genuinely absent preserved
WSI proceeds directly to from-scratch registration.
`select_e0_fromscratch_outcomes.py` then promotes complete from-scratch cells and sends
remaining location-specific geometry failures to deterministic candidates only when the
global native-to-rigid transform gate passed. A failed global transform remains explicitly
unresolved instead of recursively rerunning or relaxing a gate.
Finalization is plan-driven: `finalize_e0_native_geometry.py` replaces only explicitly
listed candidate slide manifests and scanner-cell geometry routes, verifies that every
route's canonical centers match the final candidate coordinates, and reruns the complete
65,400-row/10,900-tuple cohort gate.
`build_e0_candidate_execution_plan.py` derives scanner route kinds and the union of failed
location IDs directly from frozen action/outcome CSVs, reuses already accepted trials, and
materializes deterministic trial manifests plus primary and rigid-route execution lists.
The three `e0_candidate_*_array.sbatch` launchers consume those lists without reconstructing
route decisions in shell: primary routes reuse frozen global transforms, while each preserved
or from-scratch cell reruns cross-scanner and same-scanner local gates against the trial centers.
`audit_e0_candidate_execution_plan.py` composes those scanner roots, rewrites each trial's
accepted geometry artifact, and reports the exact remaining location IDs for deterministic
reserve advancement when trial 0 does not yet pass.
`advance_e0_candidate_execution_plan.py` retains every passing replacement, assigns each
failed or newly failing slot the next globally unused reserve rank, records the rejected
rank lineage, and materializes the next versioned trial without reranking any candidate.
Plan rebuilding selects the newest passing versioned trial only when its audited six-scanner
route kinds exactly match the current frozen route plan; a stale accepted route is not reused.
`build_e0_native_finalization_plan.py` then expands every accepted candidate slide to six
explicit scanner-cell overrides and adds only the promoted preserved/from-scratch cells for
noncandidate slides, so no original-coordinate primary shard leaks into a candidate manifest.
Candidate execution may be sharded with `--only-slides`; this limits materialization and
launch rows only, while retaining the full frozen scanner route plan for every audit.
The same execution-subset contract applies to candidate advancement, allowing a stalled
registration route to pause without blocking independent slides whose reserves are converging.
When a from-scratch similarity registration shows cohort-wide low cross-scanner NCC and
search-boundary saturation rather than isolated location failures, a separate feature-matched
VALIS `AffineTransform` diagnostic is permitted with the intensity optimizer and non-rigid
registration disabled. It must pass the unchanged 100-location gate before promotion.
If neither primary nor from-scratch passes all 100 locations, a primary route with a valid
global transform is retained when it has no more failed locations than from-scratch VALIS.
Only that route's failed slots enter the deterministic candidate loop, minimizing
registration-conditioned replacement without changing any gate.

After the six-scanner gate passes, `render_e0_native_aa_shard.py` renders one 512 px
target-grid RGB patch per scanner and location directly from the native WSI. It applies
libvips Lanczos3 reduction at the smallest native-to-target affine singular value followed
by the residual bicubic affine. The stored 512 px grid is center-cropped to the frozen
model FOVs (ResNet50/UNI v1 256 px, CONCH v1 512 px, Virchow2 224 px); historical
registered RGB is never opened by this renderer.

## Observed cohort outcome

The primary native recovery produced 65,400 expected rows, of which 10,644/10,900
six-scanner tuples and 593/654 scanner–slide cells passed. The frozen fallback selector
therefore reviewed 61 failed cells. Preserved-route audit, targeted from-scratch VALIS and
the route-min rule reduced the final candidate burden to 22 slides and 44 location slots.
The first candidate trial passed 21/22 slides; one additional deterministic reserve for
`12.5_30` location 78 completed the last route without changing any threshold.

The plan-driven finalization then passed all required counts: 109 slides, 654/654 cells,
65,400/65,400 scanner-location rows and 10,900/10,900 six-scanner tuples, with zero failed,
missing, duplicate or unexpected keys. It used 22 candidate manifests and 161 explicit
scanner-cell geometry overrides. Every final row is marked `native_wsi_only`.

Native-AA rendering subsequently produced 109/109 HDF5 shards containing all
65,400 scanner-location patches. The full audit recomputed every shard SHA-256, matched
scanner/location/center/affine/native-path identity back to the final geometry, and read
every 512 px RGB patch to exclude fully black or fully white renders. All 109 shards and
65,400 patches passed; the frozen grid occupies 47,493,295,240 bytes and contains no
missing, unexpected or temporary files.

The post-finalization alias audit selected q05/q50/q95 transforms from all 109 GT450
and 109 VERSA final cells. The explicit-AA chain passed the frozen 5% sinusoid and
broadband-noise ratios for all six profiles; its worst ratios were 0.0154 and 0.0190.
The corresponding original single-pass bicubic q50 ratios were 1.545/1.115 for GT450
and 1.319/0.960 for VERSA. This closes E0c for the final geometry rather than relying
on the earlier historical-VALIS transform distribution.

No threshold is relaxed after observing a fallback result. Non-rigid output remains
excluded unless it passes a separate interpolation audit.

## Same-chain operational background floor

E0d reuses all 65,400 outcome-blind native glass coordinates accepted by Exp07. Each
coordinate is rendered from its native WSI with the final scanner-slide affine and the
same `Lanczos3 reduction -> residual bicubic affine` chain used for E1 tissue patches.
No coordinate is removed after post-render QC. The spectral estimator is also identical
to E1: natural-log mean optical density, per-patch mean removal, a 2D Hann window and
72-bin radial power at 0.5052 micrometres/pixel.

All 109 slide shards passed the frozen identity gate. The aggregate contains 47,088
background spectra rows, 235,440 five-by-20 replicate rows, and 65,400 coordinate/QC
rows. Post-render glass QC retained 98.34% of patches (slide range 90.33--100%), while
the maximum black-pixel fraction in any patch was 0.001404. QC failures remain included
to avoid outcome-dependent reselection.

Background subtraction was positive in every one of the 47,088 tissue spectrum bins.
In the 0.60--0.90 cycles/micrometre band, median background/tissue power ranged from
0.00048 for AT2 to 0.00569 for AKOYA. The median change in anchor-normalized ERT was
-0.0014 to -0.0039 log2 across non-reference scanners. At SNR >= 10, the high-band
eligible counts were 109/109 for AT2, GT450 and S60; 108/109 for VERSA and S360; and
106/109 for AKOYA. This is an operational glass/background floor after the analysis
chain, not detector NPS, DQE or absolute MTF.

## Reproducibility

- Slide runner: `src/run_e0_native_geometry_cohort.py`
- Cohort merger/gate: `src/merge_e0_native_geometry_cohort.py`
- Final cohort composer: `src/finalize_e0_native_geometry.py`
- Finalization plan builder: `src/build_e0_native_finalization_plan.py`
- Targeted rigid-route audit: `src/run_e0_registration_sentinel.py`
- Native-pixel fallback builder: `src/build_e0_rigid_native_fallback.py`
- Passing-fallback promoter: `src/promote_e0_native_geometry_fallback.py`
- Native from-scratch rigid runner: `src/run_e0_valis_rigid_from_scratch.py`
- Array launcher: `scripts/e0_native_geometry_cohort.sbatch`
- Merge launcher: `scripts/e0_native_geometry_merge.sbatch`
- Native-AA renderer/audit: `src/render_e0_native_aa_shard.py`,
  `src/audit_e0_native_aa_grid.py`
- Native-AA launchers: `scripts/e0_native_aa_grid_array.sbatch`,
  `scripts/e0_native_aa_audit.sbatch`
- Same-chain background renderer/analysis:
  `src/render_e0d_same_chain_background_slide.py`,
  `src/analyze_e0d_same_chain_noise_floor.py`
- Same-chain background launchers: `scripts/e0d_same_chain_background_array.sbatch`,
  `scripts/e0d_same_chain_background_aggregate.sbatch`
- Fallback launchers: `scripts/e0_rigid_alignment_array.sbatch`,
  `scripts/e0_rigid_native_array.sbatch`, `scripts/e0_native_geometry_promote.sbatch`,
  `scripts/e0_valis_rigid_from_scratch.sbatch`
- Unit contract: `tests/test_e0_native_geometry_cohort.py`
- Shards and matrices: `outputs/e0_native_geometry_cohort/shards/`
- Merged manifest and failure lists: `outputs/e0_native_geometry_cohort/merged/`

Generated outputs remain ignored. Code, frozen gates, tests and launchers are tracked
in Git before the full array is submitted.
