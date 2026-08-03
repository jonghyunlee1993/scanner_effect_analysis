# E0 native geometry cohort contract

**Status:** gate frozen before the 109-slide cohort run

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

No threshold is relaxed after observing a fallback result. Non-rigid output remains
excluded unless it passes a separate interpolation audit.

## Reproducibility

- Slide runner: `src/run_e0_native_geometry_cohort.py`
- Cohort merger/gate: `src/merge_e0_native_geometry_cohort.py`
- Targeted rigid-route audit: `src/run_e0_registration_sentinel.py`
- Native-pixel fallback builder: `src/build_e0_rigid_native_fallback.py`
- Passing-fallback promoter: `src/promote_e0_native_geometry_fallback.py`
- Native from-scratch rigid runner: `src/run_e0_valis_rigid_from_scratch.py`
- Array launcher: `scripts/e0_native_geometry_cohort.sbatch`
- Merge launcher: `scripts/e0_native_geometry_merge.sbatch`
- Fallback launchers: `scripts/e0_rigid_alignment_array.sbatch`,
  `scripts/e0_rigid_native_array.sbatch`, `scripts/e0_native_geometry_promote.sbatch`,
  `scripts/e0_valis_rigid_from_scratch.sbatch`
- Unit contract: `tests/test_e0_native_geometry_cohort.py`
- Shards and matrices: `outputs/e0_native_geometry_cohort/shards/`
- Merged manifest and failure lists: `outputs/e0_native_geometry_cohort/merged/`

Generated outputs remain ignored. Code, frozen gates, tests and launchers are tracked
in Git before the full array is submitted.
