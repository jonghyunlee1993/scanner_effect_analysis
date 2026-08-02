# E0 primary-route native anti-aliased patch pilot

**Status:** pilot complete; cohort expansion approved with per-location residual QC

## Question

The historical primary registered TIFFs do not retain their exact VALIS transform
objects and their GT450/VERSA pixels were downsampled without an explicit anti-alias
stage. This pilot asks whether the frozen canonical centers can nevertheless be mapped
back to native WSIs without using historical registered pixels as the primary RGB
source.

This is a geometry-recovery pilot, not a cohort validation and not a replacement for
the frozen synthetic alias gate.

## Frozen pilot route

- Slide: `12.5_11`; scanners: GT450 and VERSA.
- Recover a same-scanner native-WSI → historical-primary-output similarity transform
  from 4,096-pixel thumbnails using SIFT, FLANN ratio matching and strict RANSAC.
- At each frozen location, estimate an integer residual from low-pass native
  reconstruction versus the historical output within a ±64 target-pixel search.
- Read final RGB only from the native WSI.
- Apply Lanczos3 reduction at the minimum affine singular value, followed by the
  remaining non-decimating bicubic affine.
- Use historical registered RGB only for transform recovery, local residual estimation
  and reconstruction QC.
- Audit 40 locations per scanner at 256 px. The cohort route must separately pass the
  512 px bounds and six-scanner completeness gate.

## Geometry result

| Metric | GT450 | VERSA |
|---|---:|---:|
| SIFT ratio matches | 3,199 | 8,210 |
| RANSAC inliers | 2,086 | 8,192 |
| Thumbnail reprojection median, px | 0.937 | 0.185 |
| Thumbnail reprojection q95, px | 2.461 | 0.500 |
| Native px / target px | 1.9231 | 1.8417 |
| Local residual magnitude median, target px | 23.04 | 7.07 |
| Local residual magnitude q95, target px | 51.09 | 10.44 |
| Residual NCC median | 0.9960 | 0.9962 |
| Residual NCC minimum | 0.9885 | 0.9890 |
| Search-boundary failures | 0/40 | 0/40 |
| Historical reconstruction RGB MAE median | 5.27 | 6.51 |

The global same-scanner recovery is well constrained on this slide, and the local
residual produces high patch agreement. However, GT450 retains a large
location-dependent residual after the global similarity transform. A single global
similarity is therefore not accepted as the final primary mapping. Each location must
retain its residual and QC fields in the cohort manifest.

## Real-tissue filtering sensitivity

Amplitude retention compares native explicit-AA patches with a native reconstruction of
the historical one-pass bicubic pixels at the same recovered geometry.

| Band, cycles/µm | GT450 AA / historical | VERSA AA / historical |
|---|---:|---:|
| 0.10–0.30 | 0.9980 | 0.9980 |
| 0.30–0.60 | 0.9716 | 0.9737 |
| 0.60–0.90 | 0.7732 | 0.7759 |

Anchor-normalized high-band retention is `0.7740` for GT450 and `0.7769` for VERSA.
The explicit-AA route therefore preserves low–mid amplitude while removing about 22%
of historical high-band amplitude on this real-tissue pilot. This is consistent in
direction with the synthetic passband cost. It does not by itself measure how much of
the removed power was alias rather than true in-band signal.

## Decision and cohort gate

1. Expand same-scanner native geometry recovery to the 109 × 100 frozen location
   manifest; do not regenerate locations with TRIDENT.
2. Store global transform provenance plus per-location residual, NCC, boundary,
   reconstruction error, native bounds and target-grid scale.
3. Extract maximum 512 px native-AA RGB at each canonical center, then derive each
   model's native FOV from that same center.
4. Promote a scanner–slide cell only if its global transform is sufficiently supported,
   all retained locations pass local/bounds QC and six-scanner completeness remains 100.
5. Rerun VALIS rigid/affine from native WSIs only for cells that fail those gates. Do not
   adopt non-rigid pixels as primary data without a separate interpolation audit.
6. TRIDENT is reserved for frozen encoder construction, official preprocessing and
   checkpoint loading. The study manifest/DataLoader owns crop identity and alignment.

## Reproducibility

- Implementation: `src/run_e0_primary_transform_recovery_pilot.py`
- SLURM launcher: `scripts/e0_primary_transform_recovery_pilot.sbatch`
- Completed clean rerun: job `15884653`, 49 seconds, 401 MiB maximum RSS
- Outputs: `outputs/e0_primary_transform_recovery_pilot/12.5_11/`
- Required numerical record: `summary.json`, `scanner_summary.csv`,
  `patch_recovery_metrics.csv`, `spectral_sensitivity.csv`
- Figure: `figure_native_aa_patch_pilot.png`; optional PDF was skipped because the
  historical VALIS environment does not include `fontTools`

Generated outputs remain ignored; code, tests, launcher and this decision record are
tracked in Git.
