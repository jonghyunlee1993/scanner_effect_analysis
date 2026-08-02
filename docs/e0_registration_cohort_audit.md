# E0 current-route registration audit — 109-slide cohort

**Status:** original 100-location audit complete; candidate-pool replacement pending

## Scope

The frozen integer-refinement contract from the 15-slide sentinel audit was applied to all
109 physical slides and five non-reference scanners. This analysis evaluates the original
109 × 100 Exp05 locations. It does not yet search the larger six-scanner common coordinate
pool for replacements.

Each slide–scanner cell contains 100 locations. A location passes when the corrected crop
has residual phase shift ≤8 px, padding ≤1%, and no search-boundary event. A cell passes
when at least 90 locations pass.

## Geometry result

| Scanner | Passing slide cells | Cell pass rate | Median location pass | Minimum location pass |
|---|---:|---:|---:|---:|
| GT450 | 102/109 | 93.6% | 99% | 44% |
| VERSA | 84/109 | 77.1% | 94% | 62% |
| AKOYA | 64/109 | 58.7% | 94% | 13% |
| S60 | 101/109 | 92.7% | 99% | 59% |
| S360 | 107/109 | 98.2% | 100% | 84% |

Across all 545 slide–scanner cells, 458 (84.0%) passed the strict 90/100 gate. AKOYA is the
dominant failure source, followed by VERSA. GT450, S60, and S360 are usually well aligned
after integer refinement but retain a small number of slide-specific failures.

## Exact six-scanner tuple retention

All five non-reference scanners must pass at the same location for a PFM paired tuple.

| Quantity | Result |
|---|---:|
| Median valid locations among the original 100 | 84 |
| Minimum | 13 |
| Slides with ≥80 valid locations | 61/109 |
| Slides with ≥90 valid locations | 37/109 |
| Slides with all 100 valid | 1/109 |

The worst slides were `12.5_29` (13), `8-12_3` (33), `2-8_8` (35), `12.5_11`
(38), and `2-8_11` (41). This does not mean those slides cannot supply 100 valid centers:
their pre-feature common coordinate pools contain 1,934–7,604 locations. Across the cohort,
the pool ranges from 1,040 to 31,892 locations (median 10,241).

## ERT sensitivity to corrected integer crops

The table reports the high-band change from the legacy zero-centred ±16 px result to the
current integer-refined result, using each route's valid locations.

| Scanner | Median absolute delta | q95 absolute delta | Maximum |
|---|---:|---:|---:|
| AKOYA | 0.00384 log2 | 0.08075 | 0.15513 |
| GT450 | 0.00096 log2 | 0.01746 | 0.04363 |
| VERSA | 0.00226 log2 | 0.01851 | 0.09441 |
| S60 | 0.00026 log2 | 0.01188 | 0.03076 |
| S360 | 0.00002 log2 | 0.00411 | 0.02887 |

Population ERT is substantially less sensitive to the old local-search failure than exact
paired feature geometry is. AKOYA has a wider tail and some cells use as few as 13 valid
locations, so E1 values remain provisional until replacement and E0c aliasing audits are
complete.

## Decision

1. The old zero-centred ±16 offsets are rejected for PFM extraction.
2. Current registered WSI plus integer refinement remains the primary candidate route.
3. Existing selected locations are retained when they pass all-scanner geometry and the
   model-FOV availability gate.
4. Failed locations must be deterministically replaced from the pre-feature common pool.
5. A scanner–slide cell becomes a targeted native-rigid rerun candidate only if the pool
   cannot supply 100 valid 512 px-aware centers under the same frozen QC contract.
6. E1–E3 can remain provisional because old→corrected ERT sensitivity is small, but they
   cannot be locked until replacement and aliasing audits finish.

## Reproducibility

- Smoke job: `15868589` (completed)
- Population array: `15868798` (108/108 completed; smoke supplies slide 0)
- Aggregate job: `15870044` (completed)
- Manifest: `outputs/e0_registration_cohort_109/cohort_manifest.csv`
- Aggregate summary: `outputs/e0_registration_cohort_109/summary.json`
- Figure: `outputs/e0_registration_cohort_109/figure_e0_registration_cohort.png`

Generated outputs remain under the ignored `outputs/` tree. Code, launchers, analysis
contracts, and this document are tracked in Git.
