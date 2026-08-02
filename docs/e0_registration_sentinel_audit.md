# E0 registration sentinel audit

**Status:** completed sentinel audit; 109-slide candidate-pool audit pending

**Frozen manifest:** 15 slides, 15 tissue types, five each from AKOYA old-search
`worst`, `median`, and `best` strata

## Question

The Exp05 spectrum analysis used a zero-centred ±16 px local search. AKOYA reached that
boundary in 54.46% of all patches, so that search could neither estimate the true residual
field nor support exact paired PFM crops. This audit asks two separate questions:

1. Can the existing registered WSI be retained after slide-prior-guided integer refinement?
2. Are the existing native-WSI VALIS rigid outputs a better replacement, without materially
   changing the effective spectrum?

## Frozen comparison contract

- Locations: the existing 100 Exp05 locations per sentinel slide.
- Scanners: GT450, VERSA, AKOYA, S60, and S360 against AT2.
- Current legacy condition: zero-centred ±16 px (`current_old_local16`).
- Corrected condition: five-fold cross-fitted constant/affine slide prior, global NCC peak
  when it lies within 48 px Manhattan distance of the prior, and ±16 px local fallback.
- VALIS condition: existing `rigid_akoya` and `rigid_all` outputs, evaluated on the same AT2
  coordinate and with the same residual integer refinement.
- No sub-pixel or non-rigid interpolation was added by the audit.
- Tile geometry pass: residual phase shift ≤8 px, padding ≤1%, and no search boundary.
- Slide–scanner cell pass: at least 90% of the 100 locations pass.
- Route pass: at least 90% of 75 slide–scanner cells pass and every scanner passes at least
  80% of its 15 slide cells.
- Route ERT equivalence: median absolute delta ≤0.10 log2 and q95 ≤0.25 log2.

The first smoke run exposed a cohort-wide padding-QC error: the previous local-gray
`std < 1` definition labeled weakly stained GT450 tissue as padding despite NCC ≈0.99 and
residual ≈0.1 px. Registered padding is locally constant, so the flatness cutoff was frozen
at `std < 0.25`; two synthetic regression tests cover constant missing regions and
low-amplitude tissue texture.

## Registration result

| Route | Passing cells | Overall | Lowest scanner | Frozen route gate |
|---|---:|---:|---:|---|
| Current WSI + integer refinement | 64 / 75 | 85.3% | 73.3% | Fail |
| Existing VALIS rigid + integer refinement | 57 / 75 | 76.0% | 53.3% | Fail |

Scanner-specific cell pass rates:

| Scanner | Current | VALIS rigid |
|---|---:|---:|
| AKOYA | 11/15 (73.3%) | 12/15 (80.0%) |
| GT450 | 15/15 (100.0%) | 8/15 (53.3%) |
| S360 | 14/15 (93.3%) | 13/15 (86.7%) |
| S60 | 13/15 (86.7%) | 14/15 (93.3%) |
| VERSA | 11/15 (73.3%) | 10/15 (66.7%) |

The current route is not uniformly valid, but it is materially better than the old ±16 px
diagnostic suggested. Failures concentrate in three low-NCC AKOYA slides, four VERSA slides,
and a small number of content/boundary failures elsewhere. The existing VALIS outputs are
not a safe cohort-wide replacement: several GT450, VERSA, AKOYA, S60, or S360 slide outputs
still peak at the ±96 px audit boundary and show residual q95 values above 10 px.

Across the five scanners, the exact same-location tuple count on the current route ranged
from 48 to 99 among the original 100 locations (median 89). This does not yet show that a
slide lacks 100 valid locations because the much larger common coordinate pool has not been
used for deterministic replacement.

## Spectrum sensitivity

Within the current route, correcting the legacy ±16 result had little effect on ERT:

| Band | Median absolute delta | q95 absolute delta | Maximum absolute delta |
|---|---:|---:|---:|
| 0.10–0.30 cycles/µm | 0.00081 log2 | 0.01550 | 0.08605 |
| 0.30–0.60 cycles/µm | 0.00112 log2 | 0.01457 | 0.08669 |
| 0.60–0.90 cycles/µm | 0.00165 log2 | 0.01667 | 0.09441 |

By contrast, current and VALIS corrected routes differed systematically in the high band:

| Band | Median absolute route delta | q95 absolute route delta |
|---|---:|---:|
| 0.10–0.30 cycles/µm | 0.00298 log2 | 0.01020 |
| 0.30–0.60 cycles/µm | 0.05722 log2 | 0.07536 |
| 0.60–0.90 cycles/µm | 0.26210 log2 | 0.30278 |

The high-band direction was consistently `current − VALIS > 0` in magnitude. This is not
evidence that either route recovers the scanner's optical transfer more faithfully. It
shows that the registration/resampling route itself changes the effective spectrum and
must be resolved by the E0 alias/interpolation audit.

Some route comparisons used as few as 13 common valid locations in catastrophic VALIS
cells. The direction is consistent across scanners, but final effect sizes require a
repaired route or a prespecified minimum-common-location analysis.

## Decision

The frozen automatic gate classifies both routes as failed. A blind cohort-wide promotion
of the existing VALIS files is rejected because their geometry is worse overall and their
resampling path materially changes high-frequency power. A blind rerun with the same VALIS
export contract is also not justified: the existing VALIS branches were already generated
from native WSI and show scanner/slide-specific catastrophic outputs.

The next E0 step is therefore:

1. Apply the current integer-refinement audit to all 109 slides.
2. Build the 512 px-aware common candidate manifest, retain valid original locations, and
   deterministically replace failures from the pre-feature common coordinate pool.
3. Identify slides that still cannot supply 100 six-scanner valid centers.
4. Audit the VALIS export/canvas and interpolation chain on those failing cells only.
5. Rerun rigid registration from native WSI only for unresolved scanner–slide cells, then
   repeat the same frozen geometry and ERT checks.

For E1–E3, the small current old→corrected ERT sensitivity supports retaining the current
numbers as provisional while E0c aliasing remains open. For PFM extraction, the original
offsets must not be used as-is; a corrected, identity-preserving feature manifest is a hard
prerequisite.

## Reproducibility

- Final array job: `15867130` (15/15 completed)
- Final aggregate job: `15867686` (completed)
- Manifest: `outputs/e0_registration_sentinel/sentinel_manifest.csv`
- Aggregate summary: `outputs/e0_registration_sentinel/summary.json`
- Figure: `outputs/e0_registration_sentinel/figure_e0_registration_route_audit.png`
- Unit test: `PYTHONPATH=src python -m pytest -q tests/test_alignment_padding.py`

Generated outputs remain outside Git under `outputs/`; code, launchers, thresholds, and this
provenance document are tracked.
