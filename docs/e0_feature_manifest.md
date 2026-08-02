# E0 six-scanner feature manifest

**Status:** complete and verified for 109 slides × 100 locations

**Manifest version:** `e0_integer_512_v1`

## Purpose

The original Exp05 locations were selected before the registration audit and before the
CONCH v1 512 px input contract was fixed. This step creates the outcome-blind location
manifest that all four core PFMs will use.

Selection does not inspect PFM features or correction outcomes. It uses only the existing
six-scanner common coordinate pool, current-route geometry, image availability, and the
maximum model FOV.

## Frozen selection contract

1. Start from each slide's original 100 Exp05 locations.
2. Preserve the original `location_id` and five `patch_index % 5` replicate groups.
3. Retain an original location only when every non-reference scanner passes:
   - slide-prior-guided, global-first integer alignment;
   - residual phase shift ≤8 px;
   - 256 px padding ≤1%;
   - no search-boundary event;
   - shifted 512 px FOV in bounds and padding ≤1%.
4. Replace failed slots from the pre-feature intersection of all six curated coordinate
   pools, using deterministic seed `20260802`.
5. Apply the same geometry and 512 px FOV checks to every replacement.
6. Stop after exactly 100 valid centers or fail after 3,000 evaluated candidates.

The final coordinate is a canonical center `(x + 128, y + 128)`. Each scanner row also
stores its integer `(dy, dx)`, so model-native crops are centered on the same tissue:

- ResNet50: 256 px
- UNI v1: 256 px
- CONCH v1: 512 px
- Virchow2: 224 px

## Result

| Quantity | Result |
|---|---:|
| Slides complete | 109/109 |
| Final locations | 10,900 |
| Tissue types | 37 |
| Original locations retained | 8,203 (75.3%) |
| Locations replaced | 2,697 (24.7%) |
| Median replacements per slide | 20 |
| Maximum replacements on one slide | 89 |
| Median replacement candidates evaluated | 23 |
| Maximum candidates evaluated | 855 |
| Rows passing six-scanner geometry | 10,900/10,900 |
| Rows passing shifted 512 px FOV | 10,900/10,900 |

The maximum search cost occurred on `12.5_29`: 11 original locations were retained and
855 of 7,604 common-pool candidates were evaluated to find 89 replacements. It still
completed far below the 3,000-candidate cap.

Most rejected candidates failed scanner geometry rather than 512 px availability:

| Leading failure reason | Count |
|---|---:|
| AKOYA geometry | 2,804 |
| VERSA geometry | 1,181 |
| GT450 geometry | 634 |
| VERSA 512 px content/padding | 496 |
| S60 geometry | 180 |
| AKOYA 512 px content/padding | 128 |

## Decision

- Current registered WSI plus integer crop refinement is sufficient to construct the full
  PanNormal PFM population manifest.
- Neither cohort-wide nor targeted native rigid re-registration is triggered by feature
  location availability: every slide supplied 100 valid centers.
- Existing VALIS rigid outputs are not promoted into the primary PFM route.
- All PFM extraction must use the combined manifest and its per-scanner offsets rather than
  `selected_patches.csv` or the legacy ±16 offsets directly.
- E0c interpolation/aliasing remains open for interpreting ERT as scanner transfer, but it
  does not block deterministic crop/feature extraction from the selected current route.

## Reproducibility

- Easy-slide smoke: `15871123` (`8-12_20`, completed)
- Worst-slide smoke: `15871470` (`12.5_29`, completed)
- Population array: `15871972` (107/107 completed; two smoke slides supplied separately)
- Aggregate verification: `15873200` (completed)
- Worst-slide deterministic rerun: `15873340` (completed; manifest and audit SHA-256 exact)
- Combined manifest: `outputs/e0_feature_manifest_109/feature_manifest.csv`
- Slide summary: `outputs/e0_feature_manifest_109/slide_summary.csv`
- Scanner QC: `outputs/e0_feature_manifest_109/scanner_summary.csv`
- Figure: `outputs/e0_feature_manifest_109/figure_e0_feature_manifest.png`

Generated CSV/JSON/figures remain under ignored `outputs/`. Code, launchers, tests, and this
contract document are tracked in Git.
