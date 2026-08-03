# E4 control-population locked results

**Status:** locked after the 2026-08-03 pre-outcome decision freeze  
**Contract:** [`e4_control_population_contract.md`](e4_control_population_contract.md)  
**Result lock:** `outputs/e4_control_results_lock/summary.json`

## Population and inference audit

- 436/436 model--slide control shards passed.
- 2,354,400/2,354,400 control embeddings passed hash, identity, shape, finiteness and
  positive-norm checks; total HDF5 size was 8,302,263,847 bytes.
- Primary inference used 109 equally weighted physical slides and 5,000 bootstrap resamples
  with seed `20260803`.
- The result lock audited 13 source/figure artifacts, 91,560 slide-level result rows and 40
  endpoint rows without failure.

## Primary control result

The registered same-location leave-one-scanner-out HF mean mixed at 25% was the only
non-raw condition that was safe and significantly improved scanner invariance in all four
PFMs. It is an attainable paired oracle, not a deployable correction.

| PFM | Raw radius | Paired-oracle RR (95% CI) | Δ content margin (95% CI) | Minimum collapse lower CI | Decision |
|---|---:|---:|---:|---:|---|
| ResNet50 | 0.2119 | 0.1246 (0.1135–0.1378) | 0.0038 (0.0030–0.0050) | 0.9840 | safe + improved |
| UNI v1 | 0.4827 | 0.0948 (0.0890–0.1017) | 0.0241 (0.0215–0.0272) | 0.9791 | safe + improved |
| CONCH v1 | 0.2625 | 0.1398 (0.1262–0.1562) | 0.0015 (−0.0007–0.0040) | 0.9903 | safe + improved |
| Virchow2 | 0.3513 | 0.1072 (0.0995–0.1164) | 0.0119 (0.0099–0.0143) | 0.9950 | safe + improved |

## Boundary-defining failures and model specificity

- Complete HF attenuation reduced scanner radius by 26.8–51.1% across PFMs, but Δ content
  margin ranged from −0.2805 to −0.7425 and every PFM failed the fidelity decision.
- Replacing all source HF coefficients by the LOSO global training mean reduced radius by
  26.2–46.7%, but Δ content margin ranged from −0.2721 to −0.7331 and every PFM failed.
  Its minimum collapse-ratio lower CI was 0.166–0.377, far below the frozen 0.85 threshold.
- HF retention 0.75 significantly improved invariance in all four PFMs, yet failed content
  non-inferiority for UNI v1 (lower CI −0.0292) and narrowly for CONCH v1 (−0.0205 versus
  the frozen −0.0200 margin). It was therefore not common-safe.
- Safe-and-improved simple controls were model-specific: ResNet50 passed retention 0.75 and
  boosts 1.25/1.50; Virchow2 passed retention 0.75 and boost 1.25; UNI v1 and CONCH v1 had
  no safe-and-improved simple attenuation/boost control.

These controls empirically separate low scanner radius caused by genuine paired alignment
from low radius caused by content loss or collapse. E4 establishes that the frozen decision
rule is operational and that a safe region is attainable. It does not select a deployable
normalization method; that question remains E5.

## Primary artifacts

- `outputs/e4_control_frontier/endpoint_summary.csv`
- `outputs/e4_control_frontier/collapse_summary.csv`
- `outputs/e4_control_frontier/common_pfm_summary.csv`
- `outputs/e4_control_figure/figure4_control_bounded_frontier.{png,pdf}`
- `outputs/e4_control_results_lock/artifact_manifest.csv`

