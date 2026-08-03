# E5 Supplement-only Macenko results

**Status:** Supplement result locked on 2026-08-03  
**Placement:** Supplement only; excluded from every primary E5 ranking and four-PFM claim  
**Contract:** [`e5_macenko_supplement_contract.md`](e5_macenko_supplement_contract.md)  
**Result lock:** `outputs/e5_macenko_supplement_lock/summary.json`

## Population gate

- The exact 109-fold LOSO Macenko reference was fit from training slides only for each
  model-specific native FOV.
- All 436 model--slide shards and 261,600 embeddings passed identity, raw-AT2 equality,
  finite norm, source/checkpoint/contract hash and schema audits.
- The prespecified numerical fallback was used for 473/261,600 rendered patches (0.181%);
  no patch or slide was removed.
- The primary E5 lock hash is recorded in the Supplement result lock. Adding Macenko did not
  reopen the five-method benchmark or alter its ranking.

## Frozen endpoint results

| PFM | Radius reduction | ΔR CI lower | Δ content margin | ΔM CI lower | Every-scanner collapse | Decision |
|---|---:|---:|---:|---:|---|---|
| ResNet50 | +10.1% | +0.0161 | −0.0062 | −0.0088 | pass | safe + improved |
| UNI v1 | −1.1% | −0.0142 | −0.1034 | −0.1146 | pass | unsafe; no improvement |
| CONCH v1 | −14.7% | −0.0445 | −0.0386 | −0.0436 | pass | unsafe; no improvement |
| Virchow2 | −19.7% | −0.0758 | −0.1200 | −0.1289 | pass | unsafe; no improvement |

All collapse diagnostics passed, but only ResNet50 met both content non-inferiority and
positive invariance. Scanner-specific content changes were negative for every source scanner
in UNI v1, CONCH v1 and Virchow2. Therefore Macenko is neither common-safe nor
common-safe-and-improved across the four frozen PFMs.

## Interpretation and provenance

Macenko's model dependence supports its prespecified Supplement placement. The result does
not imply that stain normalization is intrinsically invalid; it shows that this exact
cross-fitted implementation and target, applied to this same-stain multi-scanner cohort, did
not transfer safely across the four frozen representations.

Execution jobs were reference estimation `16046356`, full feature extraction `16046900`,
population audit `16048098`, frontier `16048266` and result lock `16048418`.

