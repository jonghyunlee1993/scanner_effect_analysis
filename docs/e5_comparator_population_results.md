# E5 actual-comparator population results

**Status:** primary five-method E5 result locked on 2026-08-03  
**Decision contract:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md)  
**Execution contract:** [`e5_comparator_execution_contract.md`](e5_comparator_execution_contract.md)  
**Result lock:** `outputs/e5_comparator_results_lock/summary.json`

## Frozen population and numerical gate

- Three image methods (Reinhard Lab, paired OD affine and frequency calibration) and two
  feature methods (CORAL and orthogonal Procrustes) were fit with exact 109-fold
  leave-one-physical-slide-out cross-fitting against raw AT2.
- The input-only stability audit was completed before E5 PFM endpoints were opened. It fixed
  the panel-wide CORAL shrinkage at `0.05` and the frequency gain cap at `1.03`; Procrustes
  stability passed. The frequency cap was the largest candidate satisfying every
  FOV/source-scanner clipping gate.
- Image features: 436/436 shards and 784,800/784,800 embeddings.
- Feature harmonization: 436/436 shards and 523,200/523,200 embeddings.
- Total primary comparator population: **1,308,000 embeddings**, about 5.16 GB. All 872
  shards passed scanner/location identity, raw-AT2 exact equality, finite positive norm,
  checkpoint/source/statistic/stability hash and schema audits.
- Endpoint aggregation produced 2,616 slide-invariance rows, 13,080 slide-content rows,
  39,240 slide-collapse rows and 24 method/PFM endpoint rows. Uncertainty used the frozen
  5,000 physical-slide bootstrap replicates with seed `20260803`.

## Primary endpoint grid

`RR` is the point relative scanner-radius reduction. `ΔR CI lower` is the lower 95% CI of
raw-minus-method scanner radius; it must be positive for invariance improvement. `ΔM CI
lower` must be at least −0.02. The collapse column is the minimum lower CI across all five
source scanners and three diagnostics; the gate additionally requires every corresponding
point estimate to be at least 0.90.

| PFM | Method | RR | ΔR CI lower | ΔM CI lower | worst collapse CI lower | Frozen decision |
|---|---|---:|---:|---:|---:|---|
| ResNet50 | Reinhard | +27.6% | +0.0556 | +0.0142 | 0.9761 | safe + improved |
| ResNet50 | Paired OD affine | +25.7% | +0.0502 | +0.0118 | 0.9578 | safe + improved |
| ResNet50 | Frequency calibration | +0.5% | +0.0010 | +0.0003 | 0.9979 | safe + improved |
| ResNet50 | CORAL | +34.9% | +0.0704 | +0.0185 | 0.9593 | safe + improved |
| ResNet50 | Orthogonal Procrustes | +43.9% | +0.0896 | +0.0228 | 0.9521 | safe + improved |
| UNI v1 | Reinhard | +6.0% | +0.0267 | +0.0190 | 0.9267 | safe + improved |
| UNI v1 | Paired OD affine | +8.9% | +0.0365 | +0.0174 | 0.8773 | **unsafe** |
| UNI v1 | Frequency calibration | −0.1% | −0.0003 | −0.0003 | 0.9972 | safe; no improvement |
| UNI v1 | CORAL | +16.3% | +0.0753 | +0.0629 | 1.0123 | safe + improved |
| UNI v1 | Orthogonal Procrustes | +22.5% | +0.1045 | +0.0859 | 0.9973 | safe + improved |
| CONCH v1 | Reinhard | +11.9% | +0.0297 | +0.0130 | 0.9843 | safe + improved |
| CONCH v1 | Paired OD affine | −3.3% | −0.0117 | −0.0087 | 0.9787 | safe; no improvement |
| CONCH v1 | Frequency calibration | −0.5% | −0.0014 | −0.0006 | 0.9944 | safe; no improvement |
| CONCH v1 | CORAL | +22.0% | +0.0555 | +0.0221 | 0.9805 | safe + improved |
| CONCH v1 | Orthogonal Procrustes | +23.9% | +0.0601 | +0.0233 | 0.9916 | safe + improved |
| Virchow2 | Reinhard | +3.7% | +0.0105 | +0.0019 | 0.9601 | safe + improved |
| Virchow2 | Paired OD affine | −11.7% | −0.0466 | −0.0341 | 0.9668 | **unsafe** |
| Virchow2 | Frequency calibration | +0.3% | +0.0010 | +0.0003 | 0.9983 | safe + improved |
| Virchow2 | CORAL | +17.6% | +0.0586 | +0.0333 | 1.0073 | safe + improved |
| Virchow2 | Orthogonal Procrustes | +20.0% | +0.0672 | +0.0374 | 0.9935 | safe + improved |

UNI paired OD affine failed because VERSA variance-trace ratio had point estimate `0.8865`
despite lower CI `0.8773`; this is why the lower-CI column alone does not show the failure.
Virchow2 paired OD affine failed pooled content non-inferiority (`ΔM` lower CI `−0.0341`).

## Locked interpretation

1. **Three actual methods were common safe and improved across all four PFMs:** Reinhard,
   CORAL and orthogonal Procrustes. Orthogonal Procrustes was the largest-RR safe method in
   each PFM (ResNet50 43.9%, UNI v1 22.5%, CONCH v1 23.9%, Virchow2 20.0%).
2. Feature-space harmonization was stronger than the deployable image methods in this
   representation endpoint. This is a claim about the frozen embeddings, not proof that an
   image was biologically restored or that a downstream clinical task improved.
3. Frequency calibration was common fidelity-safe but its conservative input-stability cap
   yielded little change: it improved ResNet50 and Virchow2 only, and did not improve UNI v1
   or CONCH v1.
4. Paired OD affine was not common-safe or common-improved. It produced a prespecified
   **invariance-positive but fidelity-unsafe** cell in UNI v1: RR was +8.9% with positive
   ΔR CI, but the VERSA variance-collapse point gate failed. Thus the protocol changed the
   decision for one actual method/PFM cell even though the top-ranked method itself remained
   Orthogonal Procrustes under both rankings in all four PFMs.
5. Scanner-specific content diagnostics exposed additional OD-affine reversals that the
   pooled endpoint must not hide: VERSA/AKOYA lower CIs were below −0.02 in CONCH v1 and UNI
   v1, and AKOYA was strongly negative in Virchow2. These are heterogeneity signals for E6,
   not a post-hoc change to the pooled primary gate.
6. This result does not invalidate the E4 degeneracy demonstration. E4 showed that low radius
   can arise from destructive blur/global coefficient averaging, whereas E5 shows that
   several real cross-fitted methods occupy the safe region. The actual-method winner happened
   to agree across unconstrained and fidelity-constrained ranking; that null rank-reversal
   result is reported without selective reframing.

## Figure and provenance

Main Figure 5 is
`outputs/e5_comparator_figure/figure5_actual_correction_frontier.{png,pdf}`. It displays the
full method × PFM grid for radius reduction, content lower CI, worst collapse lower CI and the
unconstrained versus fidelity-constrained winner. The primary result lock contains 33 hashed
artifacts and covers 54,936 slide-result rows.

Primary execution jobs:

- image sufficient statistics `15973957`; feature sufficient statistics `15973958`;
- final input-only stability gate `15976975`;
- image population `15977612`; feature LOSO population `15977374`;
- population audit `16046619`; final frontier `16047066`;
- final Figure 5 `16047381`; final result lock `16047699`.

Macenko is governed by the separate
[`e5_macenko_supplement_contract.md`](e5_macenko_supplement_contract.md) and is excluded from
all primary common-safe and winner statements above.

