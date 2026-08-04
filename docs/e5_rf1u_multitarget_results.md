# E5-RF1U unpaired multi-target results

**Status:** POST-CORE EXTENSION RESULT — PFM endpoints opened once under the frozen contract
**Updated:** 2026-08-04
**Contract:** [`e5_rf1u_multitarget_contract.md`](e5_rf1u_multitarget_contract.md)

RF1U is a post-core method-development condition. The PanNormal E0--E7 locks, the five-method E5
primary ranking and the result-locked RF1 are unchanged. RF1M remains closed.

## Execution

1,308 shards, 436 per target, covering 3 targets × 2 conditions × 4 PFMs × 109 slides ×
6 scanners × 100 locations. Gains were fitted on training folds from pooled population band
energies only. SLURM: `16191865` (9 energy tasks), `16242800`/`16243519` (12 extraction tasks),
`16250768` (frontier), all `COMPLETED 0:0`.

### Pipeline validation

The Reinhard comparator was re-rendered rather than borrowed, because RF1U uses the RF1
five-fold scheme while the locked E5 comparators use 109-fold LOSO. Despite that difference the
re-rendered `reinhard_at2` reproduces the locked E5 Reinhard relative radius reduction almost
exactly:

| PFM | locked E5 | re-rendered | difference |
|---|---:|---:|---:|
| ResNet50 | 27.56% | 27.57% | +0.01 pp |
| UNI v1 | 6.04% | 6.05% | +0.01 pp |
| CONCH v1 | 11.90% | 11.91% | +0.01 pp |
| Virchow2 | 3.68% | 3.63% | −0.05 pp |

The runtime had drifted from the frozen pin; Amendment 1 records the bit-identical drift audit
that licensed extraction under it.

## Primary result

Seven of twelve `(target, PFM)` cells are **safe and improved over Reinhard**. All seven are at
the GT450 or S60 target. **None is at AT2.**

Relative scanner-radius reduction from raw:

| Target | PFM | Reinhard | RF1U | change |
|---|---|---:|---:|---:|
| AT2 | ResNet50 | 27.57% | 19.46% | **−8.11 pp** |
| AT2 | UNI v1 | 6.05% | −1.12% | **−7.17 pp** |
| AT2 | CONCH v1 | 11.91% | 6.51% | **−5.40 pp** |
| AT2 | Virchow2 | 3.63% | −7.37% | **−11.01 pp** |
| GT450 | ResNet50 | 28.53% | **32.85%** | +4.32 pp |
| GT450 | UNI v1 | 4.04% | **6.86%** | +2.82 pp |
| GT450 | CONCH v1 | 14.52% | **19.56%** | +5.04 pp |
| GT450 | Virchow2 | −0.08% | **4.04%** | +4.12 pp |
| S60 | ResNet50 | 27.65% | **31.84%** | +4.19 pp |
| S60 | UNI v1 | 5.30% | 5.56% | +0.27 pp |
| S60 | CONCH v1 | 12.13% | **16.88%** | +4.75 pp |
| S60 | Virchow2 | 4.30% | **6.69%** | +2.39 pp |

At AT2 every paired difference favours Reinhard with a CI excluding zero. At GT450 all four
PFMs favour RF1U; at S60 three of four do, UNI being the exception with a paired CI lower bound
of −0.00024.

## Why the target decides the outcome

The fitted gains differ in direction, not only in size. Mean log gain over sources and bands:

| Target | mean log gain | regime |
|---|---:|---|
| AT2 | **−0.212** | attenuation dominates |
| GT450 | +0.168 | amplification dominates |
| S60 | +0.193 | amplification dominates |

Toward AT2, GT450 and S60 receive band gains of 0.51--0.66, a heavy blur; only AKOYA is
sharpened. Toward GT450 or S60 nearly every source is sharpened, AKOYA most of all at 2.7 in the
finest band.

The failures follow that split exactly. At AT2 the content non-inferiority gate fails for UNI v1
(Δ −0.03384, CI lower −0.03645) and Virchow2 (Δ −0.03662, CI lower −0.03949) against the frozen
−0.02 margin, and UNI v1 also fails the collapse gate on precisely the two most heavily blurred
scanners: GT450 variance trace 0.775 and median pairwise distance 0.872, S60 variance trace
0.865, all below the 0.90 threshold. The three collapse failures in the whole grid are these.

Blurring toward the softer reference did not even buy invariance in exchange: RF1U's radius is
larger than plain Reinhard's at AT2 in all four PFMs, and worse than raw for UNI v1 and
Virchow2. Matching band energy downward produces images that are statistically closer to AT2 but
representationally further from it.

## Interpretation

The controlled comparison here is the target, since the transform, folds, shrinkage, gates and
encoding pass are identical across the three. On that comparison the destination matters more
than the method: the same estimator is unsafe and counterproductive toward AT2 and safe and
improved toward either sharper scanner.

This is the study's own thesis appearing inside its method development. Removing high-frequency
content lowers image-domain discrepancy while damaging representation, which is the failure mode
the frozen collapse and content gates exist to catch — and they caught it, in the PFMs most
sensitive to it.

It also qualifies AT2's role. AT2 is the study's relative transfer reference because it is the
paired acquisition anchor, not because it is a preferred destination for harmonization. These
results give no support for treating the softest scanner as the canonical target.

## What is not claimed

- No statement is made about clinical or biological preference among scanners. Lower scanner
  radius toward GT450 is a representation-geometry result.
- RF1U does not enter the frozen five-method E5 primary ranking and does not replace RF1.
- The comparison covers three targets from one 109-slide cohort. Whether the amplification
  advantage holds for a scanner sharper than GT450, or outside this cohort, is untested.
- Absolute RF1U numbers use the RF1 five-fold scheme and are compared with the locked E5 table
  as context only.

## Artifacts

- [`summary.json`](../outputs/rf1u_multitarget/frontier/summary.json)
- [`endpoint_summary.csv`](../outputs/rf1u_multitarget/frontier/endpoint_summary.csv)
- [`incremental_vs_reinhard.csv`](../outputs/rf1u_multitarget/frontier/incremental_vs_reinhard.csv)
- [`collapse_gates.csv`](../outputs/rf1u_multitarget/frontier/collapse_gates.csv)
- [`fitted_band_gains.csv`](../outputs/rf1u_multitarget/frontier/fitted_band_gains.csv)
- [`runtime drift audit`](../outputs/rf1u_multitarget/runtime_drift/summary.json)
