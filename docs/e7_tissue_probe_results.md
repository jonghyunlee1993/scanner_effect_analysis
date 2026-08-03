# E7 grouped tissue-probe results

**Status:** result locked on 2026-08-03  
**Contract:** [`e7_tissue_probe_execution_contract.md`](e7_tissue_probe_execution_contract.md)  
**Result lock:** `outputs/e7_tissue_probe_results_lock/summary.json`

## Population and inference gate

- The probe used raw AT2 training embeddings only and excluded the held-out physical slide
  from all tissue centroids. The one singleton tissue was unevaluable, leaving 108 slides and
  36 tissue classes; the minimum-three sensitivity retained 98 slides and 31 tissues.
- Four GPU row jobs each passed exactly 6,480 source rows and 108 AT2 rows. The combined
  population contained 25,920 source rows and 432 AT2 rows, with all expected LOSO/LOTO,
  method, scanner, slide and tissue identities.
- Point estimates weighted tissue classes equally. Confidence intervals used the frozen 5,000
  tissue-then-slide hierarchical bootstrap and preserved correction-versus-raw pairing.
- This probe has no accept/reject threshold. It is secondary coarse tissue-type evidence; the
  E5 paired-content and every-scanner collapse gates remain the authoritative fidelity decision.

## Baseline tissue signal

Held-out raw AT2 top-1 accuracy was 29.2% for ResNet50, 56.3% for UNI v1, 55.6% for CONCH v1
and 50.3% for Virchow2. The corresponding raw five-source-scanner estimates were 26.4%, 54.2%,
54.8% and 48.6%. Thus the frozen embeddings contained measurable coarse tissue signal, but
performance was far from a clinical classifier and differed markedly by PFM.

## Correction-associated top-1 change

Entries are percentage-point changes from the matched raw source-scanner baseline, shown as
primary LOSO / exact-LOTO transfer sensitivity.

| PFM | Reinhard | Paired OD | Frequency | CORAL | Procrustes |
|---|---:|---:|---:|---:|---:|
| ResNet50 | +1.06 / +1.05 | +0.66 / +0.65 | +0.09 / +0.08 | +0.16 / −0.20 | +1.67 / +1.28 |
| UNI v1 | −0.38 / −0.37 | −1.08 / −1.10 | +0.02 / +0.02 | −2.92 / −3.72 | +0.56 / −0.65 |
| CONCH v1 | +0.04 / +0.04 | −0.39 / −0.37 | +0.05 / +0.06 | −0.34 / −0.60 | +0.11 / −0.09 |
| Virchow2 | −0.14 / −0.14 | −1.17 / −1.16 | +0.02 / +0.02 | −0.44 / −1.03 | +0.29 / −0.29 |

Most top-1 intervals included zero. The three exceptions under primary LOSO were a very small
positive frequency-calibration change for ResNet50 (+0.089 pp, 95% CI +0.0004 to +0.180), a
negative UNI v1 CORAL change (−2.92 pp, −5.46 to −0.69), and the already globally unsafe
Virchow2 paired-OD change (−1.17 pp, −2.23 to −0.13). After excluding classes with fewer than
three slides, the UNI v1 CORAL top-1 interval included zero, while its negative top-5, margin
and profile-agreement contrasts remained.

## Geometry-sensitive secondary metrics

Orthogonal Procrustes left LOSO top-1 statistically indistinguishable from raw in all four
PFMs, while increasing matched AT2 centroid-profile agreement in all four: +0.0142 for
ResNet50, +0.0402 for UNI v1, +0.00347 for CONCH v1 and +0.0132 for Virchow2, with every 95%
CI above zero. The positive profile-agreement result persisted under exact LOTO. However,
LOTO correct-tissue margin decreased for UNI v1 (−0.0107, 95% CI −0.0164 to −0.0051) and
Virchow2 (−0.00405, −0.00629 to −0.00177). The largest LOTO-minus-LOSO top-1 change was
UNI v1 Procrustes at −1.21 pp (−1.72 to −0.68).

CORAL showed the converse geometry diagnostic: centroid-profile agreement decreased in all
four PFMs under both LOSO and LOTO. LOSO changes ranged from −0.00292 to −0.0594 and all
intervals excluded zero; LOTO changes ranged from −0.00436 to −0.0707. These observations do
not retroactively change the frozen E5 fidelity decision, because the tissue probe is a
different, secondary endpoint without a non-inferiority margin. They show why passing pooled
content and collapse gates cannot be promoted to a claim that every class-relative direction
in the representation is preserved.

## Interpretation and provenance

No correction improved every tissue metric in every PFM. Procrustes offered the strongest
overall combination of E5 invariance, primary content safety and paired tissue-profile
agreement, but even it did not establish universal tissue-geometry preservation under LOTO.
The result supports a bounded claim: several methods reduce scanner dispersion without global
collapse, while their coarse tissue geometry remains method-, PFM- and transfer-population
dependent. Tissue type is the only available biological grouping, so this is not biological
non-inferiority, morphology validation or clinical validation.

Figure 6 is
`outputs/e7_tissue_probe/figure6_content_tissue_evidence.{png,pdf}`. Row construction,
hierarchical summary, result lock and PanNormal core-lock jobs were `16052958`, `16052959`,
`16052960` and `16053562`. The full endpoint tables, scanner diagnostics, exact bootstrap
weights and hashes are in `outputs/e7_tissue_probe/` and
`outputs/e7_tissue_probe_results_lock/`.
