# E6 LOSO correction-heterogeneity results

**Status:** primary LOSO heterogeneity and exact LOTO transfer-sensitivity results locked on
2026-08-03  
**Contract:** [`e6_heterogeneity_execution_contract.md`](e6_heterogeneity_execution_contract.md)  
**Result lock:** `outputs/e6_heterogeneity_results_lock/summary.json`

## Population and numerical gate

- The analysis used only the already locked E5 cross-fitted held-out predictions. It formed
  five disjoint 20-location replicate means within each of 109 physical slides.
- The audited input contained exactly 10,900 radius-benefit rows and 54,500 scanner-specific
  content-change rows across four PFMs, five methods, 37 tissue types and five source scanners.
- For each full cell and its minimum-three-slide sensitivity fit, profiled REML estimated a
  tissue random slope, slide-within-tissue random slope and replicate-sampling variance. All
  240 full and zero-tissue-variance reduced fits converged.
- The first execution exposed two reduced-model line-search terminations. Before accepting or
  interpreting any E6 result, the reduced fit was rerun with the same three-start numerical
  stabilization already used by the full model. No model term, threshold, endpoint or result
  selection changed. The complete analysis was then recomputed and independently audited.
- Tissue class sizes ranged from one to six slides. The prespecified sensitivity excluded one
  one-slide and five two-slide tissue types, leaving 98 slides in 31 tissues.

## Tissue heterogeneity

BH was applied separately to the 20 radius and 100 content tissue-variance tests.

| Family | Full 37 tissues | ≥3-slide sensitivity |
|---|---:|---:|
| Radius-benefit tissue variance | 18/20 cells, BH q<0.05 | 17/20 |
| Scanner-specific content-change tissue variance | 71/100 cells, BH q<0.05 | 70/100 |

Radius-benefit heterogeneity was detected for every cell except CONCH v1 Reinhard and UNI v1
CORAL in the full population. Tissue SDs were largest for paired OD affine in UNI v1 (0.0200),
Virchow2 (0.0177) and ResNet50 (0.0149). Frequency calibration also sometimes had a
statistically detectable tissue component, but its absolute effect and tissue SD were near
zero; significance is not evidence of a practically useful correction.

Content-change heterogeneity was especially frequent for AKOYA (20/20 method--PFM cells),
followed by S60 (16/20), GT450 (14/20), S360 (11/20) and VERSA (10/20). This confirms that a
pooled content endpoint can conceal structured source-scanner and tissue response, while the
primary E5 safety decision remains the prespecified pooled plus every-scanner collapse rule.
The minimum-three sensitivity changed only three of 100 content gates and one of 20 radius
gates near q=0.05.

## Tissue-specific winner diagnostic

Only methods globally safe for each PFM were eligible. Orthogonal Procrustes was not reselected;
it remained the frozen E5 global winner.

| PFM | Procrustes first, all tissues | Procrustes first, ≥3-slide tissues |
|---|---:|---:|
| ResNet50 | 37/37 (100.0%) | 31/31 (100.0%) |
| UNI v1 | 37/37 (100.0%) | 31/31 (100.0%) |
| CONCH v1 | 36/37 (97.3%) | 30/31 (96.8%) |
| Virchow2 | 34/37 (91.9%) | 29/31 (93.5%) |

CORAL ranked first for CONCH v1 choroid plexus and for Virchow2 adrenal, brain cortex and
choroid plexus; each of these tissues had three slides and therefore remained in the
minimum-three sensitivity. These BLUP rankings are descriptive response diagnostics, not
independent biological performance endpoints.

## Secondary baseline-spectrum prediction

The post-E6 secondary contract fixed six existing spectrum/content-texture predictors before
their associations were computed: three raw-AT2 band powers and the across-scanner transfer SD
in the same three bands. Exact tissue-held-out ridge prediction produced positive predictive
R² in 12/20 method--PFM cells, but the hierarchical-bootstrap lower bound exceeded zero in only
three cells: ResNet50 frequency calibration (R² 0.642, 95% CI 0.461--0.756), UNI v1 Reinhard
(0.310, 0.053--0.487) and UNI v1 paired OD affine (0.279, 0.003--0.468). Their minimum-three
estimates were 0.623, 0.334 and 0.315, respectively.

CORAL and Procrustes response generally had negative tissue-held-out predictive R² under this
small fixed predictor set even though both methods had strong mean E5 radius reduction. The
high R² for ResNet50 frequency calibration must also be read beside its very small absolute
effect: its absolute-error improvement over the null was only 0.00019 radius units. Thus the
available baseline spectrum predicts selected image-method response, not correction response
universally. Because no independent morphology annotation exists, these predictors are not
reported as a morphology model or causal mechanism.

## Exact leave-one-tissue-out transfer sensitivity

The separately materialized 37-fold LOTO population removed every slide from the held-out
tissue from correction fitting. Its audit passed all 872 image/feature shards and all
1,308,000 embeddings with SHA-256 verification. Relative to the primary LOSO analysis, none
of the 20 method--PFM cells changed either its fidelity-safe or safe-and-improved decision.
Reinhard, CORAL and orthogonal Procrustes therefore remained common-safe and improved across
all four PFMs; frequency calibration remained common-safe but did not improve invariance in
UNI v1 or CONCH v1.

| PFM | LOTO Reinhard RR | LOTO CORAL RR | LOTO Procrustes RR | Largest absolute RR change from LOSO |
|---|---:|---:|---:|---:|
| ResNet50 | 27.6% | 34.0% | 43.4% | 0.90 pp |
| UNI v1 | 6.0% | 13.9% | 19.8% | 2.66 pp |
| CONCH v1 | 11.9% | 20.3% | 22.4% | 1.65 pp |
| Virchow2 | 3.7% | 14.3% | 17.4% | 3.21 pp |

The image-space estimates were nearly unchanged under LOTO. The larger attenuation was in
feature-space gain, with CORAL and Procrustes RR decreasing by at most 3.21 and 2.66 percentage
points, respectively, without changing a decision. Thus the E5 LOSO conclusion transfers to
unseen tissue types in this cohort, while LOSO remains the prespecified primary analysis.

## Interpretation, figure and provenance

Correction response is strongly structured by tissue and physical slide even after exact LOSO
cross-fitting. Nevertheless, the E5 global Procrustes winner is highly stable rather than
universally tissue-specific. Both facts matter: a global method can be robust on average while
still having tissue-dependent effect magnitude and a small number of predicted local rank
changes.

The E6 diagnostic figure is
`outputs/e6_loso_heterogeneity/figure_e6_correction_heterogeneity.{png,pdf}`. It shows
radius-benefit tissue SD, the fraction of source scanners with content heterogeneity and global
winner retention. Execution jobs were contrast construction `16048099`, initial numerical
diagnostic `16048267`, stabilized full recomputation `16048373`, final figure recomputation
`16048439` and independent result lock `16048477`.

The baseline-spectrum secondary outputs and figure are in
`outputs/e6_baseline_spectrum_predictor/`; execution and result-lock jobs were `16051104` and
`16051375`. Its separate frozen-after-E6 contract is
[`e6_baseline_spectrum_predictor_contract.md`](e6_baseline_spectrum_predictor_contract.md).
The exact LOTO population audit, frontier and result-lock jobs were `16052955`, `16052956` and
`16052957`. Their outputs are in `outputs/e6_loto_features/audit/`,
`outputs/e6_loto_frontier/` and `outputs/e6_loto_results_lock/`; the supplemental transfer
figure is `outputs/e6_loto_frontier/figure_s_loto_transfer_sensitivity.{png,pdf}`.
