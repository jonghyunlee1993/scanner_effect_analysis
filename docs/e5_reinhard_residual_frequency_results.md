# E5-RF1 Reinhard residual-frequency results

**Status:** RESULT-LOCKED post-core extension  
**Updated:** 2026-08-03  
**Scope:** post-core method-development extension; the frozen PanNormal E0--E7 conclusions
remain unchanged

## Question and frozen design

E5-RF1 asks whether correcting residual spatial-frequency response *after* Reinhard adds
value without returning to the nonphysical clipping used by the earlier low/high-frequency
diagnostic. The method is five-fold cross-fitted over 109 physical slides. It fits Reinhard
and the post-Reinhard source-to-AT2 radial-power gain on four folds, applies both to the held-
out fold, modifies only the shared mean-OD residual, and projects that scalar residual into
the exact RGB8-supporting OD interval. Candidate gain caps were selected exclusively from
image-domain range and projection diagnostics before any new PFM endpoint was opened.

The rationale and executable rules are recorded in
[`reinhard_frequency_extension.md`](reinhard_frequency_extension.md) and
[`e5_reinhard_residual_frequency_contract.md`](e5_reinhard_residual_frequency_contract.md).

## Input-only result

All 15 FOV-by-source-scanner cells passed at gain cap **1.25**, the largest passing candidate.
Cap 1.50 failed only for AKOYA because mean materially out-of-range proposed RGB exceeded the
pre-specified 0.005 limit at all three FOVs. At the selected cap:

- the largest cell mean material range fraction was 0.003298;
- the largest cell patch-level q99 material range fraction was 0.02581;
- the largest cell mean RGB MAE induced by gamut projection was 0.0002395;
- the largest mean projection fraction was 0.02211;
- final numerical-clamp MAE was exactly zero in every cell.

| Source scanner | Reinhard spectrum RMSE | RF1 spectrum RMSE | Relative change |
|---|---:|---:|---:|
| GT450 | 1.5200 | 1.1796 | −22.4% |
| VERSA | 0.3168 | 0.3117 | −1.6% |
| AKOYA | 1.2596 | 0.9954 | −21.0% |
| S60 | 1.1298 | 0.8963 | −20.7% |
| S360 | 0.2485 | 0.3219 | +29.6% |

Values are the mean of the three model-native FOV cells. Twelve of 15 cells improved; all
three non-improving cells were S360. Across cells, the mean RF1/Reinhard spectrum-RMSE ratio
was 0.9279, or a 7.2% mean relative reduction. Spectrum RMSE did not enter cap selection.

## Paired-patch visual audit

Five source-scanner examples were selected deterministically by an outcome-blind salted hash.
The audit displays paired AT2, raw source, Reinhard, RF1 and five-times-amplified absolute RF1
residual. The visible change is concentrated around boundaries and fine texture while the
Reinhard color field is retained. The visual artifact is descriptive, not a hallucination or
morphology-preservation test. Its final clamp error was zero.

- [`paired_patch_audit.png`](../outputs/e5_rf1_visual_audit/paired_patch_audit.png)
- [`sample_manifest.csv`](../outputs/e5_rf1_visual_audit/sample_manifest.csv)
- [`selected_cap_spectrum_cells.csv`](../outputs/e5_rf1_visual_audit/selected_cap_spectrum_cells.csv)

## Four-PFM frontier

The audited population contains 436 model--slide shards and 261,600 embeddings. RF1 passed
content non-inferiority and every-scanner collapse gates in all four PFMs, and scanner radius
improved relative to raw in all four.

| PFM | Reinhard RR | RF1 RR | Reinhard - RF1 radius (paired 95% CI) | RF1 - Reinhard content margin (95% CI) |
|---|---:|---:|---:|---:|
| ResNet50 | 27.56% | 29.71% | 0.00456 [0.00423, 0.00491] | +0.00078 [+0.00071, +0.00086] |
| UNI v1 | 6.04% | 6.66% | 0.00296 [0.00227, 0.00368] | +0.00100 [+0.00045, +0.00154] |
| CONCH v1 | 11.90% | 14.65% | 0.00723 [0.00628, 0.00823] | -0.00029 [-0.00059, -0.00000] |
| Virchow2 | 3.68% | 4.66% | 0.00342 [0.00254, 0.00429] | +0.00100 [+0.00056, +0.00144] |

`RR` is relative scanner-radius reduction from raw. Positive paired radius difference favors
RF1. RF1 reduced radius beyond Reinhard in every PFM with a positive 95% CI. The incremental
effect was modest: +0.61 to +2.75 RR percentage points. Content margin increased slightly in
three PFMs; CONCH showed a very small paired decrease while remaining substantially above raw
and passing the frozen non-inferiority gate. Every collapse cell passed; the worst lower CI
over scanners and collapse metrics was 0.911 for UNI v1, above the frozen 0.85 threshold.

The raw frequency-only method had RR of +0.50%, -0.05%, -0.47% and +0.32% for ResNet50, UNI
v1, CONCH v1 and Virchow2. The RF1 result therefore supports the ordering hypothesis:
frequency correction is materially more useful after global Reinhard color alignment than as
a stand-alone raw-image transform.

## Grouped tissue-type secondary probe

The probe used 36 evaluable tissue types/108 slides, with a pre-specified sensitivity of 31
tissue types/98 slides having at least three slides. This is coarse tissue-type evidence, not
morphological, biological or clinical validation.

| PFM | RF1 top-1 delta vs raw, full (95% CI) | RF1 top-1 delta vs Reinhard, full (95% CI) | RF1 delta vs Reinhard, min-3 (95% CI) |
|---|---:|---:|---:|
| ResNet50 | +1.22 pp [-0.10, +2.74] | +0.16 pp [-0.06, +0.41] | +0.24 pp [+0.01, +0.49] |
| UNI v1 | +0.15 pp [-0.81, +1.16] | +0.52 pp [+0.27, +0.81] | +0.48 pp [+0.22, +0.77] |
| CONCH v1 | +0.08 pp [-0.55, +0.74] | +0.05 pp [-0.30, +0.43] | -0.08 pp [-0.42, +0.24] |
| Virchow2 | -0.05 pp [-0.65, +0.55] | +0.09 pp [-0.07, +0.25] | +0.07 pp [-0.10, +0.25] |

There was no uniform top-1 gain across PFMs. UNI showed a positive incremental top-1 effect
in both analysis sets, and ResNet50 was positive only in the min-three sensitivity. CONCH and
Virchow2 top-1 differences were compatible with zero. Across the other tissue endpoints,
UNI margin/profile agreement, ResNet50 profile agreement, and Virchow2 top-5/profile
agreement improved relative to Reinhard; CONCH profile agreement decreased by a very small
0.00030 [-0.00053, -0.00007]. These mixed secondary endpoints do not support a broad biology
claim, but they also do not indicate a coherent tissue-fidelity collapse.

## Interpretation and generator decision

The analytic stage is successful but incremental. A shared-OD residual moved the
post-Reinhard spectrum toward paired AT2 for GT450, AKOYA and S60 with small gamut projection,
no final hard clipping, and a statistically detectable additional scanner-radius reduction
in every PFM. S360 remains the important counterexample: its spectrum moved away from AT2,
showing that a universal radial gain is not the final solution.

These results do **not** justify replacing RF1 with CycleGAN in the current paper. Paired
same-tissue scanner images are available, so an unpaired adversarial objective would discard
useful supervision and increase hallucination risk. If a learned stage is pursued, the next
method should be a scanner-conditioned, paired residual model that can learn to attenuate or
skip the correction for cases such as S360. It should be frozen under a new contract with
image-only nested selection, identity/paired-structure constraints, frequency-band losses,
gamut-safe output parameterization, hallucination audits and later PLISM external validation.
Until those requirements are met, RF1 is the stronger manuscript result: interpretable,
auditable and already better than Reinhard on the primary representation endpoint.

## Locked artifacts

- [`endpoint_summary.csv`](../outputs/e5_rf1_frontier/endpoint_summary.csv)
- [`incremental_vs_reinhard.csv`](../outputs/e5_rf1_frontier/incremental_vs_reinhard.csv)
- [`tissue endpoint_summary.csv`](../outputs/e5_rf1_tissue_probe/endpoint_summary.csv)
- [`tissue incremental_vs_reinhard.csv`](../outputs/e5_rf1_tissue_probe/incremental_vs_reinhard.csv)
- [`result-lock summary.json`](../outputs/e5_rf1_results_lock/summary.json)
- [`artifact_manifest.csv`](../outputs/e5_rf1_results_lock/artifact_manifest.csv)
