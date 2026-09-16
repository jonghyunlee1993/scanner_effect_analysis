# E4--E7 pre-outcome decision record

**Status:** FROZEN — all seven decisions approved without amendment  
**Prepared:** 2026-08-03  
**Approved:** 2026-08-03 by the user in the project conversation  
**Outcome access at freeze:** raw feature population integrity was audited, but E4--E7 metric
outcomes had not been opened.

This document is the single decision gate between the completed E0--E3 analysis and the
four-PFM population experiment. A decision becomes frozen only after the user explicitly
approves it. Any post-approval change requires a dated amendment stating whether outcome data
had been accessed.

## Already frozen facts

- Cohort: 109 physical slides, 37 normal tissue types, six scanners and 100 matched
  locations per scanner--slide.
- Reference scanner: AT2.
- Pixel grid: native WSI rendered at 0.5052 µm/px with explicit anti-aliasing.
- Core PFM panel: ResNet50, UNI v1, CONCH v1 and Virchow2.
- Raw feature population: 436/436 model--slide shards and 261,600/261,600 embeddings passed.
- Inference, splitting and bootstrap unit: physical slide, never patch.
- Biology scope: tissue type only; no biological non-inferiority or clinical-validation claim.
- PLISM: excluded from PanNormal completion and deferred to a separate post-core protocol.

## Decision 1 — primary scanner-dispersion endpoint

**Proposed definition**

For PFM `p`, condition `m`, physical slide `l`, matched location `i` and scanner `s`,
L2-normalize the embedding `z_plism`. Let the six-scanner location centroid be

\[
\bar z_{plim}=\frac{1}{6}\sum_s z_{plism}.
\]

The location dispersion is

\[
r_{plim}=\sqrt{\frac{1}{6}\sum_s \lVert z_{plism}-\bar z_{plim}\rVert_2^2}.
\]

Average the 100 location dispersions within each slide, then give each of the 109 slides equal
weight. Define relative radius reduction as

\[
\mathrm{RR}_{pm}=1-R_{pm}/R_{p,raw}.
\]

- Report `R`, raw-minus-method difference and RR; never clip negative RR.
- Set RR to N/A when `R_raw < 0.01`.
- An invariance improvement requires the lower 95% physical-slide bootstrap CI of
  (R_{raw}-R_m) to be greater than zero.
- Scanner BACC, paired consensus gain and same-location retrieval remain secondary diagnostics.

**Decision requested:** approve the L2/centroid-RMS definition, location→slide aggregation,
`R_raw >= 0.01` gate and CI rule, or provide replacements.

## Decision 2 — primary representation/content endpoint

**Proposed definition**

For every non-AT2 scanner embedding under condition `m`, compute cosine similarity to the
raw AT2 embedding from the same physical location. Subtract the q95 cosine among the other
99 raw AT2 locations in the same slide:

\[
M_{plism}=\cos(z^{m}_{plis}, z^{raw}_{pli,AT2})
-Q_{0.95,j\ne i}\cos(z^{m}_{plis},z^{raw}_{plj,AT2}).
\]

This makes the endpoint a matched-content margin rather than raw self-similarity. Aggregate
location→scanner→slide and weight slides equally. The paired preservation contrast is

\[
\Delta M_{pm}=M_{pm}-M_{p,raw}.
\]

- Primary non-inferiority margin: lower 95% slide-bootstrap CI of ΔM must be at least `-0.02`.
- Report each source scanner separately as a safety diagnostic; the pooled primary estimate
  cannot hide a scanner-specific reversal in the heterogeneity table.
- Raw→corrected self-cosine and neighborhood preservation remain secondary because they can
  penalize useful nuisance removal.

**Decision requested:** approve the matched-minus-within-slide-unmatched-q95 margin and `-0.02`
non-inferiority threshold, or select self-cosine/retrieval/another primary endpoint and margin.

## Decision 3 — collapse guardrail

**Proposed definition**

For each PFM, held-out slide and source scanner, compare corrected with raw embeddings across
the 100 locations using three ratios:

1. total embedding variance, defined as covariance trace;
2. entropy effective rank, `exp(-sum(q_k log q_k))` for normalized covariance eigenvalues;
3. median of all 4,950 within-slide pairwise Euclidean distances after L2 normalization.

A condition passes only when all three ratios have point estimate at least `0.90` and lower
95% slide-bootstrap CI at least `0.85`. Report pooled and scanner-specific ratios. Collapse
controls are expected to fail and are retained in every figure/table.

**Decision requested:** approve the three diagnostics, entropy-rank definition, all-three rule
and `0.90/0.85` thresholds. Also confirm whether the hard gate must pass every source scanner
(recommended) or only the equal-scanner pooled estimate.

## Decision 4 — comparator set and fitting target

**Proposed minimum benchmark**

| Space | Method | Frozen fitting rule |
|---|---|---|
| Image | Reinhard | source scanner→AT2 train-slide color statistics only |
| Image | Paired OD affine | source scanner→matched AT2 train patches only |
| Image | Frequency calibration | train-slide radial source/AT2 transfer; fixed gain cap and RGB clipping audit |
| Feature | CORAL | source scanner→AT2 train-slide distribution only; regularized covariance |
| Feature | Orthogonal Procrustes | paired source/AT2 train embeddings only; no scale term |

AT2 is the fitting target for all primary comparators. Macenko is recommended as a
Supplement-only method, not part of the minimum primary benchmark. Controls remain raw,
complete/partial HF attenuation, HF boost, registered
same-location 25% HF mean and global coefficient mean.

Before execution the frequency-calibration gain cap and CORAL covariance regularizer will be
fixed by input-domain stability tests, never PFM outcomes.

**Decision requested:** approve the five-method minimum set and AT2 target; choose the
recommended Supplement-only placement for Macenko or omit it.

## Decision 5 — generalization and leakage control

**Proposed split contract**

- Primary: 109-fold leave-one-physical-slide-out (LOSO).
- Secondary: 37-fold leave-one-tissue-type-out for correction-transfer heterogeneity.
- Every learned image/feature transform is fitted inside the training fold.
- All patches and all six scans of a physical slide stay in one fold.
- Cross-fitted held-out predictions are materialized before metric aggregation.
- No retraining occurs inside the uncertainty bootstrap; the bootstrap resamples the 109
  cross-fitted slide result blocks 5,000 times with a frozen seed.
- Tissue-type retrieval/grouped probe is a separate secondary analysis with grouped physical
  slide splits; it is not treated as evidence of independent biology.

**Decision requested:** approve LOSO primary, leave-one-tissue-out secondary and the
cross-fit-then-slide-bootstrap inference scheme.

## Decision 6 — multi-PFM and method claim rule

**Proposed rule**

- Each of the four PFMs is a primary analysis stratum; no model is dropped after outcome review.
- A method is `safe for PFM p` only if it passes the content non-inferiority and collapse gates;
  invariance improvement is then evaluated and ranked.
- `Common safe across core PFMs` is allowed only when all four PFMs pass.
- Equal-weight PFM summaries and standardized effect meta-analysis are secondary and cannot
  override a failed PFM.
- Report the full method × PFM grid, including null, harmful and failed-fidelity conditions.
- Related secondary endpoint p-values use BH correction; primary decisions use prespecified
  effect/CI rules.

**Decision requested:** approve the per-PFM primary and four-of-four common-safe claim rule.

## Decision 7 — tissue heterogeneity and content evidence

**Proposed rule**

- E6 primary heterogeneity: scanner × correction × tissue and slide-within-tissue random slopes
  on cross-fitted slide/location contrasts.
- Tissue type is secondary coarse-content evidence only.
- Report tissue class size (1--6 slides), minimum-slide sensitivity and shrinkage estimates.
- Do not add a nucleus/spatial endpoint unless annotation/QC and a separate contract exist.

**Decision requested:** approve tissue type as the only current biological grouping and omit
nucleus/spatial analysis from the PanNormal core.

## Freeze record

- Approval text: `전부 승인`.
- Decisions 1--7 were accepted exactly as proposed.
- Decision 3 uses the every-source-scanner hard collapse gate.
- Decision 4 places Macenko in the Supplement only.
- E4--E7 metric computation, method ranking and PFM outcome inspection had not begun at the
  moment of approval.
- Repository commit identifier: unavailable because `.git` is read-only in the current
  workspace session; this does not alter the frozen analysis contract.

Any amendment after this point must be dated, identify the changed text and state which E4--E7
outcomes had already been accessed.

## Amendment A — post-outcome status note (2026-08-23)

**Outcome data had been accessed when this amendment was written.** It therefore changes no
decision, definition, threshold or endpoint, and none of the seven frozen decisions is reopened.
It exists only so a reader does not mistake a frozen decision for current status.

The frozen fact "PLISM: excluded from PanNormal completion and deferred to a separate post-core
protocol" is an accurate record of what was decided on 2026-08-03, and it was honoured. PLISM
was kept out of PanNormal completion, and the separate protocol was written and executed
afterwards as **E9** ([`e9_plism_native_ert_contract.md`](e9_plism_native_ert_contract.md),
results in [`e9_plism_core_results.md`](e9_plism_core_results.md)).

A second post-core condition, **E8**
([`e8_paired_residual_contract.md`](e8_paired_residual_contract.md)), was added after this
record was frozen, under its own pre-registration. Neither E8 nor E9 enters the frozen
five-method E5 primary ranking, and neither recomputed a locked artifact.

## Amendment B — submission-facing interpretation note (2026-09-16)

**All E8/E9 outcomes had been accessed when this note was written.** It changes no E0--E7
decision or locked artifact. It corrects only the terminology in Amendment A after the v3 claim
audit.

- The E8 contract remained labelled `DRAFT`, and selection ordering was amended after fold-0
  image results. Its thresholds were prospectively specified before PFM endpoint access, but the
  study is not described as formal preregistration or as establishing a universal image-domain
  ceiling.
- On E9's retained native-spectrum endpoint, fixed predictions P1--P3 all failed. The broader
  “5 of 8” exploratory scoreboard is not used as a confirmatory replication claim.
- Neither post-core extension supports clinical utility, unseen-scanner generalization or a
  deployment recommendation.
