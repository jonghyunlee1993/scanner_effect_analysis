# E6 correction-heterogeneity execution contract

**Status:** FROZEN before E6 tissue-effect outcome access on 2026-08-03  
**Parent decision:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md) Decisions 5--7  
**Input result:** [`e5_comparator_population_results.md`](e5_comparator_population_results.md)

## Primary LOSO heterogeneity population

- Use the already materialized E5 LOSO held-out predictions for the five primary actual
  methods and all four PFMs. No transform is refit for this primary analysis.
- Within each physical slide, the same frozen replicate ID partitions the 100 locations into
  five disjoint 20-location groups. Replicate means estimate patch-sampling variation; neither
  locations nor replicate groups are treated as independent slides.
- Radius benefit is the replicate mean of raw-minus-method six-scanner location centroid RMS.
  Positive values favor the correction. Fit one model per PFM × method (20 cells).
- Content change is the replicate mean corrected-minus-raw matched-content margin, separately
  for each non-AT2 source scanner. Positive values favor the correction. Fit one model per
  PFM × method × source scanner (100 cells).

For each cell fit the diagonal random-slope form

```text
contrast[tissue, slide, replicate]
  = fixed cell mean
  + tissue-specific cell slope
  + slide-within-tissue cell slope
  + 20-location replicate sampling error.
```

This is profiled REML using the same nested block implementation as E2. The tissue-variance
test compares the full model with a zero-tissue-variance boundary model and uses the
`0.5 chi-square(0) + 0.5 chi-square(1)` likelihood-ratio reference.

## Multiplicity, shrinkage and sensitivity

- Apply BH independently to the 20 radius tissue-variance tests and the 100 content
  tissue-variance tests. Report effect sizes and variance components regardless of q-value.
- Report tissue BLUPs, conditional SE, raw tissue contrast and shrinkage amount for every
  cell. Tissue class size is always shown (observed range 1--6 slides).
- Repeat every fit after excluding tissue types with fewer than three physical slides. This
  removes one one-slide and five two-slide tissues; it is a sensitivity analysis and cannot
  replace the full 37-tissue primary result.
- A tissue-heterogeneity claim requires BH q < 0.05 in the corresponding prespecified family.
  Directional tissue BLUPs without this gate are descriptive.

## Tissue-specific winner diagnostic

- For each PFM, define the global safe-and-improved winner from the frozen E5 result; no
  method is reselected after E6. The observed global winner is Orthogonal Procrustes for all
  four PFMs.
- Within each tissue, rank only methods that were globally safe for that PFM by the predicted
  radius benefit `fixed mean + tissue BLUP`. Report the fraction and identity of tissues in
  which the global winner remains first. One- and two-slide tissues remain visible and the
  minimum-three-slide sensitivity is reported separately.
- This diagnostic describes correction-response heterogeneity. It is not an independent
  biological performance endpoint.

## Secondary leave-one-tissue-out transfer

The approved 37-fold leave-one-tissue-type-out (LOTO) analysis is a separate materialized
population. All slides from the held-out tissue and all six scans leave the fit together.
Image and feature sufficient statistics use exact `all slides - held-out tissue` subtraction.
The same input-only CORAL shrinkage and frequency gain cap are reused; no outcome tuning is
allowed. LOTO results are reported as transfer sensitivity and do not replace LOSO primary
inference.

## Gates

- Expected primary replicate tables: 10,900 radius rows and 54,500 content rows.
- All 109 slides, 37 tissue types, five replicate IDs, four PFMs and five primary methods must
  be present; content also requires all five source scanners.
- E6 fitting cannot start until identity/count/finiteness checks and the E5 result lock pass.
- Macenko remains Supplement-only and is not included in primary BH families or winner claims.

