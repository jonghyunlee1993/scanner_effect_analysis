# E7 grouped tissue-probe execution contract

**Status:** FROZEN before E7 tissue-retrieval outcome access on 2026-08-03  
**Parent decision:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md), Decisions 5--7  
**Inputs:** locked raw/E5 LOSO embeddings and the independently audited E6 exact-LOTO
population

## Scope and interpretation

Tissue type is the only available biological grouping. This analysis is a coarse secondary
representation diagnostic, not biological non-inferiority, morphology validation, clinical
validation or evidence that all within-tissue information was preserved. No nucleus or spatial
endpoint is introduced.

## Frozen grouped probe

- Fit a nearest-centroid tissue probe independently for each PFM and held-out physical slide.
- The reference bank uses only L2-normalized raw AT2 embeddings from other physical slides;
  all 100 locations from the held-out slide and all six scans leave the probe fit together.
- A tissue class is evaluable only when at least one other physical slide remains after the
  held-out slide is removed. The single one-slide tissue is therefore excluded from inference,
  leaving 108 slides and 36 tissue types. It is retained in the class-size table as explicitly
  unevaluable.
- To keep the competing class universe identical across queries, the singleton tissue is also
  excluded as a distractor centroid. Each tissue centroid is the renormalized mean of all raw
  AT2 training-location unit embeddings in that tissue.
- Query the five non-AT2 scanners under raw and all five actual comparator conditions. Evaluate
  both the exact E5 LOSO population (primary) and exact E6 LOTO correction population (transfer
  sensitivity). The AT2 held-out baseline is reported once and is not duplicated by method.

## Metrics

For each held-out slide, scanner and condition, average the 100 location-level values:

1. correct-tissue top-1 accuracy;
2. correct-tissue top-5 accuracy;
3. correct-tissue cosine score minus the largest competing-tissue score;
4. Pearson correlation between the query and matched raw-AT2 vectors of similarities to all
   36 training tissue centroids (cross-scanner tissue-neighborhood agreement).

Aggregate the five source scanners within slide before population inference. Report every
source scanner separately as a diagnostic. Each tissue receives equal weight in the point
estimate. Confidence intervals use 5,000 paired hierarchical bootstrap replicates with seed
`20260804`: sample tissue types with replacement and then physical slides within each sampled
tissue. Correction-minus-raw contrasts preserve the within-slide pairing. No patch is an
independent uncertainty unit.

## Class-size sensitivity and reporting

- Repeat all summaries after excluding tissue types with fewer than three slides (98 slides,
  31 tissues).
- Report class size beside tissue-level results and report both full-evaluable and minimum-three
  estimates.
- There is no tissue-performance accept/reject threshold and no multiplicity-based tissue claim.
  Effect estimates and confidence intervals are descriptive secondary evidence.
- A correction may not be called fidelity-safe from this probe alone. E5 matched-content and
  every-scanner collapse gates remain authoritative.

## Gates

- Expected slide-level source rows per correction population: 12,960
  (`4 PFMs × 6 conditions × 5 scanners × 108 slides`).
- Expected held-out AT2 baseline rows: 432 (`4 PFMs × 108 slides`).
- Every query must use a centroid bank that excludes the held-out physical slide and contains
  exactly 36 evaluable tissue classes.
- Scanner, location, slide, tissue, correction-population, model and frozen-contract identities
  must pass before any endpoint is summarized.
