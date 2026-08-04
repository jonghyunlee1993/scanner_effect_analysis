# RF1M slide-adaptive gain: feasibility test

**Status:** IMAGE-ONLY FEASIBILITY TEST — no new method, no contract, no PFM access
**Updated:** 2026-08-03
**Follows:** [`e5_rf1m_combined_candidate_results.md`](e5_rf1m_combined_candidate_results.md)

## Question

RF1M fits one gain per scanner and band from pooled training energy, but the required
correction varies by slide. This tests whether a slide's own post-Reinhard band energy predicts
the energy its correction must reach, using only information available at deployment.

Both the current estimator and the proposed one are special cases of

```text
log E_target(i) = a + b · log E_source(i)
```

with `b = 0` reproducing RF1M's pooled ratio and `b = 1` reproducing naive per-slide
normalization. Applying a gain multiplies a band's energy by `g²`, so each estimator's held-out
error is the error of its predicted target log-energy. Validation is exact five-fold
cross-fitting on the frozen RF1 physical-slide folds, with 5,000 slide bootstrap replicates,
seed 20260803, over 45 cells (3 FOV × 5 scanners × 3 bands).

## Result

The fitted slope is decisively intermediate: mean 0.626, range 0.470--0.743, with 40/45 cells
above 0.5. Neither endpoint is correct. Naive per-slide normalization (`b = 1`) is **44% worse**
than the current pooled estimator, because it overcorrects; the regression is 36.5% better on
average.

| Scanner | mean CV slope | regression vs pooled held-out RMSE | cells better by CI |
|---|---:|---:|---:|
| AKOYA | 0.556 | −56.8% | 9/9 |
| S60 | 0.664 | −53.2% | 9/9 |
| GT450 | 0.666 | −51.9% | 9/9 |
| VERSA | 0.629 | −12.4% | 0/9 |
| S360 | 0.617 | −8.1% | 0/9 |

## Interpretation

Slide-adaptive gain estimation works, and it works well, for the three scanners that RF1M
already handled: held-out prediction error falls by more than half, confirmed in every cell by a
slide-blocked bootstrap CI. It does **not** work for VERSA and S360, the two scanners whose
correction RF1M could not make reliably safe.

The contrast with the oracle decomposition is the informative part. Projecting each slide's
required correction curve onto its scanner mean showed 25.2% of S360's correction energy lay in
slide-specific strength. That projection uses the paired AT2 target. This test uses only the
source image, and recovers almost none of it. For S360 the slide-level variation is real but not
inferable from the source acquisition; for AKOYA, GT450 and S60 the source carries it.

Consequently this estimator does not unblock RF1M. The gate failed on S360, and S360 is
precisely where the improvement is absent. A method built on slide-adaptive gains would still
have to route S360 and VERSA to identity through the no-harm selector, which is what RF1M
already did in four of five folds.

## Status of the idea

The finding is retained as an image-domain method result. It is not promoted to a method for the
current manuscript: the current paper's correction remains the result-locked RF1, and RF1M
remains blocked. The slide-adaptive estimator is a candidate starting point for a later study,
where it would be specified together with a slide-blocked CI no-harm criterion rather than the
zero-tolerance point criterion used in the RF1M contract.

## Artifacts

- [`summary.json`](../outputs/rf1m_slide_adaptive/result/summary.json)
- [`slide_adaptive_comparison.csv`](../outputs/rf1m_slide_adaptive/result/slide_adaptive_comparison.csv)
- per-slide band energies under `outputs/rf1m_slide_adaptive/energy/`, which also provide the
  slide-level resolution a CI-based no-harm criterion would require
- SLURM: `16185351` (3 energy tasks), `16185352` (analysis), all `COMPLETED 0:0`
