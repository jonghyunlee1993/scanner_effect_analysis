# E5-RF1M combined-candidate image-only results

**Status:** IMAGE-ONLY OUTCOME — **PFM access blocked by the frozen gate**
**Updated:** 2026-08-03
**Contract:** [`e5_rf1m_combined_candidate_contract.md`](e5_rf1m_combined_candidate_contract.md)

RF1M did not pass the per-fold no-harm gate of the contract §5. Under the boundary clause of
Amendment 1 no further shrinkage multiplier is tried, no scanner-specific carve-out is made, the
gate is not relaxed, and no RF1M embedding, scanner-radius or tissue endpoint was computed. The
result-locked RF1 remains the manuscript method. Nothing in the PanNormal E0--E7 locks changes.

## What was executed

- 75 cells: 15 outer (fit on four folds, evaluate the held-out fold) and 60 inner directional
  audits (fit on three folds, evaluate one of the two excluded folds), across FOV 224/256/512.
- Every cell rendered all 14 candidates, identity included, on 21--22 held-out slides × 5 source
  scanners × 100 locations.
- Band energy was accumulated natively on training patches in the mean-OD domain, replacing the
  pilot's approximation from locked 72-bin radial statistics.
- SLURM provenance: `16132140` (45 cell tasks, all `COMPLETED 0:0`), `16171428` (selection).

## Gate outcome

| Contract §5 gate | Required | Observed | Verdict |
|---|---|---|---|
| cell outputs with valid identity/hash | 45/45 | 45/45 | pass |
| 109 unique slides per FOV | 3/3 | 3/3 | pass |
| outer fold × scanner × FOV spectrum non-worse | 75/75 | **72/75** | **fail** |
| outer fold × scanner × FOV gamut | 75/75 | 75/75 | pass |
| aggregate scanner × FOV spectrum and gamut | 15/15 | 15/15 | pass |
| maximum final clamp MAE | ≤ 1e-6 | 0.0 | pass |

The three failures are S360 outer fold 4 at all three FOVs, the 21-slide fold.

## Image-domain results that do hold

Paired-AT2 log-spectrum RMSE, pooled over the five outer folds, relative to Reinhard.

| Scanner | Selected caps | RF1M | Locked RF1 (cap 1.25) |
|---|---|---:|---:|
| S60 | 2/2/2/2/2 | −84.1% | −20.6% |
| GT450 | 2/2/3/2/2 | −73.9% | −22.4% |
| VERSA | 1.1 in all folds | −41.3% | −1.4% |
| AKOYA | 2/2/1.5/1.5/1.5 | −39.8% | −21.0% |
| S360 | 1.01/I/1.01/1.01/1.04 | −6.0 to −8.0% | **+29.6%** |

Values are FOV 256 except the S360 range, which spans the three FOVs. Mean over the 15 cells is
**−49.3%** against **−7.2%** for the locked RF1, and RF1M has the lower pooled RMSE in 15/15
cells. The three FOVs agree closely, so the effect is not FOV-specific.

Two pre-stated expectations were wrong in opposite directions. The contract §1 named AKOYA the
known limitation and predicted no improvement over the locked radial RF1; per-scanner caps of
1.5--2 instead improved it from −21.0% to −39.8%. Conversely S360, whose pooled sign RF1M does
repair, is the scanner that blocked the gate.

Gamut behaviour is unchanged from RF1 and remains clean. Maximum final clamp MAE is exactly zero
in every cell and candidate. AKOYA is the only scanner whose projection is materially active:
mean materially out-of-range fraction 0.0029 against the 0.005 limit and q99 0.025 against the
0.05 limit, the thinnest margin in the panel. VERSA, S60 and S360 are at or near zero.

The explicitly rendered identity candidate agreed with the uncorrected Reinhard base to
2.78e-08 in log-spectrum RMSE, confirming that the Laplacian decomposition reconstructs the
mean-OD image exactly.

## Why the gate failed

For S360 outer fold 4, all four inner folds rated the selected candidate non-worse than
identity, and the held-out fold then disagreed. Under the original one-standard-error rule the
selector returned `cap_1.1`, whose mean inner delta was −0.0818 with SE 0.0219; `cap_1.05` at
−0.0574 missed the −0.0599 threshold by 0.0025, so no shrinkage occurred. Amendment 1 raised the
multiplier to two standard errors, which moved the selection to `cap_1.03`--`cap_1.04` and cut
the held-out damage from 33.7--37.0% to **2.3--6.6%**, a five- to eight-fold reduction. It did
not reach non-worse.

This is not a selector defect and not a scanner-specific coding error. The direct evidence is
the ratio of estimation noise to available benefit, measured within RF1M itself. Averaging the
selected candidate's inner standard error and mean inner gain over the 15 outer-fold selections
of each scanner, both expressed as fractions of that scanner's Reinhard base RMSE:

| Scanner | base RMSE | gain / base | SE / base | **SE / gain** |
|---|---:|---:|---:|---:|
| S60 | 1.127 | 0.830 | 0.015 | 0.018 |
| GT450 | 1.521 | 0.744 | 0.015 | 0.021 |
| AKOYA | 1.334 | 0.411 | 0.023 | 0.056 |
| VERSA | 0.323 | 0.479 | 0.086 | 0.177 |
| S360 | 0.268 | 0.111 | 0.031 | **0.310** |

S360's estimation noise is not the largest in the panel; VERSA's is nearly three times larger in
absolute terms, and VERSA passed. What distinguishes S360 is that only 11% of its post-Reinhard
error is correctable at all, against 74--83% for GT450 and S60, so the same estimation noise
consumes a third of the available benefit instead of a fiftieth. A scanner already close to the
AT2 reference offers little to gain and is therefore the easiest to make worse.

This is consistent with, and more direct than, the locked E2 decomposition, in which S360 has
the smallest tissue fraction of between-slide scanner-slope variance at 0.033, BH q = 0.4607.

A robustness filter on the candidate's own inner variance would not have helped, because the
candidate's benefit was consistent across all four inner folds; only reducing the applied
strength bounds the damage, and reducing it far enough to guarantee no-harm on S360 also removes
the correction that made RF1M valuable.

The absolute magnitudes should be stated alongside the verdict. The blocking harm is a log-RMSE
increase of 0.0044--0.0128, against gains of 1.114 and 0.934 for GT450 and S60 in the same
units, roughly one part in a hundred to one in two hundred fifty. The gate is a per-cell
relative no-harm criterion and does not distinguish a small worsening from a large one. That is
the frozen criterion and it is not relaxed here, but the failure is correctly described as an
unguaranteed no-harm on one already well-matched scanner, not as a large regression.

## Interpretation

The finding is that a large mean image-domain gain and a per-scanner no-harm guarantee are not
the same property, and that the second is the binding constraint. RF1M is about seven times
better than the locked RF1 on the pooled paired-AT2 spectrum and repairs the S360 sign that
motivated the whole improvement pilot, yet it cannot promise that a new slide from the most
slide-heterogeneous scanner will not be made worse.

Blocking RF1M is therefore not a disappointing outcome of the protocol; it is the protocol
working on its own method development. The same logic the study applies to scanner-radius
reduction — that an aggregate improvement does not license a claim unless a prespecified
guardrail also holds — applied here to an image-domain gain that would otherwise have looked
decisive.

## What is not claimed

- No statement is made about RF1M's effect on any PFM representation, scanner radius, content
  fidelity, collapse metric or tissue endpoint. None was computed.
- Better paired spectrum is an image-domain result and is not evidence of representation or
  biological fidelity.
- RF1M is not entered into the frozen five-method E5 primary ranking, and the locked RF1 numbers
  are unchanged.
- The S360 failure is characterized as slide-level heterogeneity in this cohort. It is not
  established as a property of the S360 device.

## Locked artifacts

- [`summary.json`](../outputs/rf1m_candidate/selection/summary.json) with `pfm_access_gate_pass: false`
- [`scanner_fov_summary.csv`](../outputs/rf1m_candidate/selection/scanner_fov_summary.csv)
- [`outer_fold_evaluation.csv`](../outputs/rf1m_candidate/selection/outer_fold_evaluation.csv)
- [`candidate_selection.csv`](../outputs/rf1m_candidate/selection/candidate_selection.csv)
- [`rf1m_strict_nested_spectrum.png`](../outputs/rf1m_candidate/selection/rf1m_strict_nested_spectrum.png)
- 75 cell outputs and summaries under `outputs/rf1m_candidate/cells/`
