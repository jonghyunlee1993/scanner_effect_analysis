# E8 paired residual results — the pre-registered ceiling test

**Status:** post-core extension; the pre-registered reading of §10 has been executed and
returned a verdict. The frozen E0--E7 locks and the E5 five-method primary ranking are
unchanged.
**Contract:** [`e8_paired_residual_contract.md`](e8_paired_residual_contract.md), §10
**Verdict artifact:** `outputs/e8_residual/reading/gt450/summary.json`
**Target:** GT450, the destination RF1U selected. Five sources: AT2, VERSA, AKOYA, S60, S360.

This is the experiment the report's ceiling claim was pre-registered against. Contract §10 was
written before any embedding existed and fixed four admissible readings; no threshold in it was
revised after an endpoint was seen.

## 1. Verdict

> **supported** — arm B cleared content non-inferiority and the collapse gate in all four PFMs
> and still left the linear scanner probe above 0.50 in all four, while the feature-space
> methods reached at most 0.17.

Contract §10 required "above 0.50 in at least 3 of 4 PFMs" for support and "below 0.30 in at
least 2 PFMs" for falsification. The observed floor is **0.759**; the falsification band was
not approached in any model.

## 2. What the two arms are, and which one ran to an endpoint

| Arm | Parameterization | Image-only gate | Encoding |
|---|---|---|---|
| A `gainfield` | spatially varying band gain on top of RF1U; guarantees retained | **failed**, 3 of 5 folds | blocked |
| B `free` | unconstrained OD residual — the ceiling probe | passed, 5 of 5 folds | unblocked |

The gate is image-only and was evaluated with no access to PFM features or any outcome
(`pfm_feature_access: false`, `outcome_access: false`).

Arm A failed on folds 3 and 4 for one reason, and it is worth being exact about which. The
`paired_improved` leg is a conjunction: pooled paired MAE must beat the RF1U baseline **and** at
least 4 of 5 source scanners must improve individually (`E8_GATE_MIN_SOURCES = 4.0`,
`src/train_e8_residual.py`). On both failing folds the pooled MAE did improve — 0.05913 vs
0.06015 and 0.05033 vs 0.05117 — and **only the per-source leg failed**, at 3 of 5 sources where
the three passing folds had 4. Band RMSE and gamut were non-inferior on all five folds. The arm
gate then requires 4 of 5 folds (`E8_GATE_MIN_FOLDS = 4`); arm A had 3.

So arm A was rejected for being unevenly good across source scanners, not for being worse on
average. That is the per-source leg doing exactly the job it was added for. Under §10 arm A's
outcome is independent of the ceiling reading: **the analytic three-number gain is confirmed as
sufficient within its family**, and the scanner × tissue interaction is not established as worth
modelling.

Arm B passed every fold with all 5 sources improved and paired MAE 0.0464--0.0552 against a
baseline of 0.0512--0.0602.

## 3. The frontier — arm B is not a weak baseline

Relative scanner-radius reduction from raw, 109 slides, GT450 target. Raw radius is 0.2119
(ResNet50), 0.4827 (UNI v1), 0.2625 (CONCH v1), 0.3513 (Virchow2).

| Condition | ResNet50 | UNI v1 | CONCH v1 | Virchow2 |
|---|---:|---:|---:|---:|
| Reinhard | +28.53% | +4.04% | +14.52% | −0.08% |
| RF1U | +32.85% | +6.86% | +19.56% | +4.04% |
| **arm B, learned free residual** | **+47.87%** | +12.13% | **+25.93%** | +3.48% |
| CORAL, feature space | +34.75% | +15.66% | +23.67% | +16.38% |
| Orthogonal Procrustes, feature space | +42.44% | +23.22% | +24.44% | +19.91% |

Arm B is safe and invariance-improved in **4 of 4** PFMs. In ResNet50 it is the largest reduction
anywhere in the panel, feature space included, and in CONCH v1 it beats both feature methods.
This matters for how the ceiling result reads: it was not produced by a weak model.

Its content margins are the closest call in the grid. `Δ content CI lower` is +0.0439, +0.0190
and +0.0284 for ResNet50, UNI v1 and CONCH v1, but **−0.0165 for Virchow2** — inside the frozen
−0.02 margin, and the only negative value among arm B's four cells.

## 4. The probe, which is what §10 was about

Balanced accuracy, six-way scanner identification, chance 0.167, frozen RF1 physical-slide
folds. This is a secondary invariance-only endpoint and is degenerate read alone; §3's content
and collapse verdicts are the fidelity decision.

| Condition | ResNet50 | UNI v1 | CONCH v1 | Virchow2 |
|---|---:|---:|---:|---:|
| raw | 0.970 | 0.997 | 0.940 | 0.996 |
| Reinhard | 0.902 | 0.990 | 0.897 | 0.992 |
| RF1U | 0.890 | 0.991 | 0.885 | 0.991 |
| **arm B, learned free residual** | **0.759** | **0.942** | **0.766** | **0.967** |
| CORAL | 0.102 | 0.082 | 0.099 | 0.088 |
| Orthogonal Procrustes | 0.162 | 0.133 | 0.148 | 0.117 |

Arm B moves the probe further than any other image-space condition — 0.13 to 0.21 in ResNet50
and CONCH v1 — and lands nowhere near feature space. A learned, pixel-registered, unconstrained
residual with full paired supervision is still, on this endpoint, an image correction.

**Reading.** Read beside the E4 controls and the E8 paired oracle, the ceiling is now bounded
from three directions. Destroying the high band outright leaves 0.808--0.950. An oracle
substituting another scanner's true registered detail leaves 0.928--0.992. A learned model that
beats every analytic method on radius leaves 0.759--0.967. The limit is a property of the
domain, not of the correction family, and not of how hard the correction was optimized.

## 5. The nonlinear probe splits the two feature methods

The same conditions under a 256-unit MLP and a cosine k-NN vote, both on the same folds.

| Condition | probe | ResNet50 | UNI v1 | CONCH v1 | Virchow2 |
|---|---|---:|---:|---:|---:|
| CORAL | MLP | 0.269 | 0.074 | 0.184 | 0.161 |
| **Orthogonal Procrustes** | **MLP** | **0.689** | **0.628** | **0.614** | **0.778** |
| CORAL | k-NN | 0.276 | 0.215 | 0.244 | 0.203 |
| Orthogonal Procrustes | k-NN | 0.337 | 0.295 | 0.266 | 0.282 |

This was not anticipated and it qualifies the report's closing recommendation.

Orthogonal Procrustes is a rigid rotation fitted per source scanner. It preserves every
within-scanner distance exactly, so it can align the clouds but cannot change their shape; a
nonlinear decoder still sees the untouched local geometry and recovers the scanner at
0.61--0.78. CORAL whitens the source second moment and recolours it with the target's, which
alters that shape, and it holds the nonlinear probe at 0.07--0.27.

So the method with the **largest** radius reduction is the one that leaves the **most** scanner
information recoverable. This is the study's own thesis appearing a fourth time, now inside
feature space and without any content being destroyed: an invariance metric and actual scanner
removal dissociate, and the linear probe alone would have hidden it.

The deployable conclusion narrows accordingly. "Feature space is where the headroom is" holds
for **CORAL specifically**, not for feature-space correction generically. Procrustes remains the
larger radius reduction and the better tissue-profile geometry (E7), but it is not the stronger
scanner removal once the probe is allowed to be nonlinear.

## 6. Hallucination audit

Invented budget 1e-3, gradient threshold 0.0411, 25 locations per slide.

| | arm A `gainfield` | arm B `free` | RF1U baseline |
|---|---:|---:|---:|
| invented fraction, mean | 5.46e-05 | 6.40e-04 | — |
| invented fraction, max | 0.0050 | 0.1285 | — |
| phase correlation, mean | 0.882 | **0.694** | 0.946 |
| phase correlation, min | 0.441 | **0.006** | 0.654 |
| re-extracted sign change, mean | 0.0349 | 0.0954 | 0.0171 |
| projection fraction, mean | 0.0568 | 0.0224 | — |

Both arms are within the invented-fraction budget and both hold their exact structural
properties. Arm A additionally satisfies its parameterization guarantees exactly: zero scaled
band violations, residual exactness 4.6e-06 mean, gains bounded in [0.25, 4.0].

Arm B's phase agreement is materially lower than RF1U's — 0.694 against 0.946 on average, with
a minimum near zero — and its re-extracted gradient sign changes are 5.6× the baseline. **This
belongs beside the verdict.** §10 required only that arm B pass content non-inferiority and the
collapse gate, which it did in 4 of 4; but the model that supports the ceiling claim is
measurably less phase-faithful than the analytic method it outperforms. The honest statement is
that the ceiling survives even a correction permitted to be structurally aggressive, not that a
structurally conservative learned model was tested and failed.

## 7. What this does and does not license

**Does.** The report's §12 ceiling wording stands, and now rests on a pre-registered learned
baseline rather than on the analytic family alone. Contract §11's exclusion of CycleGAN also
stands on firmer ground: the paired arm dominates the unpaired one for this question and it
already reaches the ceiling.

**Does not.** No claim about biological or clinical fidelity; this cohort carries no molecular
or downstream endpoint. The result is for one destination (GT450) in one 109-slide cohort. Arm
A's failure is a failure of this parameterization under this image-only gate, not evidence that
no spatially varying correction can help. And arm B is a ceiling probe by construction — it is
not proposed as a deployable method.

## 8. Artifacts

- `outputs/e8_residual/reading/gt450/summary.json` — the §10 verdict
- `outputs/e8_residual/audit/gt450/summary.json` — per-fold gates, hallucination audit
- `outputs/e8_residual/frontier/gt450/{endpoint_summary,paired_comparisons,collapse_detail}.csv`
- `outputs/e8_residual/probe/{resnet50,uni_v1,conch_v1,virchow2}.{csv,confusion.csv,summary.json}`
- `outputs/e8_residual/runs/gt450/{gainfield,free}/fold*/` — checkpoints
