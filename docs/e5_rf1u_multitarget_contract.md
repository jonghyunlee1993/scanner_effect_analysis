# E5-RF1U unpaired multi-target execution contract

**Status:** FROZEN BEFORE RF1U PFM ACCESS
**Frozen:** 2026-08-04, before any RF1U embedding, scanner-radius or content endpoint exists
**Precedes:** [`e5_rf1m_combined_candidate_results.md`](e5_rf1m_combined_candidate_results.md),
[`e5_rf1m_slide_adaptive_feasibility.md`](e5_rf1m_slide_adaptive_feasibility.md)

RF1U is a post-core method-development condition. It does not reopen the frozen E4--E7 decision
record, does not change the five-method E5 primary ranking, does not replace the result-locked
RF1, and does not alter any locked artifact. RF1M remains closed and blocked.

## 1. Why this method

RF1M failed its per-fold no-harm gate and its selection machinery depended on paired evaluation,
which is unavailable at deployment. Three diagnostics established the design:

- the band gain is a ratio of pooled population energies and never forms a per-pair ratio, so it
  is structurally unpaired in the same sense as Reinhard;
- pairing removes the tissue-composition confound worth 0.06--0.16 in log gain, which is 10--24%
  of the correction for GT450 and S60 but 67--72% for S360;
- cutting the populations along tissue type rather than at random costs only 1.07--1.33x, so
  organ composition is not the dominant error when the population is diverse.

A reliability guardrail is therefore available without pairing: the slide bootstrap standard
error of the gain itself, which resolves per band.

## 2. Frozen transform

The condition is `reinhard_unpaired_band` (RF1U), defined for a target scanner `T`.

### Step 1 — Reinhard toward T

For every non-`T` scanner, map its pooled CIE L\*a\*b\* mean and SD on the training slides to the
pooled training mean and SD of `T`, with SD floor `1e-3`, exactly as the locked E5 Reinhard does
for AT2. `T`'s own images are never rendered or re-encoded; their audited raw embeddings are
copied into the condition.

### Step 2 — population band energies

Decompose the channel mean of the OD image into the three frozen additive Laplacian bands at
`sigma = (1, 2, 4)` pixels with the residual low-pass base preserved. On the training slides
accumulate, per band, the mean square of each band over pixels, summed over patches, for the raw
`T` patches and for each post-Reinhard source scanner.

### Step 3 — unpaired gain and its uncertainty

For source `s` and band `k`,

```text
log g_raw(s,k) = 0.5 * [ log mean E_T(k) - log mean E_s(k) ]
```

Both means are population means over training slides; no per-slide or per-patch pairing is used.
The uncertainty is the standard deviation of `log g_raw` over 2,000 training-slide bootstrap
replicates with seed `20260803`, resampling slides with replacement and recomputing both pooled
means on the resampled slides.

### Step 4 — reliability shrinkage

```text
alpha(s,k) = max( 0, 1 - 2 * SE(s,k) / |log g_raw(s,k)| )
log g(s,k) = alpha(s,k) * log g_raw(s,k)
```

The multiplier is fixed at **two** standard errors, the conventional value, before any PFM
endpoint is computed. **No other multiplier is tried after seeing a PFM result.** `alpha = 0`
yields exact identity, so the condition degrades to plain Reinhard toward `T`. A hard symmetric
cap of 4.0 on `g` is retained as a numerical guard only and is not expected to bind.

### Step 5 — shared-OD residual and exact gamut projection

Unchanged from RF1 and RF1M. Reconstruct the corrected mean OD as `base + sum_k g(s,k) * b_k`,
take the scalar residual against the original mean OD, add the same scalar to all three OD
channels, and project it into `[-min(o), log(256) - max(o)]`. A final `[0,1]` clamp is a
floating-point safeguard only.

## 3. Population, folds and targets

- Same 109 physical slides, six scanners, 100 locations and native-AA grid as E5.
- Same five outcome-blind physical-slide folds from `SHA256("e5_rf1_fold_v1:" + slide_id)`.
  Statistics for a held-out fold are fit only on the other four.
- Model-native FOVs 224, 256 and 512 px; statistics fit separately per FOV.
- Target scanners are **AT2, GT450 and S60**, frozen before execution. AT2 is the reference used
  throughout the study; GT450 has the highest audited high-band transfer at 1.478 fold and S60
  the transfer closest to AT2 at 1.044, so the three span the audited spectrum.
- For each target, the five non-target scanners are sources. The target contributes its raw
  embeddings unchanged.

## 4. Conditions and comparison

Six rendered conditions, two per target:

| Condition | Meaning |
|---|---|
| `reinhard_T` | Reinhard toward `T` alone, the comparator |
| `rf1u_T` | Reinhard toward `T` plus the shrunk band correction |

The raw condition is the already audited `outputs/e0_pfm_features` population and is shared.

`reinhard_at2` is **not** expected to reproduce the locked E5 Reinhard values, and the earlier
draft of this clause was wrong to require it. The locked E5 comparators are cross-fitted with
109-fold leave-one-slide-out, fitting on 108 slides, whereas RF1U inherits the RF1 five-fold
scheme and fits on about 87. Different training sets give slightly different pooled Lab
statistics and therefore slightly different output. Measured over five slides of ResNet50, the
two agree to a minimum cosine of 0.9972 and a maximum absolute feature difference of 0.045,
consistent with that fold difference and far from a defect.

This is why the comparator is re-rendered here rather than borrowed: RF1U and its Reinhard
baseline are fitted on the same folds and encoded in the same pass, so the primary comparison is
matched. Absolute RF1U numbers are therefore compared with the locked E5 table only as context,
never as a like-for-like substitution.

## 5. Endpoints

All metric definitions, thresholds, the bootstrap seed `20260803` and 5,000 replicates are
reused unchanged from the frozen E4--E7 decision record, with one explicit generalization: the
content anchor is the raw acquisition of that condition's **own target** `T`, not always AT2. A
condition mapping toward GT450 is judged on agreement with raw GT450. For `T = AT2` this reduces
exactly to the locked definition.

- **Invariance:** scanner centroid RMS over the six scanners, and relative reduction from raw.
  This statistic is target-independent by construction.
- **Content:** per source scanner, mean cosine to the raw `T` embedding at the same location
  minus the 95th percentile of raw `T` unmatched self-similarity. Non-inferiority margin `-0.02`.
- **Collapse:** variance trace, entropy effective rank and median pairwise distance, each as a
  ratio to raw, for every source scanner. Point threshold `0.90` and CI lower threshold `0.85`.
- **Primary comparison:** paired per-slide difference in scanner centroid RMS between `rf1u_T`
  and `reinhard_T`, with a 5,000-replicate slide bootstrap CI, reported for each of the four
  frozen PFMs and each of the three targets.

A `(target, PFM)` cell is called **safe and improved over Reinhard** only if RF1U passes content
non-inferiority, passes the collapse gate for every source scanner, and its paired radius
difference CI excludes zero in its favour.

## 5a. Amendment 1 — runtime equivalence in place of a version pin (2026-08-04)

**Status:** dated pre-outcome amendment, written and frozen before any RF1U embedding existed.

The environment had drifted from the frozen PFM contract: `timm` 0.9.8 → 1.0.12,
`huggingface_hub` 0.29.1 → 0.36.2, and the `conch` package was absent, so `conch_v1` could not
be constructed at all. `geopandas` was also missing, which prevented TRIDENT from importing.
`verify_runtime_contract` correctly refused to extract.

`conch` 0.1.0 and `geopandas` were installed. A dry run confirmed neither install changes an
existing package; `torch` 2.5.1 and `torchvision` 0.20.1 still match the pin, and `timm` and
`huggingface_hub` remain at the newer versions.

The pin exists so that new features stay comparable with the locked raw population. A version
match proves that; a direct measurement proves it more strongly. The audit in
[`audit_pfm_runtime_drift.py`](../src/audit_pfm_runtime_drift.py) re-encodes audited crops under
the installed runtime and compares against the stored raw features. Over 4 encoders × 3 slides,
7,200 embeddings, **every cell was bit identical**, maximum absolute difference exactly
`0.000e+00` and relative L2 exactly zero.

RF1U extraction therefore accepts the runtime when **either** the pinned versions match **or**
the drift audit covers that encoder with a maximum absolute difference of exactly zero, under
the same observed and contract runtimes recorded in the audit. Every shard records which basis
was used and the audit hash. The locked raw population remains the comparison baseline and is
not re-extracted.

This amendment changes no endpoint, threshold, transform parameter or target. It does not modify
the locked PFM contract, whose pin continues to govern the E0--E7 artifacts.

## 6. Boundaries

- RF1U is developed after the primary E5 outcomes were known and cannot enter the frozen
  five-method primary ranking.
- Better image spectrum or lower scanner radius is not evidence of biological fidelity.
- Comparing targets tests whether AT2 is special as a canonical destination. It does not
  establish any scanner as clinically preferable.
- The shrinkage multiplier, band sigmas, target list, fold assignment and endpoint definitions
  are fixed by this document. If a result is unfavourable, it is reported as such.
