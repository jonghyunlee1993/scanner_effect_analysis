# E5-RF1M combined-candidate execution contract

**Status:** FROZEN BEFORE RF1M PFM ACCESS
**Frozen:** 2026-08-03, after the four image-only improvement branches completed and before any
RF1M embedding, scanner-radius endpoint or tissue endpoint is computed
**Parent contract:** [`e5_reinhard_residual_frequency_contract.md`](e5_reinhard_residual_frequency_contract.md)
**Selection pilot:** [`e5_rf1_improvement_pilot_contract.md`](e5_rf1_improvement_pilot_contract.md)
**Rationale:** [`reinhard_frequency_extension.md`](reinhard_frequency_extension.md)

RF1M is a post-core method-development extension. It does not reopen the frozen E4--E7 decision
record, does not change the five-method E5 primary ranking, does not replace the result-locked
RF1 condition, and does not alter any locked artifact.

## 1. Why this candidate

The improvement pilot completed four image-only branches on the frozen 109-slide grid. Mean
relative change of paired-AT2 log-spectrum RMSE against Reinhard, over 15 FOV × source-scanner
cells:

| Branch | Mean change | Improved cells | S360 |
|---|---:|---:|---:|
| locked radial RF1, cap 1.25 | −7.2% | 12/15 | +29.6% |
| 1. strict nested no-harm caps | −18.0% | 15/15 non-worse | 0% (identity) |
| 2. paired robust estimator | −15.9% | 12/15 | +9.2% |
| 3. multiscale Laplacian bands | −32.5% | 15/15 | −31.7% |

Branches 1 and 3 win on disjoint scanners: at FOV 256 the no-harm selector reaches −43.2% on
GT450 where multiscale reaches −27.2%, while multiscale reaches −46.1%, −37.6% and −31.3% on
VERSA, S60 and S360 where the no-harm selector reaches −5.1%, −20.5% and 0%. The two
mechanisms are orthogonal — one changes the transform parameterization, the other chooses a
scanner-specific correction strength with identity always available — so RF1M combines exactly
these two and nothing else.

Branch 2 is excluded: it is beaten by multiscale in four of five scanners and worsens AKOYA.
Branch 4 is excluded: it improves paired structure gradients but worsens AKOYA colour and
spectrum. Both remain reported image-domain findings, not components of RF1M.

AKOYA is the known limitation. No branch improved AKOYA beyond the locked radial RF1, and RF1M
is not expected to. This is stated in advance and is not a reason to re-select later.

## 2. Population, folds and cross-fitting

Unchanged from the RF1 contract.

- Same 109 physical slides, six scanners, 100 locations, native-WSI explicit-AA grid and four
  frozen PFMs.
- AT2 is scanner index zero, is never rendered or re-encoded, and its audited raw embedding is
  copied into the RF1M condition.
- Five balanced outcome-blind physical-slide folds from `SHA256("e5_rf1_fold_v1:" + slide_id)`,
  round-robin over the sorted order, giving 22, 22, 22, 22 and 21 slides. Tissue labels, images
  and all PFM endpoints are forbidden from fold construction.
- Model-native centered FOVs 224, 256 and 512 px. All statistics, band energies, gains and cap
  selections are fit separately per FOV and per source scanner.

## 3. Frozen RF1M transform

The single extension condition is `reinhard_multiscale_noharm`.

### Step 1 — training-fold Reinhard

Identical to RF1 step 1. Pool the fixed 8-pixel lattice over training slides, map each
scanner's CIE L\*a\*b\* mean and SD to the pooled training AT2 mean and SD with SD floor `1e-3`,
and apply that fixed transform to every training and held-out pixel. The base image is the same
`[0,1]`-clipped sRGB output used by the locked E5 Reinhard implementation.

### Step 2 — native Laplacian band decomposition

Let `m` be the channel mean of the OD image at 0.5052 µm/px. With frozen pixel scales
`sigma = (1, 2, 4)` and `sigma_0 = 0`, define successive Gaussian blurs `m_k` of `m` obtained by
reflect-padded separable convolution with incremental sigma `sqrt(sigma_k^2 - sigma_{k-1}^2)`,
truncated at four sigma. The three additive Laplacian bands and the residual base are

```text
b_k = m_{k-1} - m_k   for k = 1, 2, 3   (with m_0 = m)
base = m_3
m = base + b_1 + b_2 + b_3    (exact by construction)
```

### Step 3 — native band-energy accumulation

This replaces the pilot's approximation of band energy from locked 72-bin radial statistics.
Band energies are accumulated directly from training patches, in the same mean-OD domain in
which the correction is applied.

For every training patch, compute the per-band mean square `mean(b_k^2)` over pixels and
accumulate the sum over patches, separately for

- the post-Reinhard source patches of each non-AT2 scanner, giving `E_source(s, k)`; and
- the paired raw AT2 patches at the same physical locations, giving `E_target(k)`.

Both accumulations use the same FOV, the same pyramid and the same patch population. No tissue
label, held-out image, PFM feature or downstream endpoint enters this fit.

### Step 4 — capped band gains

For source scanner `s`, band `k` and symmetric cap `c`,

```text
g_s,k(c) = clip( sqrt( E_target(k) / E_source(s, k) ),  1/c,  c )
```

with a `1e-20` floor on both energies. `c = 1` gives exact identity in every band. The residual
low-pass base is never gained, so frequencies below the sigma-4 px cutoff (about
0.079 cycles/µm) are preserved by construction; RF1's explicit anchor normalization and
`[1,2,3,2,1]/9` log-gain smoothing do not apply and are not used.

### Step 5 — shared-OD residual and exact gamut projection

Identical in form to RF1 step 3, with the multiscale reconstruction replacing the Fourier one.
On a held-out Reinhard image with OD vector `o` and channel mean `m`,

```text
m'         = base + sum_k g_s,k * b_k
d          = m' - m                                   (proposed scalar residual)
d_projected = clip( d,  -min(o),  log(256) - max(o) )
```

Add the same projected scalar to all three OD channels, preserving the two within-pixel
chromatic OD differences. The result maps to RGB8 support under the frozen inverse
`RGB = (256 exp(-OD) - 1)/255`. A final `[0,1]` clamp is permitted only as a floating-point
safeguard. Proposed out-of-range channel fraction, materially out-of-range fraction, projection
fraction, projection-induced RGB MAE and final numerical clamp MAE are recorded per patch.

## 4. Strict nested cap selection

Candidate strengths are exact identity plus the frozen RF1 caps:

```text
1.00 (identity), 1.01, 1.02, 1.03, 1.04, 1.05, 1.10, 1.15, 1.20, 1.25, 1.50, 2.00, 3.00, 4.00
```

One cap is selected per FOV × source scanner × outer fold. Selection uses only images and never
a PFM endpoint.

- **Inner fits.** For outer fold `h` and inner validation fold `j != h`, the Reinhard statistics
  and band energies are fit on the three folds excluding both `h` and `j`, then all candidates
  are rendered on fold `j`. One unordered excluded pair `{a,b}` is fit once and produces both
  directions, so 3 FOV × 10 pairs = 30 GPU tasks yield 60 directional inner audits.
- **Eligibility.** A candidate is eligible only if, in every one of the four inner folds, it
  passes the frozen gamut gate and its paired-AT2 log-spectrum RMSE is non-worse than identity
  within tolerance `1e-12`. Exact identity must be eligible; if it is not, execution fails.
- **One-standard-error rule.** With fold-wise `delta = candidate RMSE - identity RMSE`, let
  `best` be the eligible candidate with the smallest mean delta,
  `SE = sd(delta[:, best], ddof=1)/sqrt(4)`, and select the **smallest-cap** eligible candidate
  whose mean delta is at most `mean delta[best] + SE`. Exact ties resolve toward identity.
- **Outer evaluation.** The selected cap is evaluated exactly once on outer fold `h`, using the
  transform and band gains fit on the four folds excluding `h`. This is the reported estimate.
  3 FOV × 5 outer folds = 15 further GPU tasks.

## 4a. Amendment 1 — two-standard-error shrinkage (2026-08-03)

**Status:** dated pre-outcome amendment required by §5. Written and frozen before re-running the
selection and before any PFM endpoint was opened. No RF1M embedding, scanner-radius or tissue
endpoint had been computed when this amendment was written, and none is computed by it.

### What was observed

The first execution of §4 produced 75 cells and passed every gate except per-fold no-worsening:
72/75 outer cells were spectrum non-worse. The three failures were S360 outer fold 4 at all
three FOVs, worsening by 33.7%, 34.5% and 37.0%. All 75 outer cells passed the gamut gate, all
15 aggregate cells passed both gates, and the maximum final clamp MAE was exactly zero.

### Diagnosis

For S360 outer fold 4, all four inner folds rated `cap_1.1` non-worse than identity, and its
mean inner delta was −0.0818 with `SE = 0.0219`. The one-standard-error threshold was −0.0599,
which `cap_1.05` missed by 0.0025, so the rule returned the strongest eligible candidate rather
than shrinking toward identity. The held-out fold then disagreed.

This is not a selector bug and not a scanner-specific defect. It is the regime the locked E2
decomposition already identified: S360 has the smallest tissue fraction of between-slide
scanner-slope variance, 0.033 at BH q = 0.4607, so its residual spectrum is the most
slide-idiosyncratic of the five scanners. A strength chosen on four folds is therefore least
transferable for S360.

A robustness filter on the candidate's own inner variance would **not** have prevented this:
`cap_1.1` was consistently better across all four inner folds, and even
`mean + 3 SE = −0.0161` remains below zero. Only reducing the applied correction strength
bounds the damage when the held-out fold disagrees.

### Amended rule

The shrinkage multiplier of §4 changes from one to two standard errors. Everything else in §4 is
unchanged: the candidate list, the eligibility requirement, the delta definition, the SE
formula, the tie rule toward identity, and the inner/outer fold construction.

```text
threshold = mean delta[best] + 2 * SE(best)
select the smallest-cap eligible candidate with mean delta <= threshold
```

Rationale: `k = 2` is the standard conservative variant of the one-standard-error rule. It is
applied uniformly to every FOV and source scanner, so it is not a carve-out for the scanner that
failed. Its expected cost is smaller gains where the correction was strong, for example GT450
and S60, which is accepted in exchange for per-fold no-harm.

### Boundary of this amendment

This amendment is applied **once**. The gates of §5 are unchanged and are not relaxed. If the
gate fails again under `k = 2`, RF1M does not proceed to PFM access, no further shrinkage
multiplier is tried, and the failure is reported as the outcome.

## 5. Frozen image-only gates

Thresholds are the frozen RF1 input-only values and are not relaxed.

- material out-of-range fraction: mean `<= 0.005`, patch q99 `<= 0.05`
- projection-induced RGB MAE: mean `<= 0.002`
- final numerical clamp MAE: max `<= 1e-6`
- outer spectrum non-worse than Reinhard within tolerance `1e-12`

RF1M may proceed to PFM access only if **all** of the following hold:

1. 45/45 cell outputs exist with valid identity, hash and gate summaries;
2. each FOV's five outer folds cover 109 unique slides;
3. 75/75 outer fold × scanner × FOV cells are spectrum non-worse and gamut pass;
4. 15/15 aggregate scanner × FOV cells are spectrum non-worse and gamut pass;
5. maximum final clamp MAE over all patches is `<= 1e-6`.

If any gate fails, execution stops before any RF1M PFM endpoint is opened and a dated
pre-outcome amendment to this document is required. Falling back to another branch after seeing
a failed gate is forbidden without that amendment.

## 6. PFM and endpoint rules after the gates pass

Unchanged from RF1 and from the frozen E4--E7 decision record.

- Expected feature population: 436 model--slide shards and 261,600 embeddings for the one
  condition, counting the raw AT2 copy in each shard.
- Audit exact scanner/location identity, raw-AT2 equality, source/checkpoint/statistic/cap and
  contract hashes, finite features and positive norms before any endpoint is read.
- Reuse without alteration the frozen scanner-radius definition, content-margin non-inferiority
  margin, every-source-scanner collapse metrics and thresholds, physical-slide bootstrap seed
  `20260803`, 5,000 bootstrap replicates and the four-PFM common-safe rule.
- Report RF1M beside raw, locked Reinhard, locked frequency calibration and locked RF1. Do not
  recompute or rewrite the locked E5 or RF1 tables.
- Report the paired image-domain spectrum change and gamut projection regardless of direction.
- If the feature population passes, run the existing grouped tissue-type secondary endpoints
  with the same tissue-balanced inference and the minimum-three sensitivity. Tissue results do
  not revise any frozen primary decision.

## 7. Boundaries

- RF1M is developed after the primary E5 outcomes were known. It cannot retroactively enter the
  frozen five-method primary ranking, and it is reported as a post-core extension.
- The paired robust estimator and the gamut-safe Reinhard mapping are excluded from RF1M. They
  are reported as image-only findings with their AKOYA trade-offs stated.
- Better spectrum or better RGB validity is an image-domain result, not evidence of
  representation or biological fidelity.
- Blur-to-sharp output, especially for AKOYA, is not described as recovered ground-truth detail.
- The learned-generator boundary of the RF1 contract is unchanged. RF2 remains a separate study
  requiring its own pre-outcome contract.
