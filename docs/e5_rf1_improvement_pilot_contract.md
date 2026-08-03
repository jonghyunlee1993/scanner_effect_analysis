# E5-RF1 analytic-improvement image-only pilot contract

**Status:** FROZEN BEFORE NEW VARIANT PFM ACCESS  
**Frozen:** 2026-08-03  
**Scope:** exploratory RF1 refinement for the current paper; RF2 learned generation remains a
separate future study

## Boundary

This pilot develops four analytic refinements in parallel:

1. scanner-specific no-harm gain strength with identity as an allowed action;
2. paired robust residual-spectrum estimation with reliability shrinkage;
3. localized multiscale-band correction; and
4. gamut-safe Reinhard mapping without independent RGB hard clipping.

No new PFM embedding, scanner-radius endpoint, tissue endpoint, or existing locked RF1 result
may be used to choose among variants. Existing locked results may be used only to state the
known limitation motivating a branch, such as the S360 spectrum counterexample. This stage is
an image-domain development pilot and cannot replace the result-locked RF1 condition.

## Common population and splitting

- Use the frozen 109-slide, six-scanner, 100-location native-AA grid and FOVs 224, 256 and 512.
- Retain the existing five outcome-blind physical-slide outer folds. All scanner acquisitions
  and locations from an outer-held-out slide remain excluded from fitting.
- A hyperparameter or branch choice that uses paired image outcomes must be selected inside the
  four outer-training folds. Identity/no correction must always be an explicit candidate.
- All source-to-target comparisons use the paired AT2 patch at the same physical location.

## Image-only endpoints

The pilot records, by FOV and source scanner:

- post-correction log-spectrum RMSE to paired AT2;
- paired RGB/OD error and paired mean-OD gradient error;
- proposed out-of-range fraction, correction/projection fraction and numerical clamp error;
- the distribution of chosen gain strength or multiscale coefficients; and
- deterministic image panels and amplified residuals.

Spectrum or paired-pixel criteria are descriptive under imperfect registration. A variant must
also retain the RF1 exact-output-range property and must not introduce a worse numerical clamp
than `1e-6` mean absolute RGB error.

## Branch-specific rules

### 1. No-harm strength

Scale the fitted log gain toward identity with a scanner-specific strength in `[0,1]`. Select
strength using inner-training image-only loss with a one-standard-error preference toward
identity. A scanner whose estimated improvement is not positive receives identity.

### 2. Paired robust estimator

Estimate paired patch log-power ratios before pooling. Use a robust location estimator and
shrink frequency bins toward identity when paired support or coherence is weak. Do not use
tissue labels or representation endpoints.

### 3. Multiscale correction

Use a fixed small number of prespecified spatial-frequency bands. Modify only the shared
mean-OD component and apply the same exact OD-support projection as RF1. Identity is included
for every band.

### 4. Gamut-safe Reinhard

The primary pilot candidate starts with the usual Reinhard Lab proposal. If its converted RGB
is outside the cube, retain a common fraction of the source-to-proposal RGB displacement for
all three channels, choosing the largest fraction that stays in `[0,1]`. This source-ray map
changes only pixels that the original Reinhard would clip, preserves the RGB correction
direction at each affected pixel, and uses a final clamp only as a floating-point safeguard.

## Decision sequence

1. Complete and audit all four image-only branches.
2. Compare failures and complementary behavior; do not rank by visual conspicuity.
3. Write and freeze one executable combined-candidate contract, including all coefficients and
   tie rules.
4. Only then encode the selected condition through the four frozen PFMs and apply the existing
   invariance, content and every-scanner collapse gates.

