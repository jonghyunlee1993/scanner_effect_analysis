# Reinhard and residual-frequency image harmonization extension

**Status:** result-locked post-core method-development extension; PanNormal E0--E7 locks remain
authoritative and unchanged

**Updated:** 2026-08-03

## Why this extension follows from the locked results

The original Reinhard color-transfer paper matches the per-channel mean and standard
deviation in a decorrelated color space. It discusses global color statistics, optional
swatches/clusters, higher moments, hue correction and gamma, but does not define a Fourier,
scale-space or low-/mid-/high-spatial-frequency analysis. The luminance-like `l` in the
original `lαβ` space is not a low-spatial-frequency component. Our E5 implementation applies
the same global-moment principle in CIE L*a*b* with cross-fitted scanner-to-AT2 population
statistics.

This distinction matters because the locked PanNormal results show both that Reinhard is the
only primary image method that is safe and invariance-improved in all four PFMs and that the
scanner shift is frequency-dependent, tissue/slide-dependent and not reducible to the frozen
one-dimensional blur--sharpen family. Reinhard reduced scanner radius by 27.6%, 6.0%, 11.9%
and 3.7% in ResNet50, UNI v1, CONCH v1 and Virchow2, respectively. The model dependence and
the remaining scanner signal motivate a direct test of whether residual spectrum correction
adds value after global color-statistic alignment.

The earlier low-frequency affine plus copied-source-high-frequency construction was a
mechanistic diagnostic, not a deployable normalizer. Independent low/high recombination
created out-of-gamut RGB values and hard clipping; in the AKOYA-reference diagnostic,
9--19% of non-AKOYA pixels were affected. A plausible-looking generated image would not by
itself resolve this problem because generative realism is not evidence of matched biological
content.

## Prior-work and scope boundary

- [Reinhard et al. (2001)](https://doi.org/10.1109/38.946629) introduced global color
  transfer by matching decorrelated color-channel moments; it did not analyze spatial
  frequency.
- Frequency-aware histology normalization is not itself unprecedented. For example,
  [Li et al. (BIBM 2023)](https://doi.org/10.1109/BIBM58861.2023.10385658) proposed Fourier
  stain normalization and augmentation. The novelty here is not merely separating low and
  high frequencies.
- The intended contribution is the combination of same-stain paired multi-scanner
  acquisition, audited native/resampling geometry, measured residual scanner spectra,
  gamut-aware reconstruction and the already frozen invariance--fidelity evaluation.
- FEATMAP/full-affine feature harmonization is not part of this extension. Existing CORAL and
  Procrustes results remain contextual feature-space references, not a new method claim.
- A learned generator is not allowed to replace the analytic baseline. Stage A first tests
  whether residual-frequency correction and a physically interpretable gamut projection can
  improve on Reinhard. Any learned residual generator is a later method-development stage
  with a new pre-outcome contract.

## Stage-A hypothesis

After training-slide Reinhard alignment, estimate the remaining source-to-AT2 radial power
ratio and apply only the resulting mean-optical-density residual. The same scalar residual is
added to all three OD channels, so chromatic OD differences are preserved. Instead of
independent RGB hard clipping, project that residual into the per-pixel OD interval that maps
exactly to valid RGB8 support.

This design tests three linked hypotheses:

1. global color moments and residual scanner frequency response contain complementary
   correctable components;
2. fitting the frequency gain after Reinhard avoids double-correcting the raw spectrum;
3. projecting a shared OD residual is less destructive than channel-wise RGB clipping.

The executable frozen rules are in
[`e5_reinhard_residual_frequency_contract.md`](e5_reinhard_residual_frequency_contract.md).
The locked outcome is reported in
[`e5_reinhard_residual_frequency_results.md`](e5_reinhard_residual_frequency_results.md).

## Interpretation rules

- Better RGB validity or closer target spectrum is an image-domain result, not proof of
  representation or biological fidelity.
- Scanner-radius reduction is interpreted only after the existing content non-inferiority
  and every-scanner collapse gates are applied.
- Tissue type remains the only biological grouping. The grouped tissue probe is secondary
  and cannot establish morphology preservation or clinical utility.
- Blur-to-sharp output, especially for AKOYA, cannot be described as recovered ground-truth
  detail unless the generated structure agrees with the physical paired AT2 acquisition.
- Stage A is a post-core extension developed after the primary E5 outcomes were known. It
  cannot retroactively enter the frozen five-method primary ranking.
