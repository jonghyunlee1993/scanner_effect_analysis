# E0 resampling provenance and 2D alias audit

**Status:** diagnostic complete; Phase 0 `Revise` triggered

**Audit contract:** alias/true-in-band power must be ≤5% in both the sinusoid
mixing matrix and broadband-noise validation for every q05/q50/q95 transform profile.
The gate and both resampling chains were committed before the synthetic results were run.

## Provenance recovered

The retained native-rigid route was generated with VALIS 1.2.0, pyvips 2.2.3 and
libvips 8.15.3. `warp_and_save_slide()` used its default libvips bicubic interpolator.
Native-resolution scaling and the rigid transform were combined in one affine warp;
there was no explicit anti-alias prefilter. Outputs were stored with lossless LZW.

The historical export script then reopened each AT2-grid output and wrote the moving
scanner's original MPP into its TIFF/OME metadata. This makes the scanner output tags
physically incorrect even though all registered images have the AT2 canvas. The ERT
frequency axis must therefore come from native AT2 (`0.5052 µm/px`), never from the
moving output TIFF tag.

Exact registrar objects were available for 96 slides in the retained `rigid_all` audit
route. The primary historical six-scanner outputs do not retain their transform objects,
so the exact linear-transform audit uses the known native-rigid route. Its q05/q50/q95
effective scales cover the normal transform envelope; the already rejected catastrophic
GT450 registrations are not treated as plausible resampling profiles.

| Scanner | Exact transforms | Native MPP | Native px / output px, median (q05–q95) | Affine anisotropy q95 | Output MPP tags matching AT2 |
|---|---:|---:|---:|---:|---:|
| AT2 | 96 | 0.505200 | 1.000 (1.000–1.000) | 1.00002 | 100% |
| GT450 | 96 | 0.262407 | 1.922 (1.919–1.926) | 1.00122 | 0% |
| VERSA | 96 | 0.274200 | 1.842 (1.838–1.844) | 1.00128 | 0% |
| S60 | 96 | 0.442595 | 1.141 (1.139–1.143) | 1.00118 | 0% |
| S360 | 96 | 0.460320 | 1.097 (1.095–1.099) | 1.00122 | 0% |

The recovered full-resolution matrices were numerically checked against VALIS
`warp_xy`; the maximum elementwise discrepancy for the check was `5.7e-12` output
pixels per native pixel.

## Frozen synthetic audit

GT450 and VERSA were tested at observed q05, q50 and q95 affine-scale profiles.

- Direction/phase sweep: 8 directions × cosine/quadrature phase, 0.04 cycles/µm
  input spacing, native Nyquist limit.
- Output mixing: 256 px Hann-windowed 2D periodogram on the AT2 grid, radial bins of
  0.025 cycles/µm.
- Broadband validation: six deterministic white-noise replicates split into true
  0.60–0.90 input and supra-target-Nyquist input components.
- Ringing: eight step-edge directions.
- Original chain: one-pass libvips bicubic affine.
- Explicit-AA chain: libvips Lanczos3 reduction at the smallest affine singular value,
  followed by the residual bicubic affine. This makes the final affine non-decimating.
- Translation was omitted from the linear mixing calculation because it changes phase,
  not power; the phase sweep covers sampling-phase sensitivity.

## Result

The values below are q50 transform profiles. All three profiles produced the same gate
decision.

| Scanner | Chain | Sinusoid alias / in-band | White-noise alias / in-band | Median amplitude retention, 0.60–0.90 | Max angular anisotropy | Max edge overshoot | Gate |
|---|---|---:|---:|---:|---:|---:|---:|
| GT450 | Original bicubic | 1.555 | 1.113 | 0.983 | 0.179 dB | 15.6% | fail |
| GT450 | Explicit AA | 0.0084 | 0.0101 | 0.739 | 4.081 dB | 17.2% | pass |
| VERSA | Original bicubic | 1.329 | 0.965 | 0.980 | 0.207 dB | 13.6% | fail |
| VERSA | Explicit AA | 0.0142 | 0.0154 | 0.806 | 3.294 dB | 17.2% | pass |

Across all three profiles, original-chain sinusoid ratios were `1.547–1.565` for GT450
and `1.310–1.329` for VERSA; broadband ratios were `1.113–1.117` and `0.954–0.966`.
The explicit-AA ranges were `0.0084–0.0146` and `0.0132–0.0158` for sinusoid mixing,
and `0.0101–0.0176` and `0.0150–0.0170` for broadband noise. Thus the original route
fails by roughly twenty-fold or more, while the explicit-AA route passes the 5% alias
gate.

Passing the alias gate is not cost-free. Lanczos3 reduces high-band response and has
orientation-dependent attenuation near the circular Nyquist boundary. The reported
ERT must retain this qualification; the synthetic result does not convert ERT into an
absolute scanner MTF.

## Decision

1. Do not rebuild E1–E3 from the current registered TIFFs. Their 0.60–0.90 band contains
   material folded supra-Nyquist power for GT450 and VERSA.
2. Recreate the native-WSI common grid with an explicit anti-alias stage before the
   decimating affine. Since the primary historical transform objects are unavailable,
   recover or rerun the rigid registration rather than attempting to filter already
   registered pixels.
3. Preserve the same physical slide unit and the frozen 10,900 canonical centers, then
   repeat the integer residual-alignment and 512 px bounds audit on the new grid.
4. Retain 0.60–0.90 cycles/µm only conditionally: the regenerated grid must reproduce the
   ≤5% alias gate. Report passband attenuation, angular anisotropy and ringing alongside
   the audited ERT.
5. Until that rebuild passes, legacy high-band results may only be called an
   `effective spectrum after historical registration/resampling`; they cannot support a
   scanner-transfer interpretation.

## Reproducibility

- Pre-registered implementation: commits `a25ebca`, `be0a740`
- Provenance job: `15878405` (96/96 registrar objects loaded; 0 failures)
- Synthetic audit job: `15878875` (completed); figure-layout rerun `15880307`
- Provenance outputs: `outputs/e0_resampling_provenance/`
- Alias outputs and mixing matrices: `outputs/e0_alias_audit/`
- Main audit figure: `outputs/e0_alias_audit/figure_e0_alias_audit.png`

Generated CSV, NPZ, JSON and figure artifacts remain under ignored `outputs/`; the code,
SLURM launchers, tests and this decision record are tracked in Git.
