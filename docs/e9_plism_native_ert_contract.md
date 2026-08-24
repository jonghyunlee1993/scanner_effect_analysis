# E9 — PLISM native-resolution ERT, external validation contract

**Status:** FROZEN 2026-08-08, before any PLISM spectrum was computed
**Data:** `data/PLISM_dataset/original_wsi`, 91 native WSIs, md5-verified, CC BY 4.0
**Relation to the core study:** post-core external check. Nothing here enters the
locked E0–E7 results or the frozen E5 five-method ranking.

## 1. Why this cohort answers something ours cannot

The locked high-band effective relative transfer (§3.1 of the report) is measured
on a cohort whose scanners were run in different modes — AT2 at 0.5052 µm/px
(20×), GT450 at 0.2624 µm/px (40×) — so every scanner had to be brought onto one
common grid before its spectrum could be compared. Resampling removes
high-frequency content, which is the quantity being measured. §10 answers that
objection by argument: the most heavily reduced scanner is the sharpest, so the
reduction cannot be driving the ordering.

PLISM runs **all seven scanners at 40×**, native 0.220–0.262 µm/px. The lowest
Nyquist in the panel is GT450's at 1.908 cyc/µm, well above the 0.99 cyc/µm top
of our own high band. The locked band 0.10–0.99 cyc/µm is therefore measurable on
every PLISM scanner **at native resolution, with no resampling and no
registration at all**. This replaces the §10 argument with a measurement.

Second axis: PLISM crosses 13 H&E staining conditions with the 7 scanners. Our
cohort has no independent colour axis, so it cannot test whether the spectral
signature is a property of the optics or of the chemistry.

## 2. Design and estimator

Within one stain condition the 7 scanners image the **same physical section**, so
a scanner contrast is content-matched by construction. Across stains the sections
are serial. The replicate unit for scanner inference is therefore the
**stain section, n = 13** — not the patch and not the TMA core. All bootstraps
resample sections.

Per WSI:

1. Tissue mask from a ≈64× downsampled level. A pixel is tissue when its optical
   density relative to that slide's own 99th-percentile white level exceeds 0.15.
   **The criterion is mean-OD only.** A variance- or detail-based tissue rule
   would preferentially reject the softer scanner's patches and bias the exact
   ratio being estimated.
2. 500 patch centres sampled uniformly from positions whose full patch footprint
   is tissue, seeded from the slide id.
3. Each patch is read at **level 0 on the scanner's own pixel grid**, side
   L = 129.33 µm (= 256 × 0.5052 µm, the study's ResNet50/UNI field of view), so
   the patch pixel count differs by scanner and the physical extent does not.
4. RGB → optical density, mean over channels, patch mean removed, 2-D Hann
   window, FFT. Mode power is normalised as `|X|² / N⁴`, which sums to the patch
   variance and is therefore independent of N.
5. Frequency per mode is computed with that scanner's own native MPP, in cyc/µm.
6. Band power is the mean mode power over the annulus, averaged over patches.

Bands, fixed here:

| Band | Range (cyc/µm) | Role |
|---|---|---|
| high | 0.10 – 0.99 | **primary**, identical to the locked §3.1 band |
| low | 0.02 – 0.10 | contrast reference for the normalised statistic |
| extended | 0.99 – 1.90 | descriptive only; above our own cohort's Nyquist |

`ERT_s = high_s / high_AT2`, computed within a stain and then pooled over stains,
reported as log₂. AT2 is the reference by construction, as in the locked table.

**Normalised ERT** `(high/low)_s / (high/low)_AT2` is reported beside it. A pure
difference in overall stain contrast or scanner gain moves both bands together
and cancels here, so this statistic isolates spectral *shape*.

## 3. What is predicted, frozen from the locked cohort

Locked high-band ERT against AT2 (report §3.1): GT450 1.478, S60 1.044,
AT2 1.000, VERSA 0.938, S360 0.842, AKOYA 0.335.

Four of six scanners appear in PLISM: **AT2, GT450, S60, S360**. VERSA and AKOYA
do not, so the extreme of the locked panel is not covered here. S210, SQ and
Philips UFS are new and carry no prediction.

| ID | Prediction | Locked value | Test |
|---|---|---:|---|
| **P1** | log₂ ERT(GT450) > 0 | +0.563 | section-bootstrap CI excludes 0 |
| **P2** | log₂ ERT(S360) < 0 | −0.248 | section-bootstrap CI excludes 0 |
| **P3** | GT450 > S60 > S360 | — | point-estimate ordering |
| S1 | rank correlation with the locked 4 shared values | — | descriptive |
| S2 | P1–P3 survive on the normalised ERT | — | shape, not contrast |
| S3 | scanner × stain interaction share of log high-band power | — | optics vs chemistry |

**S60 versus AT2 is deliberately not a directional prediction.** The locked
separation is +0.062 log₂, far too small to expect to replicate across cohorts,
and claiming it afterwards would be reading the data.

## 4. Success, failure, and what each would mean

- **Replicates** — P1, P2 and P3 all hold: the ordering is a property of the
  scanner models, not of our cohort's scan profiles or render chain, and the §10
  resampling caveat is closed by measurement rather than argument.
- **Fails** — P1 or P2 inverts: the locked signature is specific to the
  acquisition mode. Our cohort ran AT2/S60/S360 at 20× and GT450 at 40×; PLISM
  runs all at 40×. That outcome would not overturn the locked measurement, which
  describes the acquisitions we have, but it would confine the claim to those
  acquisitions and must be reported as such.
- **Mixed** — direction holds, magnitude does not: expected, and reported as
  ordinal agreement only.

Either way the result is reported. This contract exists so that it is reported
the same way whichever it is.

## 5. Not claimed

- Nothing here re-opens the locked E0–E7 results or the E5 ranking.
- Content matching is **statistical**, from pooling ~500 patches over a shared
  physical section, not paired. No pixel is warped and no registration is used.
- Scanner focus, compression and vendor sharpening all fold into this estimate,
  as in the locked measurement. It is effective transfer on tissue, not an MTF.
- PLISM was digitised at one institution on one TMA block. n = 13 sections is a
  smaller replicate base than our 109 slides and the CIs will be correspondingly
  wide.

## 5a. Amendment 1 — locations are registered, pixels are not (2026-08-08)

**What changed.** Section 2 selected patches independently on each WSI's own
tissue mask. That is replaced by a single set of locations defined once per
section on its AT2 reference and mapped to the other six scanners through a
coarse similarity transform. Patches are still read at level 0 on each scanner's
own pixel grid by integer crop. **No pixel is interpolated, warped or resampled at
any point**, so the property this cohort was chosen for is untouched. Registration
is used only to decide *where to read*.

**Why.** The unpaired mask failed its own content-matching requirement, and the
failure was scanner-dependent in exactly the way that would corrupt the estimate:

- A fixed relative-OD cut kept 1.2% of GT450's area as tissue against 18–19% for
  AT2 and S360, leaving 407 eligible positions against 140,227. GT450 renders
  brighter (16 µm thumbnail mean 239.9 against 204–223 for the rest).
- Replacing the fixed cut with per-slide Otsu fixed the gross failure — tissue
  fractions 0.18–0.30 — but GT450's threshold still hit the floor, and the
  resulting spectra were not usable.
- Registration also revealed a defect no mask could have handled: **the Philips
  scan is rotated 180°** relative to the others (+179.85°). Independent sampling
  would silently have compared different tissue.

**What was already seen when this amendment was written.** One section (GIVH),
60 patches per slide, under the Otsu variant: log₂ ERT of +0.36 (S360), −0.08
(S210), −0.51 (SQ), −0.78 (S60), −1.97 (GT450), −2.54 (P), and normalised log₂
of +0.05, +0.02, −1.03, −0.49, −0.04, −0.52. These are recorded here because the
amendment was made after seeing them. They are discarded and not used.

The amendment is driven by a diagnosed defect in content matching, not by the
direction of that result — P1 (GT450 sharper) reads as a strong *failure* in the
discarded numbers, and the change makes it no easier to pass. The predictions in
section 3, the bands in section 2 and the endpoints in section 4 are **unchanged**.

**Registration QC, frozen now.** Per section and scanner: AKAZE keypoints matched
to the AT2 reference at 16 µm/px, RANSAC similarity transform. A scanner enters
the analysis only if inliers ≥ 100 and the fitted scale is within 0.98–1.02 of
unity. Fitted scale is itself a check on the MPP metadata that the whole
physical-frequency argument rests on; the GIVH section returned 0.9912–1.0013.

**Residual jitter.** Registration is coarse, so mapped patches overlap rather than
coincide, with an expected offset of order 16 µm on a 129 µm patch. A power
spectrum is translation-invariant, so this adds noise and not bias. The empirical
check is that the seven scanners' distributions of patch mean OD agree; that
comparison is reported with the results.

## 5b. Amendment 2 — a mislabelled pair, a soft scan, and the response gate (2026-08-21)

Found while building the cohort's alignment report
(`data/PLISM_dataset/ALIGNMENT.md`). Nothing here changes a prediction, a band or
an endpoint. Three of the four items require action in the locked pipeline.

### 1. Two SQ files hold each other's section — **act on this**

`GIVH_SQ.ndpi` contains the HRH section and `HRH_SQ.ndpi` contains the GIVH
section. Matched against the AT2 reference of all thirteen sections, `GIVH_SQ`
scores NCC 0.747 on HRH against 0.111 on its own label, and `HRH_SQ` scores 0.831
on GIVH against 0.130; the other eleven sections score 0.09–0.32. Over eight TMA
cores the verdict is unanimous, 16 of 16, while S360 through the identical
comparison is correct 16 of 16 at 0.824–0.983. Six of the seven scanners agree
with each other under the published labels and only SQ disagrees, so it is the two
SQ files that are mislabelled. Refitting SQ's transform against the corrected file
takes RANSAC inliers from 405 to 1,324 (GIVH) and 288 to 2,354 (HRH), back into
the 1,000–6,000 range of a healthy block.

Four alternatives were excluded before this conclusion: wrong core (section-outline
Dice 0.93, transforms agreeing to 40–52 µm against a 3600 µm pitch), a large offset
(NCC over ±800 µm peaks at 0.10–0.27 with a runner-up 0.88–0.99 as high — a noise
field), scale error (0.4–2.5× sweep peaks at 0.17), and blur (band power and
contrast normal, and NCC tolerates blur anyway).

**Consequence for E9.** SQ's GIVH and HRH spectra were computed on each other's
sections. The scanner-level ERT pools over stains, so the pooled SQ estimate is
nearly unaffected — the same two spectra enter it, under swapped labels. What is
affected is anything that treats stain as a factor rather than a nuisance:
`analyze_plism_stain_covariate.py`'s stain × scanner interaction, and any bootstrap
that resamples sections, since two of SQ's thirteen replicate labels are wrong.
The correction is `src/plism_dataset_corrections.py`, applied by
`build_plism_core_map.py` and switchable with `--no-file-corrections`.

### 2. S60's HRH scan is out of focus — record, do not repair

Correctly labelled and correctly registered: patches sit 4–11 µm from the
reference and blur-tolerant NCC confirms the tissue at 0.46–0.82. High-band power
is down five- to tenfold. This is a real property of the scan, and it matters here
more than elsewhere, because the endpoint of this study *is* high-band power. Any
per-(section, scanner) sharpness statistic that includes S60/HRH is measuring a
focus failure, not the scanner.

### 3. The residual gate is not sufficient on its own — **act on this**

Every downstream analysis gates on `residual_um <= 1.0`. Phase correlation always
returns a peak, and on unrelated content that peak is noise biased toward zero
shift by the Hann window. Measured over the refinement: of the location–scanner
pairs whose response falls below 0.3, 28% land under the 1 µm gate by chance —
1.21% of all measurements admitted while aligned to unrelated tissue. Under the
residual gate alone SQ/GIVH retained 11.1% of its locations; every one was such a
leak, so the true figure was 0%.

**Recommendation:** gate on `residual_um <= 1.0 and response >= 0.3`. `response` is
already stored by `build_plism_location_refinement.py`, so no re-measurement is
needed — only the gate expression in `build_plism_population_stats.py`,
`extract_plism_reinhard_bands.py`, `render_plism_conditions.py`,
`analyze_plism_random_slopes.py` and `analyze_plism_stain_covariate.py`.

### 4. Cores can be named without the warp

Section 5a labelled TMA cores by position, on the argument that PLISM's 46
tissue-type labels live in the authors' Elastix canvas and importing them would
need the warp this work avoids. That was too strong. Naming a core does not need
pixel accuracy — a core is 2.5 mm across on a 3.6 mm pitch — so a single global
similarity between the canvas occupancy lattice and the section's own AT2 tissue
mask suffices, and it carries only a *name* inward. No analysed pixel is placed by
it. Two checks say the fit is right rather than merely converged: fitted
independently on each of the thirteen sections, the canvas pixel comes out at
0.2188–0.2219 µm, recovering the Hamamatsu 0.220 µm pitch the canvas was built on
and which the fit was never given; and the rotation comes out within 3.25° of zero
every time.

The location set for the alignment report uses the contract's fixed OD floor rather
than `max(otsu, 0.05)`. On a TMA, Otsu splits dark tissue from pale tissue rather
than tissue from glass: on MY it lands at 0.076, above the median OD of ten of the
forty-six cores, which are then never sampled. Both rules are mean-OD only, so the
bias section 2.1 warns about is not reintroduced. Locations are a lattice at the
patch pitch rather than a random sample, which takes tissue coverage from 6% to
85–90% per section. This set lives in `outputs/plism_core_registration` and is
separate from `outputs/plism_section_registration`, which the locked ERT pipeline
still reads.

## 5c. Amendment 3 — the cohort rebuilt on the core grid (2026-08-22)

Amendment 2 recorded that a location set covering 85-90% of the tissue exists in
`outputs/plism_core_registration`, separate from the sparse set the locked ERT
pipeline reads. This amendment moves the whole PLISM arm onto it, and swaps the
encoder panel. Nothing about the estimator, the bands, the thresholds or the
bootstrap changes; the cohort underneath them does.

### What moved

| | Locked (sparse) | This amendment |
|---|---|---|
| locations per section | ~285, random | 8,105–9,751, patch-pitch lattice |
| measured scanner-to-reference | 26,712 | 817,817 |
| tissue coverage | 6 % | 85.1–90.1 %, mean 88.2 % |
| cores | position index | 46 published tissue names |
| SQ file correction | not applied | applied |
| encoders | E0: ResNet50, UNI v1, CONCH v1, Virchow2 | E9: UNI2-h, CONCHv1.5, H-optimus-1 |

Outputs are written beside the locked ones rather than over them:
`plism_core_population_stats`, `plism_core_reinhard_bands`,
`plism_core_native_psd`, `plism_core_rf1u_destinations`,
`plism_core_random_slopes`, `plism_core_stain_covariate`,
`plism_core_native_ert`, `plism_core_feature_correction`,
`plism_core_pfm_frontier`, `plism_core_condition_features`. The locked results
stand as they are; this is a second cohort build, not an edit of the first.

### Four decisions that needed making

**1. Fit on everything, evaluate on a frozen subsample.** Three of the four
locked endpoints — the unmatched-content quantile, the pairwise distance and the
collapse gram — are location-by-location matrices, so their cost grows as the
square of the location count while the estimate stops moving after a few hundred.
The endpoints are therefore read on a fixed, seeded subsample of 2,000 locations
per section (`build_plism_core_eval_locations.py`), while CORAL and Procrustes
are fitted on every location. This is the split the rebuild exists for: the
fitting problem is what the extra density buys.

**2. The correction conditions are encoded on that same subsample, fused.** The
sparse route wrote rendered tiles to disk and encoded them afterwards. At this
density that is 640 GB per condition, so `extract_plism_core_pfm.py --condition`
applies Reinhard and RF1U in memory between the read and the encode. The raw
condition is not re-encoded — the full feature store already holds those vectors
from the same pass with the same crop and eval transform, so it is taken as a
subset, and every condition of a section then sits on identical physical
locations.

**3. The out-of-focus block is excluded at block level, from sharpness
statistics only.** `HRH_S60` is a real scan and stays in the dataset. Every
sharpness analysis is run twice — `all_blocks` and `focus_ok` — because this is
a judgement, not a fact, and the report should be able to show both. The
exclusion is never per patch: rejecting individual patches for looking blurred
is the detail-based selection section 2.1 forbids, and Amendment 2's per-patch
`response >= 0.3` recommendation is **withdrawn for this reason**. `response`
remains stored per location so any analysis can gate on it deliberately; with
the SQ files corrected, the leak it was aimed at falls from "the whole of two
blocks" to 0.70 % of non-reference measurements.

**4. Per-core subsampling is spread, not taken from the head.**
`build_replicates` kept the first `per_core` locations by id. Under random
sampling that was as good as any subset; on a lattice the ids run in raster
order, so the head confines every block to one corner of its core and reads a
spatial gradient as sampling error. It now takes evenly spaced positions, which
are identical across scanners because the surviving location set is.

### What the naming makes newly possible

The nesting is now fitted twice on the identical rows: with the **staining
condition** on top and the core inside it, which is the axis PanNormal does not
have, and with the **tissue type** on top and the section inside it, which is
PanNormal section 3.2's own model. Before the cores were named the second was
not available, and `analyze_plism_random_slopes.py` carried
`tissue_type = stain` as a placeholder. `--level tissue` replaces it.

### Known limit that the rebuild does not touch

The replicate count is still 13 sections. Thirty-one times the patches leaves
every section-level bootstrap interval as wide as it was, RF1U's shrinkage still
dominating its own gains, and the destructive regime still unreachable. Section
17's "untested" verdict stands unchanged and is not a data-volume problem.

### What the rebuild changed in the answers

Recorded here because two of these are corrections to the August edition rather
than additions to it, and the locked numbers stay as they are.

**CORAL.** The report's qualification -- CORAL fails on PLISM for the highest
dimensional encoders -- does not reproduce. At full density it is safe and
improved for all three E9 encoders (+10.7% to +18.3% toward AT2). The sweep shows
why: below one training sample per embedding dimension CORAL makes the scanner
spread worse by whole multiples, and it climbs through zero somewhere between 6
and 33 samples per dimension. **The break-even is not a constant after dividing
by the dimension** -- CONCHv1.5 at 768 dimensions crosses at ~6.5 per dimension
and both 1536-dimensional encoders at ~32.5, a five-fold difference in a quantity
that already normalises for parameter count. It has to be measured per encoder,
which costs one refit sweep on features that already exist.

**The three-term decomposition of section 3.1 does not survive.** Running the
contrast audit on PLISM's own pixels -- the same measure used on PanNormal, raw
fine-band power divided by the squared optical-density spread -- gives GT450 an
excess of 1.40x, against 1.36x on PanNormal. The August edition put PLISM's
excess at 0.90x and concluded that equalising sampling removes it, which is what
supported splitting GT450's advantage into roughly half optics and half sampling
density. Equalising sampling does not remove it. What survives is the
load-bearing claim, that raw band power is dominated by contrast, which is the
mechanism behind the hinge in section 12.

**The destination rule splits in two.** On the band energies the gains are fitted
from it is exact -- Spearman +0.964 over seven destinations, AT2 the worst
destination in the panel, 15 of 18 cells attenuating toward it. Carried into
representation space on the E9 panel it does not order the same way: GT450 beats
AT2 for one of three encoders. The rule is established where it is fitted and
unconfirmed downstream; the E0 panel that read 4 of 4 is not this panel.

**One encoder has no image-space headroom at all.** No image condition clears
zero for H-optimus-1, while a linear map on its embedding takes its scanner probe
from 0.979 to 0.040 and its scanner radius down 31%. An encoder can carry scanner
identity in directions that colour and band correction never touch, which is the
sharpest form of the section 12 boundary this study has produced.

**A cost of the block exclusion, for the record.** Dropping the `HRH/S60` block
removes the whole HRH *section* from the paired nesting and covariate analyses,
because those require all seven scanners at a location. `focus_ok` therefore fits
on twelve sections, not thirteen. Both variants are kept for this reason.

## 6. Execution

    python src/fetch_plism_manifest.py --article original      # done
    sbatch scripts/plism_download_array.sbatch                  # done, 91/91 md5-verified
    sbatch scripts/plism_native_psd_array.sbatch                # per-WSI native spectra
    python src/analyze_plism_native_ert.py                      # ERT, bootstrap, endpoints

### Core-grid rebuild (Amendment 3)

    python src/build_plism_core_map.py                          # done, 13 sections
    sbatch scripts/plism_core_refine_array.sbatch               # done, 817,817 locations
    python src/build_plism_core_eval_locations.py               # frozen evaluation subsample
    sbatch scripts/plism_core_population_array.sbatch           # Lab moments, OD bands
    sbatch scripts/plism_core_psd_array.sbatch                  # native spectra
    sbatch scripts/plism_core_reinhard_array.sbatch             # post-Reinhard bands
    sbatch scripts/plism_core_render_bands_array.sbatch         # RF1U fit input
    sbatch scripts/plism_core_gainfit.sbatch                    # RF1U gains, deployment grid
    bash   scripts/plism_core_submit_conditions.sh              # 4 conditions x 3 encoders
    sbatch scripts/plism_core_arma_analysis.sbatch              # ERT, nesting x2, destinations
    sbatch scripts/plism_core_featcorr.sbatch                   # CORAL/Procrustes + sweep
    sbatch scripts/plism_core_pfm_frontier.sbatch               # image-space endpoints
    python src/build_pannormal_plism_report.py                  # Part IV, data-driven
