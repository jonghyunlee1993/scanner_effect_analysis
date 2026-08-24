# E0 native WSI acquisition-metadata audit

**Status:** native-header population audit complete; operator-supplied acquisition fields partial,
2026-08-03

## Scope and gate

The frozen native geometry manifest contains 654 unique scanner--slide files: 109 physical
slides for each of six scanners. Every native file was opened with OpenSlide and tifffile;
pixel arrays were not decoded. The audit extracted physical pixel size, nominal
magnification/objective strings, full-resolution compression/photometric/tile tags, TIFF
manufacturer/model/software and scanner-specific ImageDescription fields.

- 654/654 native WSI headers passed.
- Every scanner contributed 109/109 unique native files.
- Native MPP was constant within each scanner; the maximum within-scanner range was 0.
- Every full-resolution image used JPEG compression.
- Acquisition-software strings are reported as header provenance and are not relabeled as
  firmware.

## Header-supported scanner table

| Cohort label | Header-supported model evidence | Native MPP (µm/px) | Nominal magnification/objective evidence | Full-resolution encoding | Product/software evidence |
|---|---|---:|---|---|---|
| AT2 | Aperio AT2; Aperio vendor and ICC/header label | 0.505200 | 20× | SVS, JPEG/RGB Q70, 240×240 tile | Aperio Image Library v12.0.16 |
| GT450 | Aperio Leica Biosystems GT450; `ScannerType=GT450` | 0.262407 | 40× | SVS, JPEG/YCC Q91, 256×256 tile | Aperio Leica Biosystems GT450 v1.5.1 |
| VERSA | Header label `Versa`; exact manufacturer/model pending | 0.274200 | `AppMag=20` | SVS, JPEG/YCC Q75, 512×512 tile | No versioned product/software string |
| AKOYA | PerkinElmer-QPI/Akoya cohort label; exact model pending | 0.499899 | Scan profile 20×; header objective string 10× | QPTIFF, JPEG/RGB Q70, 512×512 tile | Fusion 2.2.0; PerkinElmer-QPI |
| S60 | Hamamatsu S60; TIFF model C13210 | 0.442595 | OpenSlide objective-power 20× | NDPI, JPEG/YCC, 1792×8 tile; quality absent | NZAcquire 3.1.70 |
| S360 | Hamamatsu S360; TIFF model C13220 | 0.460320 | OpenSlide objective-power 20× | NDPI, JPEG/YCC, 1920×8 tile; quality absent | NZAcquire 3.2.20 |

The AKOYA file distinguishes a `10x` objective string from a 20× scan profile and 0.5 µm/px
pixel size. These are preserved as separate fields rather than forcing one nominal value.

## Fields not recoverable from native headers

The following remain author/operator inputs for the manuscript and must not be inferred from
product-family specifications:

- objective numerical aperture for all six scanners;
- firmware version for all six scanners;
- exact manufacturer/model for VERSA and AKOYA;
- configured JPEG quality for S60 and S360.

Acquisition dates and scanner/computer identifiers are available in some headers but are not
required for the primary scanner-effect table and can contain operational identifiers. They
are kept in the ignored audit output rather than copied into the manuscript.

## Operator-supplied acquisition notes

Recorded 2026-08-10 as **operator recollection, not a retrieved experiment record**. It is
kept here because it bears on how AKOYA is described and because nothing in the headers
carries it. Before this appears in a manuscript the underlying record should be located and
cited, or the claim softened to what can be evidenced.

- **AKOYA was re-scanned and reproduced the same result.** The low high-band transfer is
  therefore not a one-off acquisition error.
- **No instrument fault was found.**
- **The acquisition used the clinic's production setting**, not a configuration chosen for
  this study, so the setting is the deployed one rather than a research artefact.

This matters because AKOYA is the extreme point of the panel at a high-band transfer of
0.335 against AT2. The 2026-08-10 contrast audit removes the competing explanations from
the data side: the value survives contrast normalisation (residual 0.358 against AT2),
AKOYA samples at 0.4999 µm/px against AT2's 0.5052 so it is not a sampling effect, and the
per-slide residual shows no time trend across a 36-day acquisition window
(Spearman +0.110, p = 0.25) while varying strongly with tissue type (ANOVA p = 5.5e-04,
tissue medians spanning 5.7×). Low objective numerical aperture remains the only account
consistent with all of it, and the header's 10× objective string is the candidate — but NA
itself is still not recoverable, so the optical mechanism stays a hypothesis.

The practical consequence is the opposite of a caveat: a 3× softer acquisition is something
a deployed clinical configuration actually produces, which is the case a harmonization
method has to handle rather than an outlier to be excused. PLISM contains no comparable
instrument, so that is a coverage gap in the external cohort and not a weakness here.

## Provenance

- Population output: `outputs/e0_scanner_acquisition_metadata/`
- Full header table: `scanner_slide_headers.csv`
- Scanner summary: `scanner_summary.csv`
- Missing-field list: `missing_metadata.csv`
- Gate summary: `summary.json`
- Rebuild: `src/audit_e0_scanner_acquisition_metadata.py`
- SLURM job: `15951018`, `COMPLETED`, exit `0:0`

The audit summary records SHA-256 of the frozen native geometry manifest. Generated metadata
tables remain under ignored `outputs/`; this document records the manuscript-facing result.
