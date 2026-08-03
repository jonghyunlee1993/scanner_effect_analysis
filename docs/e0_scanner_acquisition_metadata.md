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
