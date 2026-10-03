# Analyses frozen before the revision

These analyses were completed before the manuscript revision of 2026-09-29 and are
still used by the current manuscript in `00_manuscript/`, either directly or as
inputs to `analysis/revision/`. Run commands from the repository root with the
`cpath` environment; results are written to `results/` (not tracked by Git).
The tested Python and package versions are in
[`environment_versions.json`](environment_versions.json).

| Analysis | Code | Results | Used in the manuscript |
| --- | --- | --- | --- |
| Scanner metadata | `scanner_native_resolution.csv` | — | Supplementary Table S1 |
| Scanner-by-tissue mixed-effects model | `build_direct_slide_contrasts.py`, `slide_level_lmm.py`, `joint_interaction_lmm.py`, `*_direct_slide_lmm.sbatch`, `run_slide_level_lmm.sbatch` | `results/direct_slide_lmm/` | Table 1, Supplementary Tables S2 and S3; the same model is reused by `analysis/revision/gain_tissue_dependence.py` |
| Color and high-frequency positions of scanners | `scanner_color_frequency_space.py`, `plism_scanner_axis_transfer.py`, `run_*` wrappers | `results/scanner_color_frequency_space/` | Fig. 2A (drawn by `analysis/revision/manuscript_figure_phenotype_transfer.py`) |
| Augmentation oracle | `augmentation_paired_metrics.py`, `aggregate_augmentation_paired_metrics.*`, `run_augmentation_paired_metrics.sbatch` | `results/augmentation_paired_metrics/` | Supplementary Table S4 and Section 3.2.1 |
| Augmentation in UNI space | `augmentation_common_uni_features.py`, `augmentation_strong_blur_features.py`, `aggregate_augmentation_*`, `compare_uni_spaces_blur_sharp.*`, `plot_augmentation_strong_blur.*` | `results/augmentation_common_uni_space/`, `results/augmentation_strong_blur/` | reference values and the inset location for RV01 and Fig. 3 |
| Image-correction benchmark | `table2_benchmark.py`, `table2_gt450_extension.py`, `run_table2_*.sbatch` | `table2_*_summary.csv`, `table_frequency_summary.csv`, `table2_manifest.json`, `results/plism_gt450_extension/` | Supplementary Table S5; inputs to RV-P0b |
| AKOYA tissue retention | `akoya_tissue_retrieval.*`, `bootstrap_akoya_tissue_retention.*`, `akoya_uni_retention_audit.py`, `frequency_pfm_audit.py` | `results/frequency_pfm_audit/` | Supplementary Section B.5 |
| Frequency increment figure | `plot_frequency_incremental_uni_gain.py` | `outputs/table4_color_frequency_crossencoder_2026-09-25/summary.csv` | Fig. 4 |
| Manuscript build | `build_scanner_tissue_manuscript.sbatch` | `results/manuscript_build/` | compiled PDF |

`table2_manifest.json` records an absolute output path from the historical run under
`00_manuscript/analysis/`; it is kept unchanged as provenance.

## Mixed-effects model

The model uses 103 physical slides, 37 tissue types, 515 scanner-slide contrasts per
measure, 13 measures, 500 parametric bootstrap replicates, and Benjamini–Hochberg
correction. With the project environment active:

```bash
JOB_OUTPUT_ROOT="$PWD/analysis/paper/results/direct_slide_lmm"
mkdir -p "$JOB_OUTPUT_ROOT/logs"
PREP=$(sbatch --parsable --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT",JOB_INPUT="$JOB_OUTPUT_ROOT/direct_slide_contrasts.csv" analysis/paper/prepare_direct_slide_lmm.sbatch)
FIT=$(sbatch --parsable --array=0-12%13 --dependency=afterok:"$PREP" --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT",JOB_INPUT="$JOB_OUTPUT_ROOT/direct_slide_contrasts.csv" analysis/paper/run_slide_level_lmm.sbatch)
sbatch --dependency=afterok:"$FIT" --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT" analysis/paper/aggregate_direct_slide_lmm.sbatch
```

## Image-correction benchmark

`table2_benchmark.py` evaluates frozen PanNormal and PLISM predictions with SSIM,
LPIPS-VGG16, and UNI, aggregated by physical slide (PanNormal) or section (PLISM).
`table2_gt450_extension.py` adds the external GT450 direction. Use
`run_table2_gpu.sbatch`, `run_table2_aggregate.sbatch`, and `run_table2_gt450_*.sbatch`
to reproduce the aggregate sources.
