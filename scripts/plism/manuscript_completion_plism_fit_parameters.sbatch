#!/bin/bash
#SBATCH --job-name=mc-plism-fit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/fit_%j.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
sha256sum -c outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/00_contract_v4/runtime_files.sha256
python -m manuscript_completion.plism_external fit-parameters \
  --vahadane-selection-manifest outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/00_contract/vahadane_selection_manifest_v4.json \
  --stain-v4-manifest outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/01_stain_v4_mu_convergence/aggregate_manifest.json \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction
