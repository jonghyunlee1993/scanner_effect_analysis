#!/bin/bash
#SBATCH --job-name=mc-plism-render
#SBATCH --partition=defq
#SBATCH --array=0-25%13
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/render_%A_%a.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
sha256sum -c outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/00_contract_v4/runtime_files.sha256
python -m manuscript_completion.plism_external render-task \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction
