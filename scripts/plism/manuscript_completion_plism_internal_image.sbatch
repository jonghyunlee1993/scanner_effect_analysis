#!/bin/bash
#SBATCH --job-name=mc-plism-int
#SBATCH --partition=defq
#SBATCH --array=0-308%32
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/internal_image_%A_%a.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE
sha256sum -c outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/00_contract_v4/runtime_files.sha256
python -m manuscript_completion.plism_external internal-image-task \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction
