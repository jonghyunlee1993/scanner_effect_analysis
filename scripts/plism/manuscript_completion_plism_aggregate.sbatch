#!/bin/bash
#SBATCH --job-name=mc-plism-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/aggregate_%j.log

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
python -m manuscript_completion.plism_external aggregate \
  --bootstraps 20000 \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction
