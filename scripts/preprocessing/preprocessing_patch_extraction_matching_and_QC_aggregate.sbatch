#!/bin/bash
#SBATCH --job-name=pn-pair-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

mkdir -p "$JOB_OUTPUT_ROOT/logs"
LOG_PATH="$JOB_OUTPUT_ROOT/logs/aggregate_${SLURM_JOB_ID}.log"
exec >"$LOG_PATH" 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

python src/preprocessing_patch_extraction_matching_and_QC.py aggregate \
  --output-root "$JOB_OUTPUT_ROOT" \
  --maximum-pairs "${JOB_MAXIMUM_PAIRS:-1000}"
