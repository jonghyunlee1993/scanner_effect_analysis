#!/bin/bash
#SBATCH --job-name=pn-pair-cache
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

mkdir -p "$JOB_OUTPUT_ROOT/logs"
LOG_PATH="$JOB_OUTPUT_ROOT/logs/preprocess_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-single}.log"
exec >"$LOG_PATH" 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

SLIDE_ARGS=(--maximum-pairs "${JOB_MAXIMUM_PAIRS:-1000}")
if [[ -n "${JOB_MAX_CANDIDATES:-}" ]]; then
  SLIDE_ARGS+=(--max-candidates "$JOB_MAX_CANDIDATES")
fi
if [[ "${JOB_OVERWRITE:-0}" == "1" ]]; then
  SLIDE_ARGS+=(--overwrite)
fi

python src/preprocessing_patch_extraction_matching_and_QC.py slide \
  --output-root "$JOB_OUTPUT_ROOT" \
  --task-index "${SLURM_ARRAY_TASK_ID:-${JOB_TASK_INDEX:?set JOB_TASK_INDEX for a non-array job}}" \
  "${SLIDE_ARGS[@]}"
