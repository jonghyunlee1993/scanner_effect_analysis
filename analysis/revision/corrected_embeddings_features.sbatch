#!/bin/bash
#SBATCH --job-name=rv03-features
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=analysis/revision/results/corrected_embeddings/logs/features_%A_%a.log

# One task per PFM (uni_v1, uni2, virchow2, hoptimus1) x target scanner (versa..s60).
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

extra=()
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
python analysis/revision/corrected_embeddings_features.py \
  --task-index "${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}" "${extra[@]}"
