#!/bin/bash
#SBATCH --job-name=rv03-assemble
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=analysis/revision/results/corrected_embeddings/logs/assemble_%A_%a.log

# One task per PFM (0 uni_v1, 1 uni2, 2 virchow2, 3 hoptimus1); needs images + features parts.
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
[[ -n "${JOB_PARTS_ROOT:-}" ]] && extra+=(--parts-root "${JOB_PARTS_ROOT}")
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
[[ -n "${JOB_DATASETS:-}" ]] && extra+=(--datasets "${JOB_DATASETS}")
[[ -n "${JOB_FEATURES_ROOT:-}" ]] && extra+=(--features-root "${JOB_FEATURES_ROOT}")
[[ -n "${JOB_SMOKE:-}" ]] && extra+=(--smoke)
python analysis/revision/corrected_embeddings_assemble.py \
  --task-index "${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}" "${extra[@]}"
