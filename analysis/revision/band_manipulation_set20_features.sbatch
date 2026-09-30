#!/bin/bash
#SBATCH --job-name=rv02-band-feat
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-20
#SBATCH --output=analysis/revision/results/band_manipulation_set20/logs/features_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${JOB_WORKDIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/band_manipulation_set20/.mplconfig"
# Rendering runs in one forked worker per CPU; keep each worker single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/band_manipulation_set20_features.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${JOB_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT}}" \
  --max-slides "${JOB_MAX_SLIDES:-0}" \
  --models "${JOB_MODELS:-uni_v1,uni2,virchow2,hoptimus1}" \
  --workers "${SLURM_CPUS_PER_TASK}"
