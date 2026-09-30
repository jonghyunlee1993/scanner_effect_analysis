#!/bin/bash
#SBATCH --job-name=rv-p0d-gan-pfm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-17%6
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/gan_pfm_%A_%a.log

# RV-P0d (ii): regenerate one (PFM, scanner, method) shard of the PLISM GAN cross-PFM
# evaluation per array task, exactly as the original array 23670688 (0-17).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${JOB_WORKDIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/restore_plism_gan_crossencoder.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" --batch-size 8 --generator-batch-size 16
