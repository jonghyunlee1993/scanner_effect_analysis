#!/bin/bash
#SBATCH --job-name=blur-sharp-uni
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/paper/results/augmentation_common_uni_space/logs/compare_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/paper/results/augmentation_common_uni_space/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/compare_uni_spaces_blur_sharp.py
