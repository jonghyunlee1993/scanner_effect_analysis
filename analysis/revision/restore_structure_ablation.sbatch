#!/bin/bash
#SBATCH --job-name=rv-p0d-struct-cpu
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/structure_%x_%j.log

# RV-P0d (iii) CPU stages. JOB_STAGE=evaluate_images or JOB_STAGE=compare (default;
# runs the recovered comparison on frozen and on regenerated intermediates and verifies).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/restore_structure_ablation.py --stage "${JOB_STAGE:-compare}"
