#!/bin/bash
#SBATCH --job-name=rv-p0d-gan-cmp
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/gan_pfm_compare_%j.log

# RV-P0d (ii): aggregate the regenerated shards with the recovered code and compare with
# the frozen outputs/plism_gan_crossencoder_2026-09-25/. Submit with
# --dependency=afterok:<restore_plism_gan_crossencoder array id>.
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${JOB_WORKDIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/restore_plism_gan_crossencoder.py --aggregate
