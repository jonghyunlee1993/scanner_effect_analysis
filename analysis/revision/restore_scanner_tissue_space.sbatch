#!/bin/bash
#SBATCH --job-name=rv-p0d-fig2
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/scanner_tissue_space_%j.log

# RV-P0d (i): regenerate the Fig. 2 tissue/core positions with the recovered producer.
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/provenance_restoration/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/restore_scanner_tissue_space.py
