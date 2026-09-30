#!/bin/bash
#SBATCH --job-name=rv-p0d-summary
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/summary_%j.log

# RV-P0d: render summary.md from the verification files (run after all restore jobs).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/restore_provenance_summary.py
