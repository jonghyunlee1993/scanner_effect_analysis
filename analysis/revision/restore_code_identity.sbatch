#!/bin/bash
#SBATCH --job-name=rv-p0d-code-id
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/code_identity_%j.log

# RV-P0d / RV06: compare run-time byte-code caches with recovered/current sources.
# cp39 caches need the project Python 3.9; cp313 caches need Python 3.13 (standard library only).
export PYTHONDONTWRITEBYTECODE=1
cd "${JOB_WORKDIR}"
set -euo pipefail

"${JOB_CONDA_PREFIX}/bin/python" analysis/revision/restore_code_identity.py
"${JOB_PYTHON313:-/home/leej70/miniconda3/bin/python3.13}" analysis/revision/restore_code_identity.py
