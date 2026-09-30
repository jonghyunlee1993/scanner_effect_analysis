#!/bin/bash
#SBATCH --job-name=rv-p0c-uni
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/uni_extraction_parity/logs/parity_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${JOB_WORKDIR}"
set -euo pipefail
export PYTHONPATH="${JOB_WORKDIR}/src"
export PYTHONDONTWRITEBYTECODE=1

python analysis/revision/uni_extraction_parity.py --workers "${SLURM_CPUS_PER_TASK}"
