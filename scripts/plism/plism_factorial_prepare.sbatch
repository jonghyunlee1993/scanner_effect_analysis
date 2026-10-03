#!/bin/bash
#SBATCH --job-name=plism-fact-prep
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$JOB_OUTPUT_ROOT/.matplotlib"

python src/plism_factorial_external.py \
  --plism-root "$JOB_PLISM_ROOT" \
  --output-root "$JOB_OUTPUT_ROOT" \
  prepare --locations-per-core "${JOB_LOCATIONS_PER_CORE:-32}"
