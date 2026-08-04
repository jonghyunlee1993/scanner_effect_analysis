#!/bin/bash
#SBATCH --job-name=rf1u-front
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rf1u_frontier_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/test_rf1u_unpaired.py
python src/analyze_rf1u_frontier.py "$@"
