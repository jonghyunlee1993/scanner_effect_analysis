#!/bin/bash
#SBATCH --job-name=rf1m-adapt
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rf1m_slide_adaptive_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/test_rf1m_slide_adaptive.py
python src/analyze_rf1m_slide_adaptive.py "$@"
