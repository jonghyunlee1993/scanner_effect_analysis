#!/bin/bash
#SBATCH --job-name=e0-candidate-reserve
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_candidate_reserve_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

ANALYSIS_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
"$ANALYSIS_PYTHON" src/build_e0_feature_manifest_slide.py "$@"
