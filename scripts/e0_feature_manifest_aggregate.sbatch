#!/bin/bash
#SBATCH --job-name=e0-manifest-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e0_feature_manifest_aggregate_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/analyze_e0_feature_manifest.py
