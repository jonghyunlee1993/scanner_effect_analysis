#!/bin/bash
#SBATCH --job-name=e1-native-aa-ert
#SBATCH --partition=defq
#SBATCH --array=0-108%12
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e1_native_aa_spectral_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/run_e1_native_aa_spectral_slide.py --slide-index "$SLURM_ARRAY_TASK_ID" "$@"
