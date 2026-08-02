#!/bin/bash
#SBATCH --job-name=exp05-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp05_cohort_aggregate_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/analyze_exp05_spectral_cohort.py \
  --shards outputs/exp05_spectral_cohort_109/shards \
  --output outputs/exp05_spectral_cohort_109 \
  --expected-slides 109 \
  --expected-patches 100 \
  --expected-replicates 5
