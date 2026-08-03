#!/bin/bash
#SBATCH --job-name=e1-native-aa-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e1_native_aa_spectral_aggregate_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/analyze_exp05_spectral_cohort.py \
  --shards outputs/e1_native_aa_spectral_109/shards \
  --output outputs/e1_native_aa_spectral_109 \
  --expected-slides 109 \
  --expected-patches 100 \
  --expected-replicates 5

python src/analyze_exp05_tissue_random_slopes.py \
  --cohort-output outputs/e1_native_aa_spectral_109 \
  --expected-slides 109
