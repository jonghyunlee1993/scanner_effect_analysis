#!/bin/bash
#SBATCH --job-name=exp05-tissue
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp05_tissue_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="/tmp/exp05_tissue_mpl_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/analyze_exp05_tissue_random_slopes.py \
  --cohort-output outputs/exp05_spectral_cohort_109 \
  --annotation /mnt/isilon/oldridge_lab/batch_effects/metadata/pan_normal_annotation.csv \
  --expected-slides 109
