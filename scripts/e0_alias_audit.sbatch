#!/bin/bash
#SBATCH --job-name=e0-alias-audit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e0_alias_audit_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python src/run_e0_alias_audit.py
