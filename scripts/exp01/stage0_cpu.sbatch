#!/bin/bash
# Parallel Stage-0 audit for GT450, VERSA, Akoya, and the S60 lattice.
#SBATCH --job-name=exp01_stage0
#SBATCH --partition=defq
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp01_stage0_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

scanners=(gt450 versa akoya s60)
extra=()
if [[ "${SLURM_ARRAY_TASK_ID}" == "3" ]]; then
  extra+=(--write-s60-view)
fi
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/audit_exp01_stage0.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
    --scanner "${scanners[$SLURM_ARRAY_TASK_ID]}" "${extra[@]}"
