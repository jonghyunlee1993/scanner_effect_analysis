#!/bin/bash
#SBATCH --job-name=e8-res-probe
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/e8_res_probe_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

T="${E8_TARGET:-gt450}"
python src/analyze_e8_condition_probe.py \
  --encoder-index "$SLURM_ARRAY_TASK_ID" \
  --source "rf1u=outputs/rf1u_multitarget/features/$T" \
  --source "e8=outputs/e8_residual/features/$T" \
  --source "feature=outputs/e8_target_harmonization/$T" \
  --output "outputs/e8_residual/probe"
