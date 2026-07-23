#!/bin/bash
# Evaluate the exact Exp-06 epoch-10 and epoch-30 LF16 milestones sequentially.
#SBATCH --job-name=exp06_eval_milestones
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/exp06_eval_milestones_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
config="${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}"
run_dir="${EXP06_RUN_DIR:-outputs/exp06a_lf16_10slide}"
for epoch in 010 030; do
  checkpoint="$run_dir/ckpt/epoch${epoch}.ckpt"
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python src/eval_exp06.py --config "$config" --checkpoint "$checkpoint" \
      --output "$run_dir/eval_epoch${epoch}"
done
