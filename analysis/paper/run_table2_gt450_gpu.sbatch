#!/usr/bin/env bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=plism_gt450

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$JOB_ORIGINAL_BENCHMARK/.torch"
export MPLCONFIGDIR="$JOB_GT450_OUTPUT/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

case "$JOB_STAGE" in
  conventional)
    python analysis/paper/table2_gt450_extension.py conventional \
      --task-index "$SLURM_ARRAY_TASK_ID" --batch-size 8 --uni-batch-size 32
    ;;
  learned)
    python analysis/paper/table2_gt450_extension.py learned \
      --task-index "$SLURM_ARRAY_TASK_ID" --batch-size 8 --generator-batch-size 16
    ;;
  *)
    echo "unknown JOB_STAGE=$JOB_STAGE" >&2
    exit 2
    ;;
esac
