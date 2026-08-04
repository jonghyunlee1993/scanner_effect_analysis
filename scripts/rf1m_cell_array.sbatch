#!/bin/bash
#SBATCH --job-name=rf1m-cell
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rf1m_cell_%A_%a.out

# One array covers both stages so no barrier idles a GPU slot between them:
# indices 0-14 build the outer cells, 15-44 build the inner cells.  Any GPU type
# in gpuq is acceptable; pinning a100-40g left three free a100 devices unused.

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

RF1M_OUTER_TASKS=15
if [ "$SLURM_ARRAY_TASK_ID" -lt "$RF1M_OUTER_TASKS" ]; then
  RF1M_STAGE=outer
  RF1M_TASK_INDEX="$SLURM_ARRAY_TASK_ID"
else
  RF1M_STAGE=inner
  RF1M_TASK_INDEX=$((SLURM_ARRAY_TASK_ID - RF1M_OUTER_TASKS))
fi

python src/build_rf1m_cell.py \
  --stage "$RF1M_STAGE" \
  --task-index "$RF1M_TASK_INDEX" \
  --batch-size 4
