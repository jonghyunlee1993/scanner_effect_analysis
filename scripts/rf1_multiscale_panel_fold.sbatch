#!/bin/bash
#SBATCH --job-name=rf1-ms-panel
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf1_multiscale_panel_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

if (( SLURM_ARRAY_TASK_ID < 5 )); then
  PILOT_FOV=224
  PILOT_FOLD="$SLURM_ARRAY_TASK_ID"
else
  PILOT_FOV=512
  PILOT_FOLD="$((SLURM_ARRAY_TASK_ID - 5))"
fi

export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
python src/analyze_rf1_multiscale.py \
  --mode fold \
  --fov "$PILOT_FOV" \
  --fold "$PILOT_FOLD" \
  --batch-size 6
