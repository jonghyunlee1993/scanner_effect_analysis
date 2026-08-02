#!/bin/bash
#SBATCH --job-name=exp07-bg
#SBATCH --partition=defq
#SBATCH --array=0-653%48
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp07_bg_%A_%a.out

source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/exp07_mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python src/run_exp07_background_cell.py \
  --manifest outputs/exp07_raw_background_109x6/manifest.csv \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-root outputs/exp07_raw_background_109x6 \
  --patches 100 \
  --attempts-per-cell 1200 \
  --fallback-attempts 4000
