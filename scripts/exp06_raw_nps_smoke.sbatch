#!/bin/bash
#SBATCH --job-name=exp06-smoke
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp06_smoke_%j.out

JOB_CONDA_PREFIX="${JOB_CONDA_PREFIX:-/home/leej70/miniconda3/envs/cpath}"
JOB_WORKDIR="${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="/tmp/exp06_smoke_mpl_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/run_exp06_raw_nps_pilot.py \
  --slides 12.5_13 \
  --patches-per-slide 30 \
  --attempts-per-cell 40 \
  --fallback-attempts 200 \
  --output outputs/exp06_raw_nps_smoke_1slide
