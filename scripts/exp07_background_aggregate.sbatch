#!/bin/bash
#SBATCH --job-name=exp07-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp07_aggregate_%j.out

source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MPLCONFIGDIR="/tmp/exp07_aggregate_mpl_${SLURM_JOB_ID}"

python src/aggregate_exp07_background.py \
  --input-root outputs/exp07_raw_background_109x6 \
  --reference gt450
