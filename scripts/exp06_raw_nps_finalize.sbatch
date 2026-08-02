#!/bin/bash
#SBATCH --job-name=exp06-final
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp06_finalize_%j.out

JOB_CONDA_PREFIX="${JOB_CONDA_PREFIX:-/home/leej70/miniconda3/envs/cpath}"
JOB_WORKDIR="${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="/tmp/exp06_finalize_mpl_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/finalize_exp06_raw_nps_pilot.py \
  --main outputs/exp06_raw_nps_pilot_5slides \
  --supplement outputs/exp06_raw_nps_supplement_versa_8-12_19 \
  --output outputs/exp06_raw_nps_pilot_5slides_final \
  --slide 8-12_19 \
  --scanner versa
