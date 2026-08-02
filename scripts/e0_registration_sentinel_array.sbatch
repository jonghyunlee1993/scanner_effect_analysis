#!/bin/bash
#SBATCH --job-name=e0-reg
#SBATCH --partition=defq
#SBATCH --array=0-14%8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_registration_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

MANIFEST="outputs/e0_registration_sentinel/sentinel_manifest.csv"
SLIDE_ID=$(python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); print(d.iloc[int(sys.argv[2])]["slide_id"])' "$MANIFEST" "$SLURM_ARRAY_TASK_ID")

python src/run_e0_registration_sentinel.py \
  --slide-id "$SLIDE_ID" \
  --output-root outputs/e0_registration_sentinel/shards
