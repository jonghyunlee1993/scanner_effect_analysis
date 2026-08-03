#!/bin/bash
#SBATCH --job-name=e0-candidate-primary
#SBATCH --partition=defq
#SBATCH --array=0-108%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_candidate_primary_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

PLAN="${PLAN:-outputs/e0_candidate_execution_plan/candidate_primary_slides.csv}"
TRANSFORM_CACHE_ROOT="${TRANSFORM_CACHE_ROOT:-outputs/e0_native_geometry_cohort}"
LIST_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
ROW=$("$LIST_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); i=int(sys.argv[2]); print("|".join(map(str,[d.iloc[i].slide_id,d.iloc[i].manifest_path,d.iloc[i].trial_root])) if i < len(d) else "")' "$PLAN" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$ROW" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID MANIFEST TRIAL_ROOT <<< "$ROW"

VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/run_e0_native_geometry_cohort.py \
  --slide-id "$SLIDE_ID" \
  --manifest "$MANIFEST" \
  --output "$TRIAL_ROOT/primary" \
  --transform-cache-root "$TRANSFORM_CACHE_ROOT"
