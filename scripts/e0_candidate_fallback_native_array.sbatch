#!/bin/bash
#SBATCH --job-name=e0-candidate-native
#SBATCH --partition=defq
#SBATCH --array=0-544%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_candidate_fallback_native_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

PLAN="${PLAN:-outputs/e0_candidate_execution_plan/candidate_fallback_cells.csv}"
LIST_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
ROW=$("$LIST_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); i=int(sys.argv[2]); cols=["slide_id","scanner","manifest_path","rigid_root","alignment_root","fallback_output_root","transform_cache_root"]; print("|".join(str(d.iloc[i][c]) for c in cols) if i < len(d) else "")' "$PLAN" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$ROW" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID SCANNER MANIFEST RIGID_ROOT ALIGNMENT_ROOT FALLBACK_OUTPUT TRANSFORM_CACHE <<< "$ROW"

VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/build_e0_rigid_native_fallback.py \
  --slide-id "$SLIDE_ID" \
  --scanner "$SCANNER" \
  --manifest "$MANIFEST" \
  --alignment-root "$ALIGNMENT_ROOT" \
  --rigid-root "$RIGID_ROOT" \
  --output "$FALLBACK_OUTPUT" \
  --transform-cache-root "$TRANSFORM_CACHE"
