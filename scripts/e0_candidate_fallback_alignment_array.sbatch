#!/bin/bash
#SBATCH --job-name=e0-candidate-align
#SBATCH --partition=defq
#SBATCH --array=0-544%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e0_candidate_fallback_align_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

PLAN="${PLAN:-outputs/e0_candidate_execution_plan/candidate_fallback_cells.csv}"
LIST_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
ROW=$("$LIST_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); i=int(sys.argv[2]); cols=["slide_id","scanner","manifest_path","rigid_root","alignment_root"]; print("|".join(str(d.iloc[i][c]) for c in cols) if i < len(d) else "")' "$PLAN" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$ROW" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID SCANNER MANIFEST RIGID_ROOT ALIGNMENT_ROOT <<< "$ROW"

REGISTRATION_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
"$REGISTRATION_PYTHON" src/run_e0_registration_sentinel.py \
  --slide-id "$SLIDE_ID" \
  --route valis \
  --scanners "$SCANNER" \
  --selected-patches "$MANIFEST" \
  --valis-root "$RIGID_ROOT" \
  --search-margin 120 \
  --output-root "$ALIGNMENT_ROOT"
