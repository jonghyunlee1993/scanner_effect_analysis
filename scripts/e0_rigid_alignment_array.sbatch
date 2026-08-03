#!/bin/bash
#SBATCH --job-name=e0-rigid-align
#SBATCH --partition=defq
#SBATCH --array=0-544%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e0_rigid_alignment_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

FAILURE_CELLS="${FAILURE_CELLS:-outputs/e0_native_geometry_cohort/merged/failure_cells.csv}"
SELECTED_PATCHES="${SELECTED_PATCHES:-outputs/e0_feature_manifest_109/feature_manifest.csv}"
VALIS_ROOT="${VALIS_ROOT:-/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/registered_images}"
ALIGNMENT_OUTPUT_ROOT="${ALIGNMENT_OUTPUT_ROOT:-outputs/e0_rigid_alignment}"
REGISTRATION_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
CELL=$("$REGISTRATION_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); c=sorted(set(zip(d["slide_id"],d["scanner"]))); i=int(sys.argv[2]); print("|".join(c[i]) if i < len(c) else "")' "$FAILURE_CELLS" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$CELL" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID SCANNER <<< "$CELL"

"$REGISTRATION_PYTHON" src/run_e0_registration_sentinel.py \
  --slide-id "$SLIDE_ID" \
  --route valis \
  --scanners "$SCANNER" \
  --selected-patches "$SELECTED_PATCHES" \
  --valis-root "$VALIS_ROOT" \
  --search-margin 120 \
  --output-root "$ALIGNMENT_OUTPUT_ROOT/$SCANNER/shards"
