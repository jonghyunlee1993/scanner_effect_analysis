#!/bin/bash
#SBATCH --job-name=e0-rigid-native
#SBATCH --partition=defq
#SBATCH --array=0-544%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_rigid_native_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

FAILURE_CELLS="${FAILURE_CELLS:-outputs/e0_native_geometry_cohort/merged/failure_cells.csv}"
SELECTED_PATCHES="${SELECTED_PATCHES:-outputs/e0_feature_manifest_109/feature_manifest.csv}"
ALIGNMENT_OUTPUT_ROOT="${ALIGNMENT_OUTPUT_ROOT:-outputs/e0_rigid_alignment}"
RIGID_ROOT="${RIGID_ROOT:-/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/registered_images}"
NATIVE_OUTPUT_ROOT="${NATIVE_OUTPUT_ROOT:-outputs/e0_rigid_native_fallback}"
LIST_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
CELL=$("$LIST_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); c=sorted(set(zip(d["slide_id"],d["scanner"]))); i=int(sys.argv[2]); print("|".join(c[i]) if i < len(c) else "")' "$FAILURE_CELLS" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$CELL" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID SCANNER <<< "$CELL"

VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/build_e0_rigid_native_fallback.py \
  --slide-id "$SLIDE_ID" \
  --scanner "$SCANNER" \
  --manifest "$SELECTED_PATCHES" \
  --alignment-root "$ALIGNMENT_OUTPUT_ROOT/$SCANNER/shards" \
  --rigid-root "$RIGID_ROOT" \
  --output "$NATIVE_OUTPUT_ROOT"
