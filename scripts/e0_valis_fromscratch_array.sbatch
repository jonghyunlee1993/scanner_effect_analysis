#!/bin/bash
#SBATCH --job-name=e0-valis-array
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --array=0-544%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e0_valis_fromscratch_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

FAILURE_CELLS="${FAILURE_CELLS:-outputs/e0_native_fallback_actions/from_scratch_cells.csv}"
VALIS_OUTPUT="${VALIS_OUTPUT:-outputs/e0_valis_from_scratch_v2}"
LIST_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
CELL=$("$LIST_PYTHON" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); c=sorted(set(zip(d["slide_id"],d["scanner"]))); i=int(sys.argv[2]); print("|".join(c[i]) if i < len(c) else "")' "$FAILURE_CELLS" "$SLURM_ARRAY_TASK_ID")
if [[ -z "$CELL" ]]; then
  exit 0
fi
IFS='|' read -r SLIDE_ID SCANNER <<< "$CELL"

VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/run_e0_valis_rigid_from_scratch.py \
  --slide-id "$SLIDE_ID" \
  --scanner "$SCANNER" \
  --output "$VALIS_OUTPUT"
