#!/bin/bash
#SBATCH --job-name=plism-reg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-12
#SBATCH --output=logs/plism_reg_%A_%a.out

# One task per stain section: coarse similarity registration of its seven
# scanners onto the section's AT2 reference, plus the shared location set.
# Used only to choose where to read; see docs/e9_plism_native_ert_contract.md.

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"

# libvips pulls in conda's libicuuc, which needs GLIBCXX_3.4.30.  The batch nodes
# ship libstdc++ 3.4.29 in /lib64, so the environment's own copy must win.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/build_plism_section_registration.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --manifest "${PLISM_MANIFEST:-data/PLISM_dataset/manifest_original.csv}" \
  --wsi-dir "${PLISM_WSI_DIR:-data/PLISM_dataset/original_wsi}" \
  --output "${PLISM_REG_OUT:-outputs/plism_section_registration}" \
  --locations "${PLISM_LOCATIONS:-500}"
