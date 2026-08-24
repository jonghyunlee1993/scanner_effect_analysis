#!/bin/bash
#SBATCH --job-name=plism-psd
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-90%16
#SBATCH --output=logs/plism_psd_%A_%a.out

# Native-resolution radial spectra for the 91 PLISM WSIs, one task per slide.
# No registration and no resampling; see docs/e9_plism_native_ert_contract.md.

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

python src/extract_plism_native_psd.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --manifest "${PLISM_MANIFEST:-data/PLISM_dataset/manifest_original.csv}" \
  --wsi-dir "${PLISM_WSI_DIR:-data/PLISM_dataset/original_wsi}" \
  --output "${PLISM_PSD_OUT:-outputs/plism_native_psd}" \
  --patches "${PLISM_PATCHES:-500}"
