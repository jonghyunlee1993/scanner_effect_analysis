#!/bin/bash
#SBATCH --job-name=plism-dl
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --array=0-90%12
#SBATCH --output=logs/plism_download_%A_%a.out

# PLISM native WSIs from figshare+ (CC BY 4.0).  Build the manifest first:
#   python src/fetch_plism_manifest.py --article original
#
# Payload lives in the shared lab tree at
# /mnt/isilon/oldridge_lab/batch_effects/PLISM_dataset and is reached through the
# repository symlink data/PLISM_dataset.
#
# Throttle: measured 4.5 MB/s aggregate at 6 concurrent tasks and 16.9 MB/s at 12,
# so figshare limits per stream rather than in aggregate.  12 finishes the 180.8 GB
# in about three hours; raising it further risks rate limiting for little gain.

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/download_plism_file.py \
  --manifest "${PLISM_MANIFEST:-data/PLISM_dataset/manifest_original.csv}" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "${PLISM_DEST:-data/PLISM_dataset/original_wsi}"
