#!/bin/bash
#SBATCH --job-name=e9-pfm-fetch
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e9_pfm_fetch_%j.out

source ~/.bashrc
conda activate "${FETCH_ENV:-trident-uni2}"
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
# token lives in ~/.cache/huggingface and is shared by every environment
python src/fetch_e9_pfm_checkpoints.py "$@"
