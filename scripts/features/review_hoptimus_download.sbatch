#!/bin/bash
#SBATCH --job-name=hoptimus-download
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/encoder_review_2026-09-25/logs/%x_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export HF_HUB_DISABLE_XET=1
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="bioptimus/H-optimus-1",
    filename="pytorch_model.bin",
    cache_dir="outputs/encoder_review_2026-09-25/hf_cache",
)
print(path, flush=True)
PY
