#!/bin/bash
#SBATCH --job-name=trident-envs
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/trident_envs_%j.out

# One environment per encoder.  They cannot share one: H-Optimus asserts
# timm==0.9.16 at import, while UNI2-h needs timm's SwiGLUPacked and reg_tokens,
# which arrived in the 1.0 line.  torch is pinned to the locked E0 runtime so a
# feature difference is a cohort difference and not a runtime difference.

source ~/.bashrc
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm

TORCH="torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124"
COMMON="transformers huggingface_hub h5py pillow scipy einops einops_exts 'numpy<2'"

build () {
  local name="$1" timm_spec="$2"
  echo "=================== $name ($timm_spec) ==================="
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "  exists, skipping create"
  else
    conda create -y -n "$name" python=3.10
  fi
  # shellcheck disable=SC2086
  conda run -n "$name" --no-capture-output pip install --quiet $TORCH
  # shellcheck disable=SC2086
  eval conda run -n "$name" --no-capture-output pip install --quiet "$timm_spec" $COMMON
  conda run -n "$name" --no-capture-output python - <<'PY'
import torch, timm, huggingface_hub as hh
print(f"  torch {torch.__version__}  timm {timm.__version__}  hub {hh.__version__}")
print(f"  SwiGLUPacked: {hasattr(__import__('timm').layers, 'SwiGLUPacked')}")
PY
}

build trident-uni2     "timm==1.0.12"
build trident-conch15  "timm==0.9.16"
build trident-hoptimus "timm==0.9.16"

echo "=================== summary ==================="
for e in trident-uni2 trident-conch15 trident-hoptimus; do
  printf "%-18s " "$e"
  conda run -n "$e" python -c "import timm,torch;print('timm',timm.__version__,'torch',torch.__version__)" 2>/dev/null || echo "FAILED"
done
