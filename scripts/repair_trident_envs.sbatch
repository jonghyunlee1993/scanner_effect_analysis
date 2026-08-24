#!/bin/bash
#SBATCH --job-name=trident-deps
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/trident_deps_%j.out

# TRIDENT's own pyproject is the authority on what importing it needs; guessing
# the list from memory is what cost the two previous rounds.  Two deliberate
# departures from it:
#
#   opencv-python-headless, not opencv-python -- identical cv2 API without the
#   GUI shared libraries, which a compute node does not have.
#
#   segmentation-models-pytorch omitted -- it depends on timm and would be free
#   to move the pinned version, and nothing in the import chain that reaches
#   encoder_factory imports it at module level (checked: segmentation_models/
#   load.py pulls only torch, torchvision and trident.IO).
#
# timm is re-pinned afterwards and the version asserted, so nothing installed
# here can quietly move it.

source ~/.bashrc
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export PYTHONNOUSERSITE=1

declare -A TIMM=( [trident-uni2]=1.0.12 [trident-conch15]=0.9.16 [trident-hoptimus]=0.9.16 )

for env in trident-uni2 trident-conch15 trident-hoptimus; do
  echo "=================== $env ==================="
  PYTHONNOUSERSITE=1 conda run -n "$env" --no-capture-output pip install \
      opencv-python-headless geopandas shapely matplotlib ipywidgets tqdm
  # anything above is free to have touched timm; put it back and prove it
  PYTHONNOUSERSITE=1 conda run -n "$env" --no-capture-output \
      pip install "timm==${TIMM[$env]}"
  PYTHONNOUSERSITE=1 conda run -n "$env" python -c "
import timm; assert timm.__version__ == '${TIMM[$env]}', timm.__version__
print(f'  timm pinned at {timm.__version__}')"
done

status=0
for pair in "trident-uni2 uni_v2" "trident-conch15 conch_v15" "trident-hoptimus"; do
  set -- $pair
  env="$1"; enc="${2:-}"
  echo "--- check $env ${enc:-(imports+slide only)} ---"
  PREFIX=$(conda run -n "$env" python -c "import sys;print(sys.prefix)")
  PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    conda run -n "$env" --no-capture-output python scripts/_env_check.py $enc || status=1
done
[ $status -eq 0 ] && echo "ALL ENVS VERIFIED" || { echo "SOME CHECKS FAILED"; exit 1; }
