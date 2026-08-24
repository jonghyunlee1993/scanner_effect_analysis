#!/bin/bash
#SBATCH --job-name=plism-cond-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plism_core_cond_smoke_%j.out

# Code-path smoke for the fused condition extractor.  The Reinhard reference is
# the old population table because the new one is still being built; the numbers
# here are throwaway, only the path is being checked.
source ~/.bashrc
conda activate trident-conch15
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/extract_plism_core_pfm.py \
  --task-index 6 --encoder-id conch_v15 --condition reinhard_at2 \
  --locations outputs/plism_core_eval_locations \
  --population outputs/plism_population_stats \
  --output outputs/plism_scratch/cond_smoke

python - <<'PY'
import h5py, json, numpy as np
p = "outputs/plism_scratch/cond_smoke/reinhard_at2/conch_v15/GIVH_SQ.h5"
with h5py.File(p) as h:
    m = json.loads(h.attrs["meta"])
    f = h["features"][:]
print("shape", f.shape, "finite", bool(np.isfinite(f).all()))
for k in ("condition","condition_detail","gamut_limited","projection_fraction",
          "eval_subsample","slide","precision","native_fov_px","tiles"):
    print(f"  {k}: {m.get(k)}")
PY
echo "COND SMOKE OK"
