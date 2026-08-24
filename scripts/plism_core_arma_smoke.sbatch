#!/bin/bash
#SBATCH --job-name=plism-arma-smoke
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/plism_arma_smoke_%j.out

# One-slide smoke over the three Arm A passes on the new core grid, before the
# 91-task arrays go out.  Task 0 is GIVH/AT2; task index 12 lands on an SQ file,
# which is the one the label correction touches.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

SQ_TASK=$(python - <<'PY'
import pandas as pd
m = pd.read_csv("data/PLISM_dataset/manifest_original.csv")
row = m[(m.stain == "GIVH") & (m.scanner == "SQ")].iloc[0]
print(int(row.task_index))
PY
)
echo "SQ/GIVH task index = $SQ_TASK"

for T in 0 "$SQ_TASK"; do
  python src/build_plism_population_stats.py --task-index "$T" --limit 40 \
    --output outputs/plism_scratch/arma_smoke_pop
  python src/extract_plism_native_psd.py --task-index "$T" --patches 40 \
    --output outputs/plism_scratch/arma_smoke_psd
  python src/extract_plism_reinhard_bands.py --task-index "$T" --limit 40 \
    --population outputs/plism_population_stats \
    --output outputs/plism_scratch/arma_smoke_rein
done
echo "SMOKE OK"
