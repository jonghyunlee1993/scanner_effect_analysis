#!/bin/bash
#SBATCH --job-name=e9-core-results-doc
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e9_core_results_doc_%j.out

# Emit docs/e9_plism_core_results.md from the analysis outputs.  This died with
# exit 137 on the login node: load_all() reads every plism_core_native_psd band
# matrix into one frame, which the ~6 GB login node cannot hold.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

out=docs/e9_plism_core_results.md
tmp="$(mktemp "${out}.XXXXXX")"
python src/build_e9_core_results.py > "$tmp"
# only replace the committed doc if the generator ran to completion
grep -q '^## 7\.' "$tmp"
mv "$tmp" "$out"
wc -l "$out"
