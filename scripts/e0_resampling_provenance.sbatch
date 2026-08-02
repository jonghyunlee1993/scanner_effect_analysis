#!/bin/bash
#SBATCH --job-name=e0-resample-prov
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e0_resampling_provenance_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

# The historical Python 3.10 environment is required only to deserialize VALIS
# 1.2.0 registrar objects.  The activated cpath environment remains unchanged.
VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/extract_e0_resampling_provenance.py
