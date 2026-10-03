#!/usr/bin/env bash
#SBATCH --job-name=p2p-bi-plism-agg
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/09_bidirectional_full_training/10_plism_external_bidirectional/logs/aggregate_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/10_plism_external_bidirectional/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/10_plism_external_bidirectional" \
  aggregate \
  --bootstrap-replicates 20000 \
  --seed 20260917
