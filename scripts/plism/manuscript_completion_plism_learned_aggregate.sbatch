#!/usr/bin/env bash
#SBATCH --job-name=plism-learned-agg
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/06_plism_learned_external/logs/aggregate_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/pix2pix" \
  aggregate \
  --bootstrap-replicates 20000 \
  --seed 20260921

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/cyclegan" \
  aggregate \
  --bootstrap-replicates 20000 \
  --seed 20261921
