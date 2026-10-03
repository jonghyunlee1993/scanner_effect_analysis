#!/usr/bin/env bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=table2_metrics

set -euo pipefail

: "${JOB_CONDA_PREFIX:?JOB_CONDA_PREFIX is required}"
: "${JOB_WORKDIR:?JOB_WORKDIR is required}"
: "${JOB_STAGE:?JOB_STAGE is required}"
: "${JOB_OUTPUT_ROOT:?JOB_OUTPUT_ROOT is required}"

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${JOB_OUTPUT_ROOT}/.torch"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
mkdir -p "${TORCH_HOME}" "${MPLCONFIGDIR}" "${JOB_OUTPUT_ROOT}/logs"
cd "${JOB_WORKDIR}"

case "${JOB_STAGE}" in
  prepare-lpips)
    exec python analysis/paper/table2_benchmark.py prepare-lpips
    ;;
  pannormal|plism-conventional|plism-learned)
    : "${SLURM_ARRAY_TASK_ID:?array task is required for ${JOB_STAGE}}"
    exec python analysis/paper/table2_benchmark.py "${JOB_STAGE}" \
      --task-index "${SLURM_ARRAY_TASK_ID}" --batch-size 8 \
      --uni-batch-size 32 --generator-batch-size 16
    ;;
  *)
    echo "unknown JOB_STAGE=${JOB_STAGE}" >&2
    exit 2
    ;;
esac
