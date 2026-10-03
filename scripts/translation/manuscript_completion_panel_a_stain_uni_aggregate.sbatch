#!/usr/bin/env bash
#SBATCH --job-name=pa-stain-uagg
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_uni_aggregate_%j.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_uni_aggregate_%j.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
export PANEL_A_STAGE=stain_uni_aggregate
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'
stain_subdir="${JOB_STAIN_SUBDIR:-01_stain_v4_mu_convergence}"

python -m manuscript_completion.panel_a stain-uni-aggregate \
  --output-root "${JOB_OUTPUT_ROOT}/${stain_subdir}/uni"
