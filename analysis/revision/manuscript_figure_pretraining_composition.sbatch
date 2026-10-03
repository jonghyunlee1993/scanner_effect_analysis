#!/bin/bash
#SBATCH --job-name=mf-pretrain-composition
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/manuscript_figures/logs/pretraining_composition_%j.log

# Optional: PAPER_FIGURES_DIR redirects output (drafts); default is 00_manuscript/figures.
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/manuscript_figures/logs/.mpl"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/manuscript_figure_pretraining_composition.py
