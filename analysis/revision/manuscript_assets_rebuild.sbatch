#!/bin/bash
#SBATCH --job-name=mf-assets-rebuild
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/manuscript_figures/logs/assets_rebuild_%j.log

# Regenerates every manuscript figure and table touched by the 2026-10-02 revision
# (all script-produced assets except the raw gallery and the coherence figure).
# Optional: PAPER_FIGURES_DIR redirects figure output (drafts); default is 00_manuscript/figures.
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/manuscript_figures/logs/.mpl"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/manuscript_tables.py > /dev/null
python analysis/revision/manuscript_figure_phenotype_transfer.py
python analysis/revision/manuscript_figure_augmentation_trajectories.py
python analysis/paper/plot_frequency_incremental_uni_gain.py
python analysis/revision/manuscript_figure_band_sensitivity.py
python analysis/revision/manuscript_figure_sensitivity_robustness.py
python analysis/revision/manuscript_figure_pretraining_composition.py
echo done
