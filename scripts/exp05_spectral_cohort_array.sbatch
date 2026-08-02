#!/bin/bash
#SBATCH --job-name=exp05-slide
#SBATCH --partition=defq
#SBATCH --array=0-108%12
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp05_cohort_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="$JOB_WORKDIR/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

REGISTRY="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/registered_images/qc_registered_eval/registered_pair_summary.csv"
SLIDE_ID=$(python -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1],dtype={"slide_id":str}); d=d[d["paired_all_scanners"]].sort_values("slide_id"); print(d.iloc[int(sys.argv[2])]["slide_id"])' "$REGISTRY" "$SLURM_ARRAY_TASK_ID")

python src/run_exp05_spectral_pilot.py \
  --slide-id "$SLIDE_ID" \
  --slides 1 \
  --patches-per-slide 100 \
  --replicate-groups 5 \
  --skip-figures \
  --output "outputs/exp05_spectral_cohort_109/shards/$SLIDE_ID"
