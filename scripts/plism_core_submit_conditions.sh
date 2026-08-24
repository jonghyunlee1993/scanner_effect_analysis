#!/bin/bash
# Submit the four correction conditions for all three encoders.
# One array per encoder because the panel cannot share a conda environment;
# each is throttled to 3 GPUs so the three together stay under the 10-GPU QOS cap.
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
DEP="${1:-}"
[ -n "$DEP" ] && DEP="--dependency=afterok:$DEP"

submit () {
  local encoder="$1" env="$2"
  # shellcheck disable=SC2086
  sbatch --parsable $DEP \
    --job-name="cond-$encoder" \
    --array=0-363%3 \
    --export=ALL,PFM_ENCODER="$encoder",PFM_ENV="$env" \
    scripts/plism_core_condition_array.sbatch
}

U=$(submit uni_v2 trident-uni2)
C=$(submit conch_v15 trident-conch15)
H=$(submit hoptimus1 trident-hoptimus)
echo "uni_v2=$U conch_v15=$C hoptimus1=$H"
echo "$U:$C:$H" > outputs/plism_scratch/condition_jobs.txt
