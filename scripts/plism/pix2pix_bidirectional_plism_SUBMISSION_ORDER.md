# Bidirectional full-training PLISM validation

These commands are documentation only. The implementation does not submit
jobs. Run them only after all ten full-training run manifests are complete.

```bash
export JOB_OUTPUT_ROOT="$PWD/outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/09_bidirectional_full_training"
export JOB_PLISM_RENDER_CONTRACT_ROOT="$PWD/outputs/scanner_batch_effect_analysis_2026-09-17/08_plism_external/pix2pix_gt450_to_at2_image_safety"
mkdir -p "$JOB_OUTPUT_ROOT/10_plism_external_bidirectional/logs"

getent hosts respublica-01 && scontrol ping
sinfo -p defq -o "%.8P %.5D %.15C %.10m %t"
PREP=$(sbatch --parsable \
  --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT",JOB_PLISM_RENDER_CONTRACT_ROOT="$JOB_PLISM_RENDER_CONTRACT_ROOT" \
  scripts/plism/pix2pix_bidirectional_plism_prepare.sbatch)

getent hosts respublica-01 && scontrol ping
bash ~/Scripts/gpu_availability.sh gpuq
EVAL=$(sbatch --parsable --dependency="afterok:$PREP" \
  --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT" \
  scripts/plism/pix2pix_bidirectional_plism_evaluate_array.sbatch)

AGG=$(sbatch --parsable --dependency="afterok:$EVAL" \
  --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD",JOB_OUTPUT_ROOT="$JOB_OUTPUT_ROOT" \
  scripts/plism/pix2pix_bidirectional_plism_aggregate.sbatch)
```

The prepare step freezes all ten checkpoint hashes before PLISM evaluation.
The evaluation array contains five folds per direction. The aggregate step
treats folds as model replicates, averages them within each physical PLISM
section, and bootstraps the 13 sections.

The existing PLISM GT450/AT2 renders are read-only inputs. No render, prior
evaluation, training output, or report is overwritten.
