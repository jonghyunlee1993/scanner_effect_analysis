#!/usr/bin/env bash
#SBATCH --job-name=pa-gan-pred
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-19%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/gan_predict_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/gan_predict_%A_%a.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
export PANEL_A_STAGE=gan_predict
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'

targets=(versa akoya s360 s60)
target_index=$((SLURM_ARRAY_TASK_ID / 5)); test_fold=$((SLURM_ARRAY_TASK_ID % 5)); target_scanner="${targets[${target_index}]}"
if [[ "${JOB_METHOD}" == "pix2pix" ]]; then
  seed=$((20263017 + target_index * 100 + test_fold))
  method_root="${JOB_OUTPUT_ROOT}/03_pix2pix"
  run_dir="${method_root}/01_training/runs/at2_to_${target_scanner}/fold_${test_fold}/seed_${seed}"
else
  seed=$((20264017 + target_index * 100 + test_fold))
  method_root="${JOB_OUTPUT_ROOT}/04_cyclegan"
  run_dir="${method_root}/01_training/runs/${target_scanner}_at2/fold_${test_fold}/seed_${seed}"
fi
prediction_dir="${method_root}/02_predictions/at2_to_${target_scanner}/fold_${test_fold}"
mkdir -p "${prediction_dir}" "${MPLCONFIGDIR}"

python -m manuscript_completion.panel_a predict-gan \
  --method "${JOB_METHOD}" \
  --checkpoint "${run_dir}/checkpoints/best_image_only.pt" \
  --run-manifest "${run_dir}/run_manifest.json" \
  --sample-index "${JOB_SAMPLE_INDEX}" \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --target-scanner "${target_scanner}" \
  --test-fold "${test_fold}" \
  --output-dir "${prediction_dir}" \
  --batch-size 32 --num-workers "${SLURM_CPUS_PER_TASK}"
