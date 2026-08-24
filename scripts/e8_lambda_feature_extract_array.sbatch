#!/bin/bash
#SBATCH --job-name=e8-lambda-feat
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/e8_lambda_feature_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src"

# GPU is the scarce resource, so the sweep is not run at full width. The curve
# shape is resolved on ResNet50 and UNI across all four lambdas; CONCH and
# Virchow2 only get the two ends, which is enough to confirm the direction
# transfers without spending four times the GPU hours on it.
ENCODERS=(0 1 0 1 0 1 0 1 2 3 2 3)
TARGETS=(gt450_l050 gt450_l050 gt450_l150 gt450_l150 \
         gt450_l200 gt450_l200 gt450_l300 gt450_l300 \
         gt450_l050 gt450_l050 gt450_l300 gt450_l300)

ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]}
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> encoder $ENCODER target $TARGET"

python src/extract_rf1u_features.py \
  --encoder-index "$ENCODER" \
  --target "$TARGET" \
  --energy outputs/e8_lambda_energy \
  --output outputs/e8_lambda_features
