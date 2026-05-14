#!/bin/bash
#SBATCH --job-name ns_uifo
#SBATCH --output neural_surrogate/logs/%x-%j.out
#SBATCH --error neural_surrogate/logs/%x-%j.err
#SBATCH --partition a100-galvani
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

echo "Working directory: $PWD"
echo "Host: $(hostname)"
echo "Started at: $(date)"

mkdir -p neural_surrogate/logs

# External environment to load on the cluster.
ENV_DIR="${ENV_DIR:-/home/soham/repos/envs/dfbench_env}"
source "$ENV_DIR/bin/activate"

# External directory containing campaign .h5 files.
DATA_DIR="${DATA_DIR:-/path/to/external/neural_surrogate_data}"

LOSS_KEY="${LOSS_KEY:-loss_senspow}"
EPOCHS="${EPOCHS:-250}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
TOPOLOGY_DIM="${TOPOLOGY_DIM:-128}"
SEED="${SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
DEVICE="${DEVICE:-auto}"
MULTI_GPU="${MULTI_GPU:-data-parallel}"

echo "Environment: $ENV_DIR"
echo "Data directory: $DATA_DIR"
echo "Loss key: $LOSS_KEY"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Topology dim: $TOPOLOGY_DIM"
echo "Seed: $SEED"
echo "Validation fraction: $VAL_FRACTION"
echo "Device: $DEVICE"
echo "Multi-GPU mode: $MULTI_GPU"
echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"

start=$(date +%s)

python -m neural_surrogate.pipeline \
  --data "$DATA_DIR" \
  --loss-key "$LOSS_KEY" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --topology-dim "$TOPOLOGY_DIM" \
  --seed "$SEED" \
  --val-fraction "$VAL_FRACTION" \
  --device "$DEVICE" \
  --multi-gpu "$MULTI_GPU"

end=$(date +%s)
runtime=$((end - start))

echo "Finished at: $(date)"
echo "Runtime: ${runtime} seconds"
