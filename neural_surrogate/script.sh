#!/bin/bash
#SBATCH --job-name ns_uifo
#SBATCH --output neural_surrogate/logs/%x-%j.out
#SBATCH --error neural_surrogate/logs/%x-%j.err
#SBATCH --partition 2080-galvani
#SBATCH --gpus=8
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

echo "Working directory: $PWD"
echo "Host: $(hostname)"
echo "Started at: $(date)"

mkdir -p neural_surrogate/logs

# External environment to load on the cluster.
ENV_DIR="/mnt/lustre/home/krenn/klz468/repos/envs/dfbench_env"
source "$ENV_DIR/bin/activate"

# External directory containing campaign .h5 files.
DATA_DIR="/mnt/lustre/work/krenn/klz077/datasets/UIFOs"

# Checkpoint path for saving the best model
mkdir -p neural_surrogate/checkpoints
CHECKPOINT_PATH="neural_surrogate/checkpoints/best_model.pt"

LOSS_KEY="${LOSS_KEY:-loss_senspow}"
EPOCHS="${EPOCHS:-250}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
TOPOLOGY_DIM="${TOPOLOGY_DIM:-128}"
SEED="${SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
DEVICE="${DEVICE:-cuda}"
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
  --seed 1 \
  --val-fraction "$VAL_FRACTION" \
  --device "$DEVICE" \
  --multi-gpu "$MULTI_GPU" \
  --checkpoint-path "$CHECKPOINT_PATH"

end=$(date +%s)
runtime=$((end - start))

echo "Finished at: $(date)"
echo "Runtime: ${runtime} seconds"
