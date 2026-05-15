#!/bin/bash
#SBATCH --job-name ns_uifo
#SBATCH --output neural_surrogate/logs/%x-%j.out
#SBATCH --error neural_surrogate/logs/%x-%j.err
#SBATCH --partition 2080-galvani
#SBATCH --gpus=8
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-task=30
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

# Checkpoint directory for saving the best model during training.
mkdir -p neural_surrogate/checkpoints
CHECKPOINT_PATH="neural_surrogate/checkpoints"

RUN_MODE="${RUN_MODE:-train}"
LOSS_KEY="${LOSS_KEY:-loss_senspow}"
EPOCHS="${EPOCHS:-250}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
TOPOLOGY_DIM="${TOPOLOGY_DIM:-128}"
D_MODEL="${D_MODEL:-64}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
MAX_INPUT_DIM="${MAX_INPUT_DIM:-1024}"
SEED="${SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
DEVICE="${DEVICE:-cuda}"
MULTI_GPU="${MULTI_GPU:-ddp}"
TOPOLOGY_STRATEGY="${TOPOLOGY_STRATEGY:-hashing}"
PARAMETER_STRATEGY="${PARAMETER_STRATEGY:-bounds}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
if [[ "$MULTI_GPU" == "ddp" ]]; then
  DEFAULT_DATASET_WORKERS=$(( (${SLURM_CPUS_PER_TASK:-30} + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
else
  DEFAULT_DATASET_WORKERS="${SLURM_CPUS_PER_TASK:-30}"
fi
DATASET_WORKERS="${DATASET_WORKERS:-$DEFAULT_DATASET_WORKERS}"
INVERSE_CHECKPOINT_PATH="${INVERSE_CHECKPOINT_PATH:-$CHECKPOINT_PATH/${TOPOLOGY_STRATEGY}_${PARAMETER_STRATEGY}_checkpoint.pt}"
INVERSE_OUTPUT_PATH="${INVERSE_OUTPUT_PATH:-neural_surrogate/inverse_design_result.json}"
INVERSE_STEPS="${INVERSE_STEPS:-1000}"
INVERSE_LR="${INVERSE_LR:-1e-2}"
TARGET_LOSS="${TARGET_LOSS:-}"
REFERENCE_INDEX="${REFERENCE_INDEX:-}"

echo "Environment: $ENV_DIR"
echo "Data directory: $DATA_DIR"
echo "Run mode: $RUN_MODE"
echo "Loss key: $LOSS_KEY"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Topology dim: $TOPOLOGY_DIM"
echo "d_model: $D_MODEL"
echo "nhead: $NHEAD"
echo "num_layers: $NUM_LAYERS"
echo "dim_feedforward: $DIM_FEEDFORWARD"
echo "Max input dim: $MAX_INPUT_DIM"
echo "Seed: $SEED"
echo "Validation fraction: $VAL_FRACTION"
echo "Device: $DEVICE"
echo "Multi-GPU mode: $MULTI_GPU"
echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Topology strategy: $TOPOLOGY_STRATEGY"
echo "Parameter strategy: $PARAMETER_STRATEGY"
echo "Dataset workers: $DATASET_WORKERS"
echo "Checkpoint path: $CHECKPOINT_PATH"

start=$(date +%s)

if [[ "$RUN_MODE" == "train" ]]; then
  train_args=(
    --data "$DATA_DIR"
    --loss-key "$LOSS_KEY"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --topology-dim "$TOPOLOGY_DIM"
    --d-model "$D_MODEL"
    --nhead "$NHEAD"
    --num-layers "$NUM_LAYERS"
    --dim-feedforward "$DIM_FEEDFORWARD"
    --max-input-dim "$MAX_INPUT_DIM"
    --seed "$SEED"
    --val-fraction "$VAL_FRACTION"
    --device "$DEVICE"
    --multi-gpu "$MULTI_GPU"
    --checkpoint-path "$CHECKPOINT_PATH"
    --topology-strategy "$TOPOLOGY_STRATEGY"
    --parameter-strategy "$PARAMETER_STRATEGY"
    --dataset-workers "$DATASET_WORKERS"
  )
  if [[ "$MULTI_GPU" == "ddp" ]]; then
    torchrun --standalone --nproc_per_node "$GPUS_PER_NODE" \
      -m neural_surrogate.pipeline "${train_args[@]}"
  else
    python -m neural_surrogate.pipeline "${train_args[@]}"
  fi
elif [[ "$RUN_MODE" == "inverse" ]]; then
  inverse_args=(
    --data "$DATA_DIR"
    --checkpoint-path "$INVERSE_CHECKPOINT_PATH"
    --loss-key "$LOSS_KEY"
    --steps "$INVERSE_STEPS"
    --lr "$INVERSE_LR"
    --topology-dim "$TOPOLOGY_DIM"
    --topology-strategy "$TOPOLOGY_STRATEGY"
    --parameter-strategy "$PARAMETER_STRATEGY"
    --device "$DEVICE"
    --output-path "$INVERSE_OUTPUT_PATH"
    --dataset-workers "$DATASET_WORKERS"
  )
  if [[ -n "$TARGET_LOSS" ]]; then
    inverse_args+=(--target-loss "$TARGET_LOSS")
  fi
  if [[ -n "$REFERENCE_INDEX" ]]; then
    inverse_args+=(--reference-index "$REFERENCE_INDEX")
  fi
  python -m neural_surrogate.inverse_design "${inverse_args[@]}"
else
  echo "Unknown RUN_MODE '$RUN_MODE'. Expected 'train' or 'inverse'." >&2
  exit 2
fi

end=$(date +%s)
runtime=$((end - start))

echo "Finished at: $(date)"
echo "Runtime: ${runtime} seconds"
