#!/bin/bash
#SBATCH --job-name=adam-voyager
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o /mnt/lustre/work/hennig/hmx892/logs/%A_%a_adam_voyager.out

REPO_DIR="/mnt/lustre/work/hennig/hmx892/repos/updated-differometor/Differometor-Benchmark"
VENV_NAME=".venv"

# Load W&B credentials from .env file (copy from .env.template)
ENV_FILE="${REPO_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "Warning: ${ENV_FILE} not found. W&B logging may fail."
    echo "Copy .env.template to .env and fill in your credentials."
fi

echo "Running adam voyager on node: $(hostname)"
cd ${REPO_DIR}
source ${VENV_NAME}/bin/activate

python scripts/thomas_voyager_adam_gd.py \
    --max-evals 20000 \
    --random-seed 42 \
    --learning-rate 0.1 \
    --problem constrained_voyager \
    --use-arcsinh-transform \
    --log-wandb
