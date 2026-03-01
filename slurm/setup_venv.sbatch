#!/bin/bash
#SBATCH --job-name=setup-venv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --exclude=galvani-cn115,galvani-cn111,galvani-cn108
#SBATCH -o /mnt/lustre/work/hennig/hmx892/logs/%j_setup_venv.out

# Setup GPU venv for JAX with CUDA support
set -e  # Exit on error

REPO_DIR="/mnt/lustre/work/hennig/hmx892/repos/updated-differometor/Differometor-Benchmark"
VENV_NAME=".venv"

echo "Setting up GPU venv on node: $(hostname)"
cd ${REPO_DIR}

# Create fresh venv and sync with all extras
echo "Creating ${VENV_NAME} and syncing dependencies..."
uv sync --group cuda12 --group wandb

# Verify installation
echo "Verifying JAX GPU installation..."
source ${VENV_NAME}/bin/activate
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
devices = jax.devices()
gpu_found = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
if gpu_found:
    print('GPU successfully detected!')
else:
    print('WARNING: No GPU detected!')
    exit(1)
"

echo "GPU venv setup complete!"