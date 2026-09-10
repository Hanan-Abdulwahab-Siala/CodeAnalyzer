#!/bin/bash -l

#SBATCH --job-name=mamba-analyzer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/%u/mamba-%j.out
#SBATCH --error=/scratch/users/%u/mamba-%j.err


set -e


# ============================================================
# CUDA
# ============================================================

module load cuda


# ============================================================
# PROJECT
# ============================================================

cd "$HOME/mamba-code-analyzer"


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

source .venv/bin/activate


# ============================================================
# GPU INFORMATION
# ============================================================

echo "========================================"

echo "HOSTNAME:"

hostname

echo "========================================"

echo "GPU:"

nvidia-smi

echo "========================================"


# ============================================================
# PYTORCH TEST
# ============================================================

python -c "

import torch

print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        'GPU:',
        torch.cuda.get_device_name(0)
    )

    print(
        'CUDA:',
        torch.version.cuda
    )

"


# ============================================================
# RUN ANALYSIS
# ============================================================

python analyze.py input/sample.txt
