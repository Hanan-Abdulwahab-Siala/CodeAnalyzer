#!/bin/bash -l

#SBATCH --job-name=mamba-analyzer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/mamba-%j.out
#SBATCH --error=/scratch/users/%u/mamba-%j.err

export PYTHONNOUSERSITE=1

set -e

echo "========================================"
echo "Mamba Code Analyzer - KCL GPU Job"
echo "========================================"

echo
echo "Compute node:"
hostname

# --------------------------------------------------
# Check GPU
# --------------------------------------------------

echo
echo "Checking GPU..."

if ! nvidia-smi >/dev/null 2>&1; then
    echo
    echo "ERROR: No GPU is available on this node."
    echo "The job will not continue."
    echo
    exit 1
fi

echo
echo "GPU available:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --------------------------------------------------
# Load CUDA
# --------------------------------------------------

echo
echo "Loading CUDA..."
module load cuda

# --------------------------------------------------
# Project
# --------------------------------------------------

cd "$HOME/Mamba-Code-Analyzer"

source .venv/bin/activate

# --------------------------------------------------
# Python
# --------------------------------------------------

echo
echo "Python:"
python --version

echo
echo "Python executable:"
which python

# --------------------------------------------------
# PyTorch / CUDA check
# --------------------------------------------------

echo
echo "Checking PyTorch CUDA..."

python -c "
import torch

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)

if not torch.cuda.is_available():
    print()
    print('ERROR: PyTorch cannot access the allocated GPU.')
    print('The job will not continue.')
    raise SystemExit(1)

print('GPU:', torch.cuda.get_device_name(0))
"

# --------------------------------------------------
# Run analyzer
# --------------------------------------------------

echo
echo "Running inference..."
python analyze.py input/sample.txt

# --------------------------------------------------
# Finished
# --------------------------------------------------

echo
echo "========================================"
echo "Job completed successfully"
echo "========================================"
