#!/bin/bash -l

#SBATCH --job-name=mamba-gradio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/mamba-gradio-%j.out
#SBATCH --error=/scratch/users/%u/mamba-gradio-%j.err

export PYTHONNOUSERSITE=1

set -e

echo "========================================"
echo "Mamba Code Analyzer - KCL Gradio"
echo "========================================"

echo
echo "Compute node:"
hostname

# ============================================================
# GPU CHECK
# ============================================================

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
echo "GPU detected:"
nvidia-smi

# ============================================================
# Load CUDA
# ============================================================

echo
echo "Loading CUDA..."
module load cuda

# ============================================================
# Activate environment
# ============================================================

cd "$HOME/Mamba-Code-Analyzer"

source .venv/bin/activate

# ============================================================
# Python
# ============================================================

echo
echo "Python:"
python --version

echo
echo "Python executable:"
which python

# ============================================================
# PyTorch / CUDA CHECK
# ============================================================

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

# ============================================================
# Gradio configuration
# ============================================================

export GRADIO_SERVER_PORT=7860
export GRADIO_SERVER_NAME=0.0.0.0

echo
echo "========================================"
echo "Gradio configuration"
echo "========================================"
echo "Host: $GRADIO_SERVER_NAME"
echo "Port: $GRADIO_SERVER_PORT"

echo
echo "Compute node:"
hostname

echo
echo "SSH tunnel command:"
echo "ssh -m hmac-sha2-512 -L 7861:$(hostname):7860 k20122072@hpc.create.kcl.ac.uk"

echo
echo "Then open:"
echo "http://localhost:7861"

echo
echo "========================================"

# ============================================================
# Start Gradio
# ============================================================

echo
echo "Starting Gradio..."

python app.py
