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

echo
echo "GPU:"
nvidia-smi

echo
echo "Loading CUDA..."
module load cuda

cd "$HOME/Mamba-Code-Analyzer"

source .venv/bin/activate

echo
echo "Python:"
python --version

echo
echo "Python executable:"
which python

echo
echo "PyTorch:"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    raise RuntimeError('CUDA is not available')
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

echo
echo "Starting Gradio..."

python app.py
