#!/bin/bash -l

#SBATCH --job-name=code-analyzer-gradio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/mambapy-gradio-%j.out
#SBATCH --error=/scratch/users/%u/mambapy-gradio-%j.err

export PYTHONNOUSERSITE=1

set -e

echo "========================================"
echo "Unified Code Analyzer - Gradio"
echo "========================================"

# --------------------------------------------------

echo
echo "Compute node:"
hostname

echo
echo "Slurm job information:"
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

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
echo "Allocated GPU:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# --------------------------------------------------
echo
echo "Loading CUDA..."
module load cuda

# --------------------------------------------------

cd "$HOME/Code-Analyzer"

source ~/venvs/bin/activate
# --------------------------------------------------

echo
echo "Python:"
python --version

echo
echo "Python executable:"
which python

# --------------------------------------------------

echo
echo "Checking PyTorch CUDA..."

python -c "
import torch
import os

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))

if not torch.cuda.is_available():
   print()
   print('ERROR: PyTorch cannot access the allocated GPU.')
   print('The job will not continue.')
   raise SystemExit(1)

print('GPU count visible to PyTorch:', torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
   print(f'GPU {i}:', torch.cuda.get_device_name(i))
   print(f'GPU {i} BF16 supported:', torch.cuda.is_bf16_supported(i))
"

# --------------------------------------------------
export GRADIO_SERVER_PORT=7860

echo
echo "========================================"
echo "Starting Gradio"
echo "========================================"

echo
echo "Node:"
hostname

echo
echo "Port:"
echo "$GRADIO_SERVER_PORT"

echo
echo "Run this command on your LOCAL machine:"
echo
echo "ssh -L 7860:$(hostname):7860 $USER@$(hostname)"
echo
echo "Then open:"
echo
echo "http://localhost:7860"
echo
###############
NODE=$(hostname)
echo "$NODE"
###############
# --------------------------------------------------
python app.py
# --------------------------------------------------
