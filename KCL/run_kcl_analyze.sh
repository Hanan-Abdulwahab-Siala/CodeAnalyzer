#!/bin/bash -l

#SBATCH --job-name=mamba-analyzer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/mambapy-%j.out
#SBATCH --error=/scratch/users/%u/mambapy-%j.err

export PYTHONNOUSERSITE=1

set -e

echo "========================================"
echo "Unified Code Analyzer - KCL GPU Job"
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

cd "$HOME/Code-Analyzer"

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
# Run Analyzer
# --------------------------------------------------

echo
echo "Running inference..."

# --------------------------------------------------
# Choose ONE command below starting by python and uncomment it.
# --------------------------------------------------

# --------------------------------------------------
# MISTRAL - MAMBA
# --------------------------------------------------

## Mistral + Mamba + Version 1 + LoRA Adapter

# python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 1 --model-type "LoRA Adapter" 

# --------------------------------------------------

## Mistral + Mamba + Version 2 + LoRA Adapter

python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 2 --model-type "LoRA Adapter" 

# --------------------------------------------------

## Mistral + Mamba + Version 1 + Full Model

# python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 1 --model-type "Full Model" 

# --------------------------------------------------

## Mistral + Mamba + Version 2 + Full Model

# python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 2 --model-type "Full Model" 

# --------------------------------------------------
# MISTRAL - PYTHON FLAW DETECTION
# --------------------------------------------------

## Mistral + Python + Flaw Detection + Version 1 + LoRA Adapter

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Flaw Detection" --model-version 1 --model-type "LoRA Adapter" 

# --------------------------------------------------

## Mistral + Python + Flaw Detection + Version 2 + LoRA Adapter

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Flaw Detection" --model-version 2 --model-type "LoRA Adapter" 

# --------------------------------------------------

## Mistral + Python + Flaw Detection + Version 1 + Full Model

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Flaw Detection" --model-version 1 --model-type "Full Model" 

# --------------------------------------------------

## Mistral + Python + Flaw Detection + Version 2 + Full Model

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Flaw Detection" --model-version 2 --model-type "Full Model" 

# --------------------------------------------------
# MISTRAL - PYTHON REFACTORING
# --------------------------------------------------

## Mistral + Python + Refactoring + Version 1 + LoRA Adapter

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Refactoring" --model-version 1 --model-type "LoRA Adapter" 

# --------------------------------------------------

## Mistral + Python + Refactoring + Version 1 + Full Model

# python analyze.py input/sample.py --language Python --model-family Mistral --task "Refactoring" --model-version 1 --model-type "Full Model" 

# --------------------------------------------------
# DEEPSEEK - MAMBA
# --------------------------------------------------

## DeepSeek + Mamba + Version 1 + LoRA Adapter

# python analyze.py input/sample.txt --language Mamba --model-family DeepSeek --model-version 1 --model-type "LoRA Adapter" 

# --------------------------------------------------

## DeepSeek + Mamba + Version 1 + Full Model

# python analyze.py input/sample.txt --language Mamba --model-family DeepSeek --model-version 1 --model-type "Full Model" 

# --------------------------------------------------
# DEEPSEEK - PYTHON
# --------------------------------------------------

## DeepSeek + Python + Version 1 + LoRA Adapter

# python analyze.py input/sample.py --language Python --model-family DeepSeek --model-version 1 --model-type "LoRA Adapter" 

# --------------------------------------------------

## DeepSeek + Python + Version 1 + Full Model

# python analyze.py input/sample.py --language Python --model-family DeepSeek --model-version 1 --model-type "Full Model" 

# --------------------------------------------------

echo
echo "========================================"
echo "Job completed successfully"
echo "========================================"
# --------------------------------------------------





