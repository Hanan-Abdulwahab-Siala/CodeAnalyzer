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

set -e

echo "========================================"
echo "Mamba Code Analyzer - KCL Gradio"
echo "========================================"

echo "Compute node:"
hostname

echo
echo "GPU:"
nvidia-smi

module load cuda

cd "$HOME/mamba-code-analyzer"

source .venv/bin/activate

# Choose a port for this Gradio session.
export GRADIO_SERVER_PORT=7860

# The exact interface will be determined on the compute node.
export GRADIO_SERVER_NAME=$(hostname -I | tr ' ' '\n' | grep '^10\.211\.4\.' | head -n 1)

echo
echo "Gradio server:"
echo "Address: $GRADIO_SERVER_NAME"
echo "Port:    $GRADIO_SERVER_PORT"

echo
echo "Starting Gradio..."
python app.py
