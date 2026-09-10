# Mamba Code Analyzer

A local GPU-powered application for analyzing Mamba code using fine-tuned Mamba/Mistral language models.

The project provides:

- A command-line analyzer
- A Gradio web interface
- LoRA adapter and full-model options
- Automatic GPU/CPU hardware detection
- Local inference without a central inference server
- Optional KCL CREATE HPC scripts for the project author's GPU workflow

---

## Important: How Inference Works

This project **does not provide a central GPU server**.

Every user runs model inference using their **own available computing resources**.

### Public Users

A user who clones this repository runs inference on their own computer or GPU provider.

```text
User's computer / GPU provider
            |
            v
        app.py
            |
            v
    model_service.py
            |
            v
       User's GPU
```

The public application does **not** connect to the author's KCL GPU.

### Project Author on KCL CREATE

The project author can run the same application on a GPU allocated through KCL CREATE using Slurm.

```text
KCL CREATE
    |
    v
Slurm GPU allocation
    |
    v
Allocated KCL GPU
    |
    v
model_service.py
    |
    v
Model inference
```

The KCL GPU is therefore used only by the author's KCL jobs and is **not a public inference server**.

---

## Requirements

### For Public Users

- Python 3.11
- Git
- NVIDIA GPU recommended
- NVIDIA driver with CUDA support
- Sufficient GPU VRAM
- Internet connection for downloading the required models

A CUDA-capable NVIDIA GPU is strongly recommended because the project uses large language models.

CPU execution may be possible, but inference can be extremely slow or impractical for large models.

### For KCL CREATE

The KCL workflow additionally requires:

- A KCL CREATE account with HPC access
- SSH access to the KCL HPC system
- Slurm
- Access to the KCL GPU partition
- A suitable Python virtual environment

The KCL-specific setup is described separately below.

---

# Installation for Public Users

## 1. Clone the Repository

Replace `YOUR_REPOSITORY_URL` with the actual GitHub repository URL.

```bash
git clone YOUR_REPOSITORY_URL
cd mamba-code-analyzer
```

---

## 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

### Windows Command Prompt

```bat
.venv\Scripts\activate
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

After activation, the terminal should show something similar to:

```text
(.venv)
```

---

# Install PyTorch

PyTorch is installed separately from the project's main requirements because the appropriate PyTorch package depends on the user's operating system, GPU, NVIDIA driver, and CUDA environment.

Install a CUDA-compatible PyTorch version appropriate for your system.

After installation, verify the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A working CUDA installation should report something similar to:

```text
PyTorch: <version>
CUDA available: True
GPU: <your NVIDIA GPU>
```

If it reports:

```text
CUDA available: False
```

the application may fall back to CPU, but large-model inference may be extremely slow or impractical.

You can also check the NVIDIA driver directly with:

```bash
nvidia-smi
```

---

# Install Project Dependencies

After installing PyTorch:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the application and model-related Python dependencies.

PyTorch is intentionally not pinned in `requirements.txt` so that users can install a PyTorch build appropriate for their own hardware and environment.

---

# Run the Command-Line Analyzer

The command-line analyzer accepts a text file containing Mamba code.

A sample input file is included in:

```text
input/sample.txt
```

Run:

```bash
python analyze.py input/sample.txt
```

The generated analysis is written to:

```text
output/output.txt
```

---

# Run the Gradio Web Application

Start the local Gradio interface:

```bash
python app.py
```

The application normally runs at:

```text
http://127.0.0.1:7860
```

Open the address in a web browser.

The interface provides:

- Hardware information
- Model selection
- Model version selection
- File upload
- Mamba code input
- Code analysis
- Formatted analysis results
- Downloadable output

The application automatically detects whether CUDA is available.

---

# Model Options

The application supports two model modes.

## LoRA Adapter

The LoRA configuration uses:

```text
mistralai/Mistral-7B-v0.3
```

as the base model and loads a Mamba LoRA adapter.

Available LoRA adapters:

```text
HA-Siala/Mamba-v0.1
HA-Siala/Mamba-v0.2
```

## Full Model

The full-model configuration uses:

```text
HA-Siala/Mamba-full-v0.1
HA-Siala/Mamba-full-v0.2
```

The required model files are downloaded from Hugging Face when they are needed.

The first run may therefore take significantly longer than subsequent runs.

---

# GPU Usage

The project does not hard-code a particular GPU.

The application checks whether CUDA is available through PyTorch.

When a CUDA-capable GPU is available, the model is loaded for GPU inference.

For example:

```text
Public User A
     |
     v
Their NVIDIA GPU
```

is completely independent from:

```text
Project Author
     |
     v
KCL CREATE allocated GPU
```

No public user is given access to the author's KCL GPU.

---

# KCL CREATE HPC Usage

The repository contains two KCL-specific Slurm scripts:

```text
run_kcl_analyze.sh
run_kcl_gradio.sh
```

These scripts are intended for the **project author's KCL CREATE workflow**.

## Important

**Public users do not need to run these scripts.**

They should follow the normal installation and usage instructions above.

The KCL scripts request GPUs from the KCL Slurm scheduler and are specific to the KCL HPC environment.

---

## KCL Command-Line Analysis

After configuring the Python environment on KCL, submit the analysis job with:

```bash
sbatch run_kcl_analyze.sh
```

Check the job status with:

```bash
squeue -u $USER
```

The Slurm output and error files are configured by the script.

The job requests a GPU and runs:

```text
analyze.py
    |
    v
model_service.py
    |
    v
Allocated KCL GPU
```

---

## KCL Gradio Application

The Gradio interface can also be started inside a KCL GPU job:

```bash
sbatch run_kcl_gradio.sh
```

The Gradio service runs on the allocated KCL compute node.

The service should **not** be exposed as a public Internet service.

Instead, use SSH port forwarding from your own computer to access the private Gradio application.

The exact tunnel command depends on the compute node and port assigned to the job.

---

# GitHub Actions

GitHub Actions is used for automated software validation.

The GitHub Actions workflow performs CPU-based checks such as:

- Installing Python
- Installing project dependencies
- Installing CPU-only PyTorch
- Compiling Python files
- Checking the repository structure
- Checking the sample input
- Creating a CI output artifact

GitHub Actions does **not** perform model inference.

It does not require an NVIDIA GPU.

Therefore:

```text
GitHub Actions
      |
      v
CPU checks only
```

while actual inference is performed separately:

```text
User's computer
      |
      v
User's GPU
```

or:

```text
KCL Slurm job
      |
      v
Allocated KCL GPU
```

---

# Repository Structure

```text
mamba-code-analyzer/
│
├── .github/
│   └── workflows/
│       └── checks.yml
│
├── input/
│   └── sample.txt
│
├── output/
│   └── .gitkeep
│
├── analyze.py
├── app.py
├── app_test.py
├── model_service.py
├── requirements.txt
├── run_kcl_analyze.sh
├── run_kcl_gradio.sh
└── README.md
```

---

# KCL Scripts vs Public Usage

The two `.sh` files are **not public inference scripts**.

| File | Purpose | Public Users |
|---|---|---|
| `analyze.py` | Command-line inference | Yes |
| `app.py` | Local Gradio application | Yes |
| `model_service.py` | Model loading and inference | Yes |
| `requirements.txt` | Python dependencies | Yes |
| `run_kcl_analyze.sh` | KCL Slurm GPU analysis | No |
| `run_kcl_gradio.sh` | KCL Slurm Gradio application | No |
| `.github/workflows/checks.yml` | Automated CPU checks | Automatic |

Public users should normally use:

```bash
python analyze.py input/sample.txt
```

or:

```bash
python app.py
```

They do **not** need to use:

```bash
sbatch run_kcl_analyze.sh
```

or:

```bash
sbatch run_kcl_gradio.sh
```

---

# Troubleshooting

## CUDA Is Not Available

Check PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the NVIDIA driver:

```bash
nvidia-smi
```

If CUDA is unavailable, check:

- NVIDIA driver installation
- PyTorch installation
- CUDA compatibility
- GPU visibility
- Whether another process is using the GPU

---

## Out of GPU Memory

The models require substantial GPU memory.

If you receive a CUDA out-of-memory error:

1. Check GPU usage:

```bash
nvidia-smi
```

2. Close other GPU applications.
3. Make sure the intended model configuration is selected.
4. Use a GPU with more available VRAM if necessary.

---

## First Run Is Slow

The first run may be slow because the required model files must be downloaded from Hugging Face.

Later runs can reuse the locally cached model files.

---

## Gradio Does Not Start

Check that the virtual environment is activated:

```bash
python --version
```

Check that Gradio is installed:

```bash
python -c "import gradio; print(gradio.__version__)"
```

Then start the application again:

```bash
python app.py
```

---

# Development Checks

The Python files can be compiled manually with:

```bash
python -m py_compile analyze.py
python -m py_compile app.py
python -m py_compile app_test.py
python -m py_compile model_service.py
```

Run the application test with:

```bash
python app_test.py
```

---

# Data and Model Downloads

The application downloads model files from Hugging Face when required.

The models are not hosted by this repository.

Users should ensure that they have:

- Internet access
- Enough local disk space
- Enough GPU memory for the selected model
- Permission to access and use the referenced models under their respective licenses

---

# Security and Deployment

This project is intended primarily for **local execution**.

The Gradio application should normally be bound to the local machine when running locally.

Do not expose the application to the public Internet unless you have intentionally configured and secured the deployment.

The KCL Gradio workflow is intended to remain private to the KCL environment and should be accessed through SSH tunneling.

---

# License

Add the project's license information here.

If this project uses an open-source license, replace this section with the appropriate license name and license text or a reference to the license file.
