# Mamba Code Analyzer

A GPU-powered application for analyzing Mamba code using fine-tuned Mistral LLM.

The project provides:

- A command-line analyzer
- A Gradio web interface
- LoRA adapter and full-model options
- Automatic GPU/CPU hardware detection
- Optional KCL CREATE HPC scripts for the project author's GPU workflow

---

## Project Structure

A typical project structure is:

```text
Mamba-Code-Analyzer/
│
├── README.md
├── requirements.txt
│
├── analyze.py
├── app.py
├── model_service.py
│
├── input/
│   └── sample.txt
│
├── output/
│   └── output.txt
│
└── KCL/
    └── run_kcl_analyze.sh
    └── run_kcl_gradio.sh

```

---

## Supported Models

The project supports both LoRA adapters and full model checkpoints.

### Base Model

The LoRA versions use:

```
mistralai/Mistral-7B-v0.3
```

### LoRA Adapter Models

- **Version 1:** `HA-Siala/Mamba-v0.1`
- **Version 2:** `HA-Siala/Mamba-v0.2`

### Full Models

- **Version 1:** `HA-Siala/Mamba-full-v0.1`
- **Version 2:** `HA-Siala/Mamba-full-v0.2`

The full-model workflow loads the complete checkpoint directly.

---

## Requirements

Recommended:

- Python 3.10+
- PyTorch
- Transformers
- PEFT
- Accelerate
- SentencePiece
- Safetensors
- Protobuf
- Huggingface_hub
- Gradio
- NVIDIA GPU with CUDA support for GPU inference

---

## Installation

Clone the repository:

```bash
git clone https://github.com/HA-Siala/Mamba-Code-Analyzer.git
cd Mamba-Code-Analyzer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
. .venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---
## Running

### 1) Running by the Command-Line 

If the project contains a command-line entry point, run it according to the project's entry script.

For example:

```bash
python analyze.py
```

The application should display hardware information when it starts.

Example:

```text
============================================================
MAMBA CODE ANALYZER - HARDWARE
============================================================
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
PyTorch        : 2.x.x
CUDA version   : 12.x
Dtype          : torch.bfloat16
============================================================
```

---

### 2) Running by Gradio Web Interface

If the project includes a Gradio interface, start the application using:

```bash
python app.py
```

The terminal will provide a local URL, usually similar to:

```
http://127.0.0.1:7860
```

Open that address in a web browser.

The Gradio interface can be used to:

- Select the model version
- Select LoRA or full model
- Enter Mamba/Python code
- Run the analyze
- View detected flaws
- View refactored versions
- View inference information

---

### 3) Running by using KCL CREATE HPC Workflow without Gradio

We have optional KCL CREATE scripts for running the project in an HPC/GPU environment without using Gradio interface

#### 1. Connect to KCL HPC

From your local computer, connect to the KCL HPC login node:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```

After connecting, you should see a shell prompt on the HPC login node.

---

#### 2. Go to the Project Directory

Move into the Mamba Code Analyzer project:

```bash
cd ~/Mamba-Code-Analyzer
```

Check that the project is there:

```bash
ls
```

You should see files such as:

```
analyze.py
input/
.venv/
```

You can also check your current directory:

```bash
pwd
```

---

#### 3. Check the Python Virtual Environment

The project should contain a Python virtual environment:

```bash
ls .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Check Python:

```bash
python --version
```

Check where Python is coming from:

```bash
which python
```

It should point to something similar to:

```
.../Mamba-Code-Analyzer/.venv/bin/python
```

---

#### 4. Verify the Input File Exists

Check the input file:

```bash
ls -lh input/sample.txt
```

You can also test:

```bash
cat input/sample.txt
```
---

---

#### 5. Check SLURM script and Make the Script Executable

Run:

```bash
cat KCL/run_kcl_analyze.sh
```

Then:

```bash
chmod +x KCL/run_kcl_analyze.sh
```

Check the file:

```bash
ls -l KCL/run_kcl_analyze.sh
```

You should see executable permissions, for example:

```
-rwxr-xr-x ... run_kcl_analyze.sh
```

---

#### 6. Submit the SLURM Job

Submit the script using:

```bash
sbatch KCL/run_kcl_analyze.sh
```

You should receive something similar to:

```
Submitted batch job 37143242
```

The number is the **JOBID**. For example:

```
JOBID=37143242
```

Your JOBID will be different each time you submit a new job.

---

#### 7. Check Whether the Job Is Running

Use:

```bash
squeue -j 37143242
```

Or check all your jobs:

```bash
squeue -u $USER
```

Example:

```
JOBID      PARTITION   NAME             USER       ST   TIME   NODES   NODELIST(REASON)
37143242   gpu         mamba-analyzer   k20122072  R    00:05      1   erc-hpc-comp035
```

The important column is **ST**. Common states include:

| State | Meaning              |
|-------|-----------------------|
| R     | Running                |
| PD    | Pending / waiting for resources |
| CG    | Completing              |
| CD    | Completed               |
| F     | Failed                  |
| CA    | Cancelled               |

If you see `R`, the job is currently running.

---

#### 8. Monitor the Job Continuously

You can monitor the job every 2 seconds:

```bash
watch -n 2 squeue -j 37143242
```

Press `Ctrl + C` to stop `watch`.

If `watch` is not available, simply run:

```bash
squeue -j 37143242
```

again whenever you want to check the status.

---

#### 9. Monitor the Output File

The SLURM script contains:

```bash
#SBATCH --output=/scratch/users/%u/mamba-%j.out
```

`%j` is automatically replaced with the job ID. For example, if the JOBID is `37143242`, the output file is:

```
/scratch/users/$USER/mamba-37143242.out
```

You can view it with:

```bash
cat /scratch/users/$USER/mamba-37143242.out
```

To monitor it live:

```bash
tail -f /scratch/users/$USER/mamba-37143242.out
```

Press `Ctrl + C` to stop monitoring.

---

#### 10. Monitor Errors

The SLURM script contains:

```bash
#SBATCH --error=/scratch/users/%u/mamba-%j.err
```

For job `37143242`, the error file is:

```
/scratch/users/$USER/mamba-37143242.err
```

View it:

```bash
cat /scratch/users/$USER/mamba-37143242.err
```

Or monitor it live:

```bash
tail -f /scratch/users/$USER/mamba-37143242.err
```

If the file is empty, that is usually a good sign.

---

#### 11. Check the Generated Results

Finally, if everything succeeds:

```
========================================
Job completed
========================================
```

Check the project directory:

```bash
cd ~/Mamba-Code-Analyzer
```

Then:

```bash
ls -lh output/
```

Finally:

```bash
sacct -j 37143242
```

---

This allows the Mamba Code Analyzer to run as a GPU-accelerated SLURM job on KCL HPC while giving you several ways to monitor its progress.

If you are not using the KCL CREATE environment, you can ignore the HPC scripts and run the project using the normal Python environment.

---

## Expected Model Response

The model is instructed to return a Python dictionary containing exactly two top-level keys:

```python
{
    "Flaws": ...,
    "Refactored Versions": ...
}
```

For example:

```python
{
    "Flaws": [
        {
            "Flaw": "Example flaw",
            "Explanation": "Explanation of the detected problem."
        }
    ],
    "Refactored Versions": [
        "def corrected_function():\n    pass"
    ]
}
```

The actual response depends on the model and the submitted code. The project includes a function for extracting the dictionary from the model output:

```python
extract_clean_dict()
```

### Formatting Output

The function:

```python
format_output()
```

converts the parsed dictionary into readable output.

Example:

```text
Flaws:
- Example flaw: Explanation of the problem.

Refactored versions code:
def corrected_function():
    ...
```

---

## Hardware Information

The application provides hardware information through:

```python
get_hardware_info()
```

Example GPU output:

```text
GPU: NVIDIA ...
CUDA: 12.x
PyTorch: 2.x.x
VRAM: 20.50 GB free / 24.00 GB total
Compute capability: (8, 9)
Dtype: torch.bfloat16
```
---

## Model and Repository Access

The project uses models hosted on Hugging Face. You may need to make sure the required model repositories are accessible from your environment.

Model identifiers used by the project include:

- `mistralai/Mistral-7B-v0.3`
- `HA-Siala/Mamba-v0.1`
- `HA-Siala/Mamba-v0.2`
- `HA-Siala/Mamba-full-v0.1`
- `HA-Siala/Mamba-full-v0.2`

Please review the applicable model licenses and terms before redistributing model files or using them commercially.

---

## License

MIT License

---

## Contact

Student: hanan.siala@kcl.ac.uk
Supervisor: kevin.lano@kcl.ac.uk

Mamba Code Analyzer project.
