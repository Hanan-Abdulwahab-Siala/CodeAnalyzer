# Code Analyzer

A GPU-powered application for analyzing Mamba/Python code using a fine-tuned Mistral LLM.

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
Code-Analyzer/
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
│   └── sample.py
│
├── output/
│   └── output.txt
│
└── KCL/
    └── run_kcl_analyze.sh
    └── run_kcl_gradio.sh

```

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
git clone https://github.com/HA-Siala/Code-Analyzer.git
cd Code-Analyzer
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

### 1. [Running via Gradio Web Interface](./documentation/README1.md)
### 2. [Running from the Command-Line](./documentation/README2.md)
### 3. [Running Using KCL CREATE HPC Workflow with Gradio](./documentation/README3.md)
### 4. [README 4](./documentation/README4.md)





### 4) Running using KCL CREATE HPC Workflow without Gradio

We have an optional KCL CREATE script for running the project in an HPC/GPU environment without using the Gradio interface.

#### 1. Connect to KCL HPC

From your local computer, connect to the KCL HPC login node:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```

After connecting, you should see a shell prompt on the HPC login node.
k12345@arc-hpc-login3:~$

---

#### 2. Go to the Project Directory

Move into the Code Analyzer project:

```bash
cd ~/Code-Analyzer
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

#### 3. Verify the Input File Exists

Check the input file:

```bash
ls -lh input/sample.txt
or
ls -lh input/sample.py
```

You can also test:

```bash
cat input/sample.txt
or
cat input/sample.py
```

---

#### 4. Check the SLURM script and make the Script Executable

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

#### 5. Submit the SLURM Job

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

#### 6. Check Whether the Job Is Running

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
37143242   gpu         mamba-   k12345  R    00:05      1   erc-hpc-comp035
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

#### 7. Monitor the Job Continuously

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

#### 8. Monitor the Output File

The SLURM script contains:

```bash
#SBATCH --output=/scratch/users/%u/mambapy-%j.out
```

`%j` is automatically replaced with the job ID. For example, if the JOBID is `37143242`, the output file is:

```
/scratch/users/$USER/mambapy-37143242.out
```

You can view it with:

```bash
cat /scratch/users/$USER/mambapy-37143242.out
```

To monitor it live:

```bash
tail -f /scratch/users/$USER/mambapy-37143242.out
```

Press `Ctrl + C` to stop monitoring.

---

#### 9. Monitor Errors

The SLURM script contains:

```bash
#SBATCH --error=/scratch/users/%u/mambapy-%j.err
```

For job `37143242`, the error file is:

```
/scratch/users/$USER/mambapy-37143242.err
```

View it:

```bash
cat /scratch/users/$USER/mambapy-37143242.err
```

Or monitor it live:

```bash
tail -f /scratch/users/$USER/mambapy-37143242.err
```

If the file is empty, that is usually a good sign.

---

#### 10. Check the Generated Results

Finally, if everything succeeds:

```
========================================
Job completed
========================================
```

Check the project directory:

```bash
cd ~/Code-Analyzer
```

Then:

```bash
ls -lh output/
```

To view the output:
```bash
cat output/output.txt
```

Finally:

```bash
sacct -j 37143242
```

This allows the Mamba/Python Code Analyzer to run as a GPU-accelerated SLURM job on KCL HPC while giving you several ways to monitor its progress.

---

**Note**
You can modify the following line in the KCL/run_kcl_analyze.sh file as previously mentioned.

---
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

## Supported Models

The project uses models hosted on Hugging Face. You may need to make sure the required model repositories are accessible from your environment.

Model identifiers used by the project include:

### Base Model

```
mistralai/Mistral-7B-v0.3
```
### LoRA Adapter Models
#### Mistral
`Mamba`  
- **Version 1:** `HA-Siala/Mamba-v0.1`
- **Version 2:** `HA-Siala/Mamba-v0.2`
  
`Python`
- **Version 1:** 'HA-Siala/Detect-Flaws-v0.1'
- **Version 2:** 'HA-Siala/Detect-Flaws-v0.2'
- **Version 1:** 'HA-Siala/RefactoringPy-v0.1'
#### DeepSeek

### Full Models
#### Mistral
`Mamba`  
- **Version 1:** `HA-Siala/Mamba-full-v0.1`
- **Version 2:** `HA-Siala/Mamba-full-v0.2`

`Python`
- **Version 1:** 'HA-Siala/Detect-Flaws-full-v0.1'
- **Version 2:** 'HA-Siala/Detect-Flaws-full-v0.2'
- **Version 1:** 'HA-Siala/RefactoringPy-full-v0.1'
#### DeepSeek
The full-model workflow loads the complete checkpoint directly.

Please review the applicable model licenses and terms before redistributing model files or using them commercially.

---

## License

MIT License

---

## Contact

Student: hanan.siala@kcl.ac.uk
Supervisor: kevin.lano@kcl.ac.uk

Code Analyzer project.
