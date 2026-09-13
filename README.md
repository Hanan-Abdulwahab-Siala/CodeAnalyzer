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
- **Version 1:** 'HA-Siala/Detect-Flaws-v0.1'
- **Version 2:** 'HA-Siala/Detect-Flaws-v0.2'
- **Version 1:** 'HA-Siala/RefactoringPy-v0.1'

### Full Models

- **Version 1:** `HA-Siala/Mamba-full-v0.1`
- **Version 2:** `HA-Siala/Mamba-full-v0.2`
- **Version 1:** 'HA-Siala/Detect-Flaws-full-v0.1'
- **Version 2:** 'HA-Siala/Detect-Flaws-full-v0.2'
- **Version 1:** 'HA-Siala/RefactoringPy-full-v0.1'

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

### 1) Running by Gradio Web Interface
The Gradio interface can be used to:

- Select the model version
- Select LoRA or full model
- Enter Mamba code
- Run the analyze
- View detected flaws
- View refactored versions
- View inference information

Please follow these instructions:  

#### 1. Connect to GPU provider

From your local computer, connect to the remote server, for example:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```
After connecting, you should see a shell prompt on the HPC login node.
k12345@arc-hpc-login3:~$

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
```

You can also test:

```bash
cat input/sample.txt
```

---

#### 4. Ask for GPU from GPU provider:
For example, in KCL, we use:

```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --time=04:00:00 \
     --cpus-per-task=4 \
     --mem=32G \
     --pty /bin/bash -l
```
Then:

```bash
nvidia-smi
```
Then:

```bash
module load cuda
```

---

#### 5. Check the Python Virtual Environment
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
.../Code-Analyzer/.venv/bin/python
```

---

#### 5. Run the program

Run the Gradio program:

```bash
python analyze.py
```

You should receive something similar to:

```
============================================================
Starting Code Analyzer
Model loading is manual.
Gradio server: http://0.0.0.0:7860
Port: 7860
============================================================
* Running on local URL:  http://0.0.0.0:7860
```
---

#### 6. Get HostName and Open Gradio
To get the hostname, you can open another terminal/SSH window and connect to the HPC again; for example, in KCL we use:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```
Then run:

```bash
hostname
```
You will get something like:
```bash
gpu-node-123
```

Then, create the tunnel by running:

```bash
ssh -m hmac-sha2-512 -L 7861:gpu-node-123:7860 k12345@arc-hpc-login4.create.kcl.ac.uk
```
Replace gpu-node-123 with the hostname you got.

Then open in a browser:
```bash
http://localhost:7861
```

**Important**
Your first terminal must stay running:

Terminal 1
```bash
$ python app.py
```

* Running on local URL: http://0.0.0.0:7860

The second terminal handles the SSH tunnel:

Terminal 2
```bash
ssh -m hmac-sha2-512 -L 7861:gpu-node-123:7860 k12345@arc-hpc-login4.create.kcl.ac.uk
```

---

### 2) Running from the Command-Line 

Please follow these instructions:  

#### 1. Connect to the GPU provider

From your local computer, connect to the remote server, for example:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```
After connecting, you should see a shell prompt on the HPC login node.
k12345@arc-hpc-login3:~$

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

#### 4. Ask for a GPU from the GPU provider:
For example, in KCL, we use:

```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --time=04:00:00 \
     --cpus-per-task=4 \
     --mem=32G \
     --pty /bin/bash -l
```
Then:

```bash
nvidia-smi
```
Then:

```bash
module load cuda
```

---

#### 5. Check the Python Virtual Environment
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
.../Code-Analyzer/.venv/bin/python
```

---

#### 5. Run the program

Run the analyze program using one of the following:
Default values
```bash
python analyze.py input/sample.txt
or
python analyze.py input/sample.py
```

Or (version 1 and LoRA Adapter)
```bash
python analyze.py input/sample.txt --model-version 1 --model-type "LoRA Adapter"
or
python analyze.py input/sample.py --model-version 1 --model-type "LoRA Adapter"
```

Or (version 2 and LoRA Adapter)
```bash
python analyze.py input/sample.txt --model-version 2 --model-type "LoRA Adapter"
or
python analyze.py input/sample.py --model-version 2 --model-type "LoRA Adapter"
```

Or (version 1 and Full Model)
```bash
python analyze.py input/sample.txt --model-version 1 --model-type "Full Model"
or
python analyze.py input/sample.py --model-version 1 --model-type "Full Model"
```

Or (version 2 and Full Model)
```bash
python analyze.py input/sample.txt --model-version 2 --model-type "Full Model"
or
python analyze.py input/sample.py --model-version 2 --model-type "Full Model"
```

You should receive something similar to:

```bash
Starting analyzer...
Model version: 2
Model type: LoRA Adapter
Input characters: 94
Loading model...
Loading model version 2 (LoRA Adapter)...
Checkpoint: HA-Siala/Mamba-v0.2
Dtype: torch.bfloat16
Loading base Mistral model...
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.94s/it]
Loading LoRA adapter...
Model loaded successfully.
Starting inference...
Input tokens: 146
============================================================
GENERATION DIAGNOSTICS
Input tokens: 146
Generated tokens: 327
Maximum generated tokens: 32768
Reached token limit: False
Output characters: 1033
============================================================

Flaws:
   - PF — Performance Fault: ...

Refactored versions code:

Option 1
...

========================================
Inference Metrics
========================================
Model type:                 LoRA Adapter
Model version:              2
Input tokens:               146
Generated tokens:           327
Inference time:             16.442539s
Time per input token:       0.112620s
Time per generated token:   0.050283s
========================================

Output saved to: output/output.txt
```
---

### 3) Running using KCL CREATE HPC Workflow with Gradio

We provide an optional KCL CREATE script to run the project in an HPC/GPU environment using a Gradio interface.

#### 1. Connect to KCL HPC

From your local computer, connect to the KCL HPC login node:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```
After connecting, you should see a shell prompt on the HPC login node.
k12345@arc-hpc-login3:~$

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
cat KCL/run_kcl_gradio.sh
```

Then:

```bash
chmod +x KCL/run_kcl_gradio.sh
```

Check the file:

```bash
ls -l KCL/run_kcl_gradio.sh
```

You should see executable permissions, for example:

```
-rwxr-xr-x ... run_kcl_gradio.sh
```
---

#### 5. Submit the SLURM Job

Submit the script using:

```bash
sbatch KCL/run_kcl_gradio.sh
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
37143242   gpu         mamba-gr   k12345  R    00:05      1   erc-hpc-comp035
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
#SBATCH --output=/scratch/users/%u/mamba-%j.out
```

`%j` is automatically replaced with the job ID. For example, if the JOBID is `37143242`, the output file is:

```
/scratch/users/$USER/mamba-gradio-37143242.out
```

You can view it with:

```bash
cat /scratch/users/$USER/mamba-gradio-37143242.out
```

To monitor it live:

```bash
tail -f /scratch/users/$USER/mamba-gradio-37143242.out
```

You should see information similar to:

```bash
Python 3.10.12

PyTorch: 2.14.0+cu130
CUDA available: True
CUDA version: 13.0
GPU: NVIDIA A100-SXM4-40GB

Gradio configuration
Host: 0.0.0.0
Port: 7860

Compute node:
erc-hpc-comp035

Starting Gradio...
```

The exact GPU may be different depending on what SLURM allocates.

Press `Ctrl + C` to stop monitoring.

---

#### 9. Monitor Errors

The SLURM script contains:

```bash
#SBATCH --error=/scratch/users/%u/mamba-gradio-%j.err
```

For job `37143242`, the error file is:

```
/scratch/users/$USER/mamba-gradio-37143242.err
```

View it:

```bash
cat /scratch/users/$USER/mamba-gradio-37143242.err
```

Or monitor it live:

```bash
tail -f /scratch/users/$USER/mamba-gradio-37143242.err
```

If the file is empty, that is usually a good sign.

---
#### 10. Wait for Gradio to Start

Keep monitoring:

```bash
tail -f /scratch/users/$USER/mamba-gradio-JOBID.out
```

The application may take some time to start if the Mamba model needs to be loaded.

The important point is that your job should remain:

```bash
ST = R
```

in:

```bash
squeue -u $USER
```

---

#### 11. Find the Compute Node

The output will show something like:

```bash
Compute node:
erc-hpc-comp035
```

The compute node can change every time the job runs.

Therefore, **do not permanently hard-code the compute node** in your SSH command.

For example, if the current node is:

```bash
erc-hpc-comp035
```

the tunnel will use that node.

---

#### 12. Create the SSH Tunnel

The Gradio server is running on the HPC compute node on port:

```bash
7860
```

Your local computer needs an SSH tunnel to access it.

##### On your local Windows computer

Open **PowerShell** or **Command Prompt**.

You should see a prompt similar to:

```bash
C:\Users\PC>
```

Run:

```powershell
ssh -m hmac-sha2-512 -L 7861:erc-hpc-comp035:7860 YOUR_KCL_USERNAME@hpc.create.kcl.ac.uk
```

Replace:

```text
erc-hpc-comp035
```

with the compute node assigned to your current SLURM job.

Replace:

```text
YOUR_KCL_USERNAME
```

with your KCL HPC username.

For example:

```powershell
ssh -m hmac-sha2-512 -L 7861:erc-hpc-comp035:7860 k20122072@hpc.create.kcl.ac.uk
```

Enter your KCL credentials if requested.

##### Important

Keep this SSH terminal **open** while using Gradio.

The SSH connection provides the tunnel between your computer and the HPC compute node.

---

#### 13. Open Gradio in Your Browser

Once the SSH tunnel is active, open Chrome, Edge, Firefox, or another browser.

Go to:

```text
http://localhost:7861
```

The Gradio interface should appear.

#### 14. Stopping the Application

When you are finished, cancel the SLURM job:

```bash
scancel JOBID
```

For example:

```bash
scancel 37143242
```

You can confirm that it has stopped with:

```bash
squeue -u $USER
```

Also, close the SSH tunnel on your local computer with:

```text
Ctrl + C
```

---

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

#### 9. Monitor Errors

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
You can modify the following line in KCL/run_kcl_analyze.sh file:
```python
python analyze.py input/sample.txt
or
python analyze.py input/sample.py
```
with one of the following:
```python
(version 1 and LoRA Adapter)
python analyze.py input/sample.txt --model-version 1 --model-type "LoRA Adapter"
or
python analyze.py input/sample.py --model-version 1 --model-type "LoRA Adapter"

(version 2 and LoRA Adapter)
python analyze.py input/sample.txt --model-version 2 --model-type "LoRA Adapter"
or
python analyze.py input/sample.py --model-version 2 --model-type "LoRA Adapter"

(version 1 and Full Model)
python analyze.py input/sample.txt --model-version 1 --model-type "Full Model"
or
python analyze.py input/sample.py --model-version 1 --model-type "Full Model"

(version 2 and Full Model)
python analyze.py input/sample.txt --model-version 2 --model-type "Full Model"
or
python analyze.py input/sample.py --model-version 2 --model-type "Full Model"
```

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

## Model and Repository Access

The project uses models hosted on Hugging Face. You may need to make sure the required model repositories are accessible from your environment.

Model identifiers used by the project include:

- `mistralai/Mistral-7B-v0.3`
  
**Mamba**  
- `HA-Siala/Mamba-v0.1`
- `HA-Siala/Mamba-v0.2`
- `HA-Siala/Mamba-full-v0.1`
- `HA-Siala/Mamba-full-v0.2`
  
**Python**
- `HA-Siala/Detect-Flaws-v0.1`
- `HA-Siala/Detect-Flaws-v0.2`
- `HA-Siala/RefactoringPy-v0.1`
- `HA-Siala/Detect-Flaws-full-v0.1`
- `HA-Siala/Detect-Flaws-full-v0.2`
- `HA-Siala/RefactoringPy-full-v0.1`
  
Please review the applicable model licenses and terms before redistributing model files or using them commercially.

---

## License

MIT License

---

## Contact

Student: hanan.siala@kcl.ac.uk
Supervisor: kevin.lano@kcl.ac.uk

Mamba Code Analyzer project.
