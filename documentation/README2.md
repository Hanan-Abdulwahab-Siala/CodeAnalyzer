## 2) Running from the Command-Line 

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

Mamba — combined Flaws + Refactoring

There is no task argument:
```bash
python analyze.py \
    input/sample.txt \
    --language Mamba \
    --model-version 2 \
    --model-type "LoRA Adapter" \
    --output output/output.txt
```
This uses: HA-Siala/Mamba-v0.2

or

```bash
python analyze.py \
    input/sample.txt \
    --language Mamba \
    --model-version 1 \
    --model-type "LoRA Adapter" \
    --output output/output.txt
```
This uses: HA-Siala/Mamba-v0.1

Mamba — combined Flaws + Refactoring v2 Full

```bash
python analyze.py \
    input/sample.txt \
    --language Mamba \
    --model-version 2 \
    --model-type "Full Model" \
    --output output/output.txt
```
This uses: Mamba-full-v0.2

or

```bash
python analyze.py \
    input/sample.txt \
    --language Mamba \
    --model-version 1 \
    --model-type "Full Model" \
    --output output/output.txt
```
This uses: HA-Siala/Mamba-full-v0.1

And the same Mamba inference mechanism for Python.

Python — Flaw Detection v2 LoRA
```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Flaw Detection" \
    --model-version 2 \
    --model-type "LoRA Adapter" \
    --output output/output.txt
```
Uses: HA-Siala/Detect-Flaws-v0.2

or 

```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Flaw Detection" \
    --model-version 1 \
    --model-type "LoRA Adapter" \
    --output output/output.txt
```
Uses: HA-Siala/Detect-Flaws-v0.1

Python — Flaw Detection v2 Full
```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Flaw Detection" \
    --model-version 2 \
    --model-type "Full Model" \
    --output output/output.txt
```
Uses: HA-Siala/Detect-Flaws-full-v0.2

Python — Flaw Detection v1 Full
```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Flaw Detection" \
    --model-version 1 \
    --model-type "Full Model" \
    --output output/output.txt
```
Uses: HA-Siala/Detect-Flaws-full-v0.1

Python — Refactoring LoRA
You do not need to specify version 1 because the program automatically forces it:
```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Refactoring" \
    --model-type "LoRA Adapter" \
    --output output/output.txt
```
Uses: HA-Siala/RefactoringPy-v0.1

Python — Refactoring Full Model
```bash
python analyze.py \
    input/sample.py \
    --language Python \
    --task "Refactoring" \
    --model-type "Full Model" \
    --output output/output.txt
```
Uses: HA-Siala/RefactoringPy-full-v0.1

mmmmmmmmmmmmmmmmmmmmmm
python analyze.py \
    input/sample.txt \
    --model "DeepSeek" \
    --language Mamba \
    --model-version 1 \
    --model-type "LoRA Adapter" \
    --output output/output.txt


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
