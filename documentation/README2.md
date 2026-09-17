## 2) Running from the Command-Line 

Please follow these instructions:  

#### 1. Connect to the GPU Provider

From your local computer, connect to the remote server, for example:

```bash
ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk
```
After connecting, you should see a shell prompt on the HPC login node:
```bash
k12345@arc-hpc-login3:~$
```
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
app.py
input/
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

#### 4. Ask for a GPU from the GPU Provider
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
And then:

```bash
module load cuda
```

---

#### 5. Activate the Python Virtual Environment
Activate the virtual environment:

```bash
. ~/venvs/bin/activate
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
.../venvs/bin/python
```
---

#### 6. Run the Program

Run the analyze program using one of the following:

Mistral + Mamba + Version 1 + LoRA Adapter
```bash
 python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 1 --model-type "LoRA Adapter" 
```

Mistral + Mamba + Version 2 + LoRA Adapter
```bash
 python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 2 --model-type "LoRA Adapter" 
```

Mistral + Mamba + Version 1 + Full Model
```bash
 python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 1 --model-type "Full Model" 
```

Mistral + Mamba + Version 2 + Full Model
```bash
 python analyze.py input/sample.txt --language Mamba --model-family Mistral --model-version 2 --model-type "Full Model" 
```
---
Mistral + Python + Flaw Detection + Version 1 + LoRA Adapter
```bash
 python analyze.py input/sample.py --language Python --model-family Mistral --task "Flaw Detection" --model-version 1 --model-type "LoRA Adapter" 
```

Mistral + Python + Flaw Detection + Version 2 + LoRA Adapter
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family Mistral \
     --task "Flaw Detection" \
     --model-version 2 \
     --model-type "LoRA Adapter" 
```

Mistral + Python + Flaw Detection + Version 1 + Full Model
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family Mistral \
     --task "Flaw Detection" \
     --model-version 1 \
     --model-type "Full Model" 
```

Mistral + Python + Flaw Detection + Version 2 + Full Model
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family Mistral \
     --task "Flaw Detection" \
     --model-version 2 \
     --model-type "Full Model" 
```
---
Mistral + Python + Refactoring + Version 1 + LoRA Adapter
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family Mistral \
     --task "Refactoring" \
     --model-version 1 \
     --model-type "LoRA Adapter" 
```

Mistral + Python + Refactoring + Version 1 + Full Model
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family Mistral \
     --task "Refactoring" \
     --model-version 1 \
     --model-type "Full Model" 
```
---
DeepSeek + Mamba + Version 1 + LoRA Adapter
```bash
 python analyze.py \
     input/sample.txt \
     --language Mamba \
     --model-family DeepSeek \
     --model-version 1 \
     --model-type "LoRA Adapter" 
```

DeepSeek + Mamba + Version 1 + Full Model
```bash
 python analyze.py \
     input/sample.txt \
     --language Mamba \
     --model-family DeepSeek \
     --model-version 1 \
     --model-type "Full Model" 
```
---
DeepSeek + Python + Version 1 + LoRA Adapter
 ```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family DeepSeek \
     --model-version 1 \
     --model-type "LoRA Adapter" 
```

DeepSeek + Python + Version 1 + Full Model
```bash
 python analyze.py \
     input/sample.py \
     --language Python \
     --model-family DeepSeek \
     --model-version 1 \
     --model-type "Full Model" 
```

The output will be saved to: output/output.txt

---
