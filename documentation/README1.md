## 1) Running Via Gradio Web Interface

The Gradio interface can be used to:

- Select the model version
- Select LoRA or full model
- Enter Mamba code
- Run the analyze
- View detected flaws
- View refactored versions
- View inference information

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
...
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

#### 4. Ask for GPU from the GPU Provider:
For example, in KCL, we use:

```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --time=04:00:00 \
     --cpus-per-task=4 \
     --mem=32G \
     --pty /bin/bash -l
```
Then you will see:
```bash
srun: job 37241627 has been allocated resources
```
This number is important when we open the Gradio interface. Please keep it.

```bash
nvidia-smi
```
Then:

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

Run the Gradio program:

```bash
python app.py
```
---

#### 7. Open a Second CMD Window
Now go to your Windows PC and open a second CMD window, and put:

```bash
for /f "delims=" %N in ('ssh -m hmac-sha2-512 USER@HPC_HOST "squeue -n JOB_NAME -h -o %%N"') do ssh -m hmac-sha2-512 -N -L LOCAL_PORT:%N:REMOTE_PORT USER@HPC_HOST

or

for /f "delims=" %N in ('ssh -m hmac-sha2-512 USER@HPC_HOST "squeue -j JOB_ID -h -o %%N"') do ssh -m hmac-sha2-512 -N -L LOCAL_PORT:%N:REMOTE_PORT USER@HPC_HOST
```
Where:
 -  USER → your HPC username
 - HPC_HOST → your HPC login host
 - JOB_NAME → your Slurm job name
 - LOCAL_PORT → local port, e.g. 7860
 - REMOTE_PORT → application port, e.g. 7860
 - JOB_ID → the Slurm job ID, e.g. 12345678

For example:
```bash
for /f "delims=" %N in ('ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk "squeue -j Jupyter Lab 12345678 -h -o %%N"') do ssh -m hmac-sha2-512 -N -L 7860:%N:7860 k12345@hpc.create.kcl.ac.uk

or

for /f "delims=" %N in ('ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk "squeue -j 12345678 -h -o %%N"') do ssh -m hmac-sha2-512 -N -L 7860:%N:7860 k12345@hpc.create.kcl.ac.uk
```
---

#### 8. Windows: Open the Browser

Please leave both windows open, then open your browser and enter:
```bash
http://localhost:7860
```
Your Gradio application should open.

---
