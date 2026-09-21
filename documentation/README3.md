## 3) Running Using KCL CREATE HPC Workflow with Gradio

We provide an optional KCL CREATE script to run the project in an HPC/GPU environment using a Gradio interface.

#### 1. Connect to the GPU Provider

From your local computer, connect to the remote GPU/HPC server using SSH. For example:

```bash
ssh -m hmac-sha2-512 USERNAME@HPC_HOST
```

Replace:

- USERNAME with your account username on the HPC/GPU provider.
- HPC_HOST with the hostname of the remote HPC/GPU server.

After connecting successfully, you should see a shell prompt on the HPC login node, similar to:

```bash
USERNAME@HPC_LOGIN_NODE:~$
```

For example, if your provider gives you the username `k12345` and the login hostname `hpc.example.org`:

```bash
ssh -m hmac-sha2-512 k12345@hpc.example.org
```

> **Note:** You must have an account and SSH access to the HPC/GPU provider before running this command. The hostname, username, authentication method, and SSH options may differ between providers.

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

#### 4. Check the SLURM Script and Make the Script Executable

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
Submitted batch job 12345678
```

The number is the **JOBID**. For example:

```
JOBID=12345678
```

Your JOBID will be different each time you submit a new job.

---
#### 6. Open the Tunnel Automatically

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

#### 7. Open Gradio in Your Browser

Once the SSH tunnel is active, open Chrome, Edge, Firefox, or another browser.

Go to:

```text
http://localhost:7860
```

The Gradio interface should appear.

---
