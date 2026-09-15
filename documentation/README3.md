## 3) Running Using KCL CREATE HPC Workflow with Gradio

We provide an optional KCL CREATE script to run the project in an HPC/GPU environment using a Gradio interface.

#### 1. Connect to KCL HPC

From your local computer, connect to the KCL HPC login node:

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
input/
.venv/
```

You can also check your current directory:

```bash
pwd
```
---

#### 3. Check the SLURM Script and Make the Script Executable

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

#### 4. Submit the SLURM Job

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
#### 5. Open the Tunnel Automatically

Open **PowerShell** or **Command Prompt**.

You should see a prompt similar to:

```bash
C:\Users\PC>
```

Run:

```powershell
for /f "delims=" %N in ('ssh -m hmac-sha2-512 k12345@hpc.create.kcl.ac.uk "squeue -j 37243372 -h -o %%N"') do ssh -m hmac-sha2-512 -N -L 7860:%N:7860 k12345@hpc.create.kcl.ac.uk
```
where 37243372 is a JOBID

---

#### 6. Open Gradio in Your Browser

Once the SSH tunnel is active, open Chrome, Edge, Firefox, or another browser.

Go to:

```text
http://localhost:7860
```

The Gradio interface should appear.

---
#### 7. Stopping the Application

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
