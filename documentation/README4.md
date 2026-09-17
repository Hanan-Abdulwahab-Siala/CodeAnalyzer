## 4) Running Using KCL CREATE HPC Workflow without Gradio

We provide an optional KCL CREATE script to run the project in an HPC/GPU environment without the Gradio interface.

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

#### 4. Check the SLURM Script and Make the Script Executable

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
You can modify the `KCL/run_kcl_analyze.sh` file to perform the requested task, as described in the second method and explained in the shell script.

---
