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

## Running the Command-Line Analyze

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

## Gradio Web Interface

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

## KCL CREATE HPC Workflow

The repository may include optional KCL CREATE scripts for running the project in an HPC/GPU environment.

These scripts are intended for the project author's workflow and are not required for normal local execution.

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

```

---

## Author

**HA-Siala**

Mamba Code Analyzer project.
