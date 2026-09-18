# Code Analyzer

A GPU-powered application for analyzing Mamba/Python code using a fine-tuned Mistral LLM.

The project provides:

- A command-line analyzer
- A Gradio web interface
- LoRA adapter and full-model options
- Automatic GPU hardware detection
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
python3 -m venv ~/venvs
```

Activate the virtual environment.

**Linux / macOS**

```bash
source ~/venvs/bin/activate

```

**Windows**

```bash
. ~/venvs/bin/activate
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
### 4. [Running Using KCL CREATE HPC Workflow without Gradio](./documentation/README4.md)

If you are not using the KCL CREATE environment, you can ignore the HPC scripts (3 and 4) and run the project using the normal Python environment.

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

## Formatting Output

We use a function that converts the parsed dictionary into readable output.

Example:

```text
Flaws:
- Example flaw: Explanation of the problem.

Refactored versions code:
def corrected_function():
    ...
```

---

## Supported Models

The project uses models hosted on Hugging Face. You may need to make sure the required model repositories are accessible from your environment.

Model identifiers used by the project include:

### Base Model

- 👉 [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)
- 👉 [deepseek-ai/deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)

### LoRA Adapter Models
#### Mistral
`Mamba`  
- 👉 [Version 1: HA-Siala/Mamba-v0.1](https://huggingface.co/HA-Siala/Mamba-v0.1)  
- 👉 [Version 2: HA-Siala/Mamba-v0.2](https://huggingface.co/HA-Siala/Mamba-v0.2)  
  
`Python`
- 👉 [Version 1: HA-Siala/Detect-Flaws-v0.1](https://huggingface.co/HA-Siala/Detect-Flaws-v0.1)  
- 👉 [Version 2: HA-Siala/Detect-Flaws-v0.2](https://huggingface.co/HA-Siala/Detect-Flaws-v0.2)  
- 👉 [Version 1: HA-Siala/RefactoringPy-v0.1](https://huggingface.co/HA-Siala/RefactoringPy-v0.1)  

#### DeepSeek
`Mamba`  
- 👉 [Version 1: HA-Siala/Mamba-DeepSeek-v0.1](https://huggingface.co/HA-Siala/Mamba-DeepSeek-v0.1)  

`Python`
- 👉 [Version 1: HA-Siala/RefactoringPy-DeepSeek-v0.1](https://huggingface.co/HA-Siala/RefactoringPy-DeepSeek-v0.1)  

### Full Models
#### Mistral
`Mamba`  
- 👉 [Version 1: HA-Siala/Mamba-full-v0.1](https://huggingface.co/HA-Siala/Mamba-full-v0.1)  
- 👉 [Version 2: HA-Siala/Mamba-full-v0.2](https://huggingface.co/HA-Siala/Mamba-full-v0.2)  
`Python`
- 👉 [Version 1: HA-Siala/Detect-Flaws-full-v0.1](https://huggingface.co/HA-Siala/Detect-Flaws-full-v0.1)
- 👉 [Version 2: HA-Siala/Detect-Flaws-full-v0.2](https://huggingface.co/HA-Siala/Detect-Flaws-full-v0.2)
- 👉 [Version 1: HA-Siala/RefactoringPy-full-v0.1](https://huggingface.co/HA-Siala/RefactoringPy-full-v0.1)

#### DeepSeek
`Mamba`
- 👉 [Version 1: HA-Siala/Mamba-DeepSeek-full-v0.1](https://huggingface.co/HA-Siala/Mamba-DeepSeek-full-v0.1)

`Python`
- 👉 [Version 1: HA-Siala/RefactoringPy-DeepSeek-full-v0.1](https://huggingface.co/HA-Siala/RefactoringPy-DeepSeek-full-v0.1)

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
