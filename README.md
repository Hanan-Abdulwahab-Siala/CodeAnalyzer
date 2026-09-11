# Mamba Code Analyzer

A local GPU-powered application for analyzing Mamba code using fine-tuned Mamba/Mistral language models.

The project provides:

- A command-line analyzer
- A Gradio web interface
- LoRA adapter and full-model options
- Automatic GPU/CPU hardware detection
- Local inference without a central inference server
- Optional KCL CREATE HPC scripts for the project author's GPU workflow

---

## Important: How Inference Works

This project **does not provide a central GPU server**.

The models run locally on the machine where the application is started.

The application:

1. Loads the selected model locally.
2. Loads the tokenizer locally.
3. Sends the input code directly to the local model.
4. Performs inference on the local GPU or CPU.
5. Returns the generated analysis locally.

No project-hosted inference API is required.

If you have a compatible NVIDIA GPU, inference can be performed using CUDA. If CUDA is not available, the application falls back to CPU execution.

---

## Project Structure

A typical project structure is:

```text
Mamba-Code-Analyzer/
│
├── README.md
├── requirements.txt
│
├── mamba_analyzer.py
├── app.py
│
├── models/
│
├── examples/
│   └── example.py
│
└── KCL/
    └── create scripts
```

The exact file names may differ depending on the version of the project.

---

## Features

### Local Model Inference

The application runs the language model locally. There is no requirement for a central inference server.

### GPU Acceleration

The application automatically detects CUDA:

```python
torch.cuda.is_available()
```

When CUDA is available, the model is loaded on the GPU. The application also detects the GPU name and available VRAM.

### Automatic Data Type Selection

For NVIDIA GPUs, the application automatically selects:

- `bfloat16` when supported
- `float16` otherwise

For CPU execution, it uses:

- `float32`

Example:

```text
CUDA available : True
Device         : cuda
GPU            : NVIDIA ...
Dtype          : torch.bfloat16
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

The LoRA workflow is:

```
Mistral 7B base model
        +
  Mamba LoRA adapter
        |
        v
 Mamba Code Analyzer
```

### Full Models

- **Version 1:** `HA-Siala/Mamba-full-v0.1`
- **Version 2:** `HA-Siala/Mamba-full-v0.2`

The full-model workflow loads the complete checkpoint directly.

### Model Selection

The model loader supports two versions:

```
version=1
```

and:

```
version=2
```

It also supports:

```
model_type="LoRA Adapter"
```

and:

```
model_type="Full Model"
```

**Example: Version 2 LoRA**

```python
load_model(
    version=2,
    model_type="LoRA Adapter",
)
```

**Example: Version 1 LoRA**

```python
load_model(
    version=1,
    model_type="LoRA Adapter",
)
```

**Example: Version 2 Full Model**

```python
load_model(
    version=2,
    model_type="Full Model",
)
```

**Example: Version 1 Full Model**

```python
load_model(
    version=1,
    model_type="Full Model",
)
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
- NVIDIA GPU with CUDA support for GPU inference

CPU inference is also supported, although it can be significantly slower.

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
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
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### PyTorch and CUDA

For GPU inference, install a PyTorch build compatible with your NVIDIA driver and CUDA environment.

Check whether CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected GPU result:

```
True
```

Check the detected GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Check the installed PyTorch and CUDA versions:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
```

### Hugging Face Models

The first time a model is loaded, Hugging Face may download the required model files to the local Hugging Face cache.

For LoRA models, the application downloads:

```
mistralai/Mistral-7B-v0.3
```

and the selected adapter.

For full models, the selected full checkpoint is downloaded.

Make sure the machine has:

- Internet access during the first model download
- Enough disk space
- Enough RAM
- Enough GPU VRAM for the selected model

After the model has been downloaded, the Hugging Face cache can allow subsequent runs to use the local files.

---

## Running the Command-Line Analyzer

If the project contains a command-line entry point, run it according to the project's entry script.

For example:

```bash
python mamba_analyzer.py
```

or:

```bash
python app.py
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
- Run the analyzer
- View detected flaws
- View refactored versions
- View inference information

---

## Basic Python Usage

The main model-loading function is:

```python
load_model()
```

Example:

```python
load_model(
    version=2,
    model_type="LoRA Adapter",
)
```

Then provide code to:

```python
generate_inference_output()
```

Example:

```python
code = """
def calculate(x):
    result = x * 2
    return result
"""

output, input_tokens, generated_tokens, inference_time = (
    generate_inference_output(code)
)

print(output)
```

### Inference Output

The inference function returns:

- `output`
- `input_tokens`
- `generated_tokens`
- `inference_time`

Example:

```python
output, input_tokens, generated_tokens, inference_time = (
    generate_inference_output(code)
)

print("Output:")
print(output)
print("Input tokens:", input_tokens)
print("Generated tokens:", generated_tokens)
print("Inference time:", inference_time)
```

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

The actual response depends on the model and the submitted code.

### Output Parsing

The project includes a function for extracting the dictionary from the model output:

```python
extract_clean_dict()
```

The function:

1. Searches for the generated dictionary.
2. Locates the opening `{`.
3. Tracks nested braces.
4. Handles strings containing braces.
5. Finds the matching closing `}`.
6. Parses the result using `ast.literal_eval()`.
7. Verifies that the result is a dictionary.
8. Verifies the `Flaws` key.
9. Verifies the `Refactored Versions` key.

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

On CPU:

```text
GPU: CPU
PyTorch: 2.x.x
CUDA: Not available
```

---

## VRAM Management

Large language models require significant GPU memory.

The application includes:

```python
clear_model()
```

which removes the currently loaded model and clears the CUDA cache.

When a different model is requested, the application clears the previous model before loading the new one. When the same model is requested again, the application reuses the existing model instead of loading it again.

Example:

```python
load_model(
    version=2,
    model_type="LoRA Adapter",
)
# The second call reuses the already loaded model.
load_model(
    version=2,
    model_type="LoRA Adapter",
)
```

---

## Generation Configuration

The maximum number of newly generated tokens is controlled by:

```python
MAX_NEW_TOKENS = 4096
```

You can reduce this value if GPU memory is limited:

```python
MAX_NEW_TOKENS = 2048
```

Or increase it if your hardware has sufficient resources:

```python
MAX_NEW_TOKENS = 8192
```

Larger values can increase:

- GPU memory usage
- Inference time
- Generated response length

### Deterministic Generation

The default configuration is:

```python
DO_SAMPLE = False
```

This is intentional for code analysis because deterministic generation generally makes results more reproducible.

Sampling can be enabled with:

```python
DO_SAMPLE = True
```

if more varied responses are desired.

### Inference Timing

The application measures inference time using:

```python
time.perf_counter()
```

CUDA synchronization is performed before and after generation. This is important because CUDA operations are asynchronous. The reported inference time therefore measures the actual model generation operation more accurately than a simple unsynchronized timer.

---

## Transformers Compatibility

Depending on the installed version of Transformers, the model-loading argument may be:

```python
dtype=DTYPE
```

or:

```python
torch_dtype=DTYPE
```

The current implementation uses:

```python
dtype=DTYPE
```

If your Transformers installation reports an error similar to:

```text
TypeError: ... got an unexpected keyword argument 'dtype'
```

replace:

```python
dtype=DTYPE
```

with:

```python
torch_dtype=DTYPE
```

inside the model-loading function.

---

## Common Problems

### CUDA Is Not Available

If:

```python
torch.cuda.is_available()
```

returns:

```
False
```

check:

- NVIDIA driver installation
- PyTorch installation
- CUDA compatibility
- GPU visibility
- Python environment

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### CUDA Out of Memory

If you receive:

```
CUDA out of memory
```

try:

1. Reducing `MAX_NEW_TOKENS`.
2. Closing other GPU applications.
3. Clearing the currently loaded model.
4. Using the LoRA configuration if appropriate.
5. Using a GPU with more VRAM.
6. Checking available VRAM before loading the model.

You can check VRAM with:

```python
get_hardware_info()
```

### Model Loading Takes a Long Time

The first model load can take longer because model files may need to be downloaded. Subsequent loads can be faster when the files are already available in the local Hugging Face cache.

### LoRA Adapter Loading Error

When using:

```python
model_type="LoRA Adapter"
```

the application loads:

```
mistralai/Mistral-7B-v0.3
```

as the base model and then attaches the selected adapter. The adapter must be compatible with the selected base model.

### Dictionary Parsing Error

If the model does not return a valid dictionary, you may see:

```text
ValueError: Could not parse model output as a Python dictionary
```

Possible causes include:

- Incomplete model output
- Invalid Python syntax
- Markdown instead of a dictionary
- Extra text around the result
- Incorrect quotation marks
- Malformed nested dictionaries
- Output being truncated

---

## Security

The generated code should be treated as **untrusted model output**.

Do not automatically execute generated code. For example, do not pass model output directly to:

```python
exec()
```

or:

```python
eval()
```

The parser uses:

```python
ast.literal_eval()
```

to parse the expected dictionary instead of executing arbitrary Python expressions.

Generated refactored code should still be manually reviewed and tested before execution.

---

## Local Execution

This project is designed for local inference. The basic architecture is:

```
User
  |
  v
Local Application
  |
  v
Tokenizer
  |
  v
Local Mamba/Mistral Model
  |
  v
Local GPU / CPU
  |
  v
Generated Analysis
  |
  v
User
```

There is no requirement for:

```
User
  |
  v
Central GPU Server
  |
  v
Remote Model
```

unless you choose to build such infrastructure separately.

---

## KCL CREATE HPC Workflow

The repository may include optional KCL CREATE scripts for running the project in an HPC/GPU environment.

These scripts are intended for the project author's workflow and are not required for normal local execution.

If you are not using the KCL CREATE environment, you can ignore the HPC scripts and run the project using the normal Python environment.

---

## Recommended Project Workflow

A typical workflow is:

```
1. Start the application
        |
        v
2. Detect GPU / CPU
        |
        v
3. Select model
        |
        +---- LoRA Adapter
        |
        +---- Full Model
        |
        v
4. Load tokenizer
        |
        v
5. Load model
        |
        v
6. Submit Mamba code
        |
        v
7. Generate analysis
        |
        v
8. Extract result dictionary
        |
        v
9. Validate result
        |
        v
10. Display flaws
        |
        v
11. Display refactored code
```

---

## Limitations

The analyzer is based on a language model and therefore does not guarantee that every analysis is correct.

The model may:

- Miss bugs
- Report false positives
- Produce incorrect explanations
- Produce incomplete refactoring
- Produce syntactically invalid code
- Produce logically incorrect code
- Return malformed structured output

Generated code should always be reviewed and tested.

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

## Development

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
```

or use the appropriate entry point for the specific project version.

---

## Contributing

Contributions are welcome.

Before submitting a change:

1. Test the application locally.
2. Verify model loading.
3. Test both CPU and GPU behavior when possible.
4. Check LoRA loading.
5. Check full-model loading.
6. Test model switching.
7. Test malformed model output.
8. Check that generated code is not automatically executed.

---

## License

Add the project's license here.

For example:

```
MIT License
```

If the project uses a different license, replace the above with the actual license.

Note that the project's license does not necessarily change the licenses or terms of the underlying models.

---

## Acknowledgements

This project uses:

- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face Accelerate
- Mistral 7B
- HA-Siala Mamba model checkpoints
- Gradio
- KCL CREATE HPC infrastructure, where applicable

---

## Author

**HA-Siala**

Mamba Code Analyzer project.
