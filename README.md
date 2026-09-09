Mamba Code Analyzer
AI-Assisted Mamba Code Analysis
Mamba Code Analyzer is an AI-assisted application designed to analyse Mamba source code, detect potential flaws, explain identified issues, and generate improved versions of submitted code.
The project provides two ways to use the analyser:
Command-line interface.
Gradio graphical user interface.
The project is designed so that users can run the application using their own computing environment and available hardware.
---
Features
The application provides the following functionality:
Upload a source code file.
Automatically load the uploaded file contents into an editable code field.
Paste code manually.
Edit code before analysis.
Select a model type.
Select a model version.
Run AI-assisted code analysis.
Detect potential flaws.
Generate explanations of detected flaws.
Generate improved/refactored versions of code.
Display analysis results.
Save results to `output/output.txt`.
Download generated results through the Gradio interface.
Run inference using compatible hardware available to the user.
---
System Architecture
```text
                         USER
                           |
                           v
                  +----------------+
                  |   Gradio UI    |
                  +----------------+
                           |
              +------------+------------+
              |                         |
              v                         v
       Upload Source File          Paste/Edit Code
              |                         |
              +------------+------------+
                           |
                           v
                  Editable Code Input
                           |
                           v
                 Select Model Type
                 Select Model Version
                           |
                           v
                    Run Inference
                           |
                           v
             +-------------------------+
             | User Computing System   |
             |                         |
             | CPU or Compatible GPU   |
             +-------------------------+
                           |
                           v
                    Analysis Result
                           |
                           v
                  output/output.txt
                           |
                           v
                    Download Result
```
---
User-Provided Compute Model
This project is designed as software that users can run in their own computing environment.
The intended workflow is:
```text
Public GitHub Repository
          |
          v
User downloads or clones repository
          |
          v
User installs dependencies
          |
          v
User starts the Gradio application
          |
          v
User selects a model
          |
          v
Inference runs on the user's hardware
          |
          v
User receives analysis results
```
The repository itself does not need to provide a central GPU for every user.
Users may run the application using:
Their own compatible GPU.
Their own CPU.
A personal workstation.
University computing infrastructure.
Laboratory computing resources.
Institutional GPU servers.
Other computing resources available to them.
The availability and cost of computing resources depend on the environment selected by each user.
---
Project Structure
```text
mamba-code-analyzer/
|
├── analyze.py
├── model_service.py
├── app_test.py
├── requirements.txt
├── README.md
├── LICENSE
|
├── input/
│   └── sample.txt
|
├── output/
│   └── output.txt
|
└── .github/
    └── workflows/
        └── analyze.yml
```
---
Project Files
`analyze.py`
`analyze.py` provides the command-line application.
The program performs the following steps:
```text
Read Input File
      |
      v
Validate Input
      |
      v
Load Model
      |
      v
Run Inference
      |
      v
Extract Model Result
      |
      v
Format Result
      |
      v
Calculate Metrics
      |
      v
Save output/output.txt
```
The application also records inference metrics, including:
Input tokens.
Generated tokens.
Total inference time.
Time per input token.
Time per generated token.
---
`model_service.py`
`model_service.py` contains the model loading and inference functionality.
It is responsible for:
Detecting CUDA availability.
Detecting available GPU hardware.
Selecting an appropriate PyTorch data type.
Loading the tokenizer.
Loading the base model.
Loading LoRA adapters.
Loading full models.
Creating prompts.
Running inference.
Extracting structured model output.
Formatting the result.
---
`app_test.py`
`app_test.py` contains the Gradio graphical user interface.
The interface allows users to:
Select a model type.
Select a model version.
Upload an input code file.
Automatically load the file content into the code editor.
Paste code manually.
Edit uploaded or pasted code.
Submit the final code for analysis.
View analysis results.
Download the generated output.
The interface can also be tested without loading the AI model.
---
Model Configuration
The project supports multiple model configurations.
LoRA Adapter
When the user selects:
```text
LoRA Adapter
```
the application performs the following workflow:
```text
Load Tokenizer
      |
      v
Load Base Model
      |
      v
Load LoRA Adapter
      |
      v
Set Model to Evaluation Mode
      |
      v
Run Inference
```
---
Full Model
When the user selects:
```text
Full Model
```
the application loads the selected full model checkpoint.
The workflow is:
```text
Load Tokenizer
      |
      v
Load Full Model
      |
      v
Set Model to Evaluation Mode
      |
      v
Run Inference
```
---
Model Versions
The project currently supports multiple model versions.
Users can select:
Version 1
Version 2
The model paths are configured in `model_service.py`.
The current configuration includes:
```python
FULL_MODEL_V1 = (
    "HA-Siala/Mamba-full-v0.1"
)

FULL_MODEL_V2 = (
    "HA-Siala/Mamba-full-v0.2"
)

LORA_MODEL_V1 = (
    "HA-Siala/Mamba-v0.1"
)

LORA_MODEL_V2 = (
    "HA-Siala/Mamba-v0.2"
)

BASE_MODEL = (
    "mistralai/Mistral-7B-v0.3"
)
```
---
Installation
1. Clone the Repository
Clone the repository:
```bash
git clone https://github.com/HA-Siala/mamba-code-analyzer.git
```
Move into the project directory:
```bash
cd mamba-code-analyzer
```
---
2. Create a Python Virtual Environment
Using a virtual environment is recommended.
Windows
Create the environment:
```bash
python -m venv venv
```
Activate the environment:
```bash
venv\Scripts\activate
```
Linux
Create the environment:
```bash
python3 -m venv venv
```
Activate the environment:
```bash
source venv/bin/activate
```
---
3. Install Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Depending on the user's hardware environment, a compatible version of PyTorch and CUDA may be required for GPU inference.
---
Hardware Requirements
Gradio Interface Testing
The Gradio interface can be tested without loading the model.
For interface-only testing:
```text
GPU Required: No

Model Loading: No

Mistral Inference: No

LoRA Loading: No
```
This mode can be used to test:
Gradio layout.
Dropdown menus.
File upload.
Automatic loading of file contents.
Manual code input.
Code editing.
Output display.
Output file generation.
---
GPU and Device Detection
The model service automatically checks whether CUDA is available.
The application checks:
```python
torch.cuda.is_available()
```
If CUDA is available, the application uses:
```text
cuda:0
```
If CUDA is not available, the application falls back to:
```text
cpu
```
CPU inference may be significantly slower for large language models.
The application reports information including:
GPU name.
GPU family.
Compute capability.
PyTorch version.
CUDA version.
BF16 support.
Selected data type.
---
Running the Gradio Interface
To start the Gradio interface:
```bash
python app_test.py
```
After starting the application, Gradio will provide a local address similar to:
```text
http://127.0.0.1:7860
```
Open the displayed address in a web browser.
---
Using the Gradio Interface
Step 1: Start the Application
Run:
```bash
python app_test.py
```
Open the local URL displayed in the terminal.
---
Step 2: Select Model Type
Select one of:
```text
LoRA Adapter
```
or:
```text
Full Model
```
---
Step 3: Select Model Version
Select either:
```text
Version 1
```
or:
```text
Version 2
```
---
Step 4: Provide Input Code
There are two methods for providing code.
Option A: Upload a File
Upload a source code file.
The application automatically performs:
```text
Upload File
     |
     v
Read File Content
     |
     v
Load Content into Code Textbox
     |
     v
User Can Review/Edit Code
```
The uploaded file contents appear in the editable code textbox.
Option B: Paste Code Manually
Paste or type code directly into the code input area.
For example:
```python
print("Hello")
```
The code field remains editable.
---
Step 5: Review and Edit Code
Before analysis, the user can review and modify the submitted code.
The final content of the code textbox is the content that will be analysed.
```text
Uploaded File
      |
      v
Code Textbox
      ^
      |
Manual Editing
      |
      v
Final Code Submitted for Analysis
```
---
Step 6: Analyze Code
Click the Analyze Code button.
The application uses the selected model configuration to process the submitted code.
When connected to the model service, inference is performed using the hardware available in the user's environment.
---
Step 7: View the Result
The generated analysis appears in the result area of the Gradio interface.
The result may contain:
```text
Detected Flaws

Explanation of Flaws

Refactored Versions
```
---
Step 8: Save and Download the Result
The result is saved to:
```text
output/output.txt
```
The Gradio interface can also provide the generated output file for download.
---
Running the Command-Line Analyzer
The command-line analyzer can be executed using:
```bash
python analyze.py input/sample.txt
```
The program performs:
```text
Read Input File
      |
      v
Load Model
      |
      v
Run Inference
      |
      v
Extract Model Output
      |
      v
Format Result
      |
      v
Calculate Metrics
      |
      v
Save output/output.txt
```
---
Output
The generated result is written to:
```text
output/output.txt
```
The output contains detected flaws, explanations, refactored versions, and inference metrics.
The metrics include:
```text
Input tokens

Generated tokens

Inference time

Time per input token

Time per generated token
```
---
Example Command-Line Usage
```bash
python analyze.py input/sample.txt
```
Example workflow:
```text
Starting analyzer...

Reading input file:
input/sample.txt

Loading model...

Running inference...

Extracting model result...

Inference complete

Output saved to:
output/output.txt
```
---
Prompt and Analysis Task
The model receives an instruction to analyse the submitted Mamba code.
The task includes:
```text
Analyse the submitted code.

Detect potential flaws.

Provide short explanations.

Generate one or more corrected versions.

Do not reproduce the original code.

Show only improved/refactored versions.
```
---
Inference Configuration
The project uses deterministic generation.
The generation configuration includes:
```python
MAX_NEW_TOKENS = 32768

DO_SAMPLE = False
```
Inference is executed using:
```python
with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        **generation_kwargs
    )
```
---
Public Distribution
This project can be publicly distributed through GitHub.
Users can:
```text
View Repository
      |
      v
Clone Repository
      |
      v
Install Dependencies
      |
      v
Configure Their Environment
      |
      v
Run Gradio Interface
      |
      v
Run Inference Using Their Hardware
```
The software can be publicly available while computing resources are provided by individual users.
---
Compute Responsibility
The repository does not provide a central GPU inference service.
The intended model is:
```text
User A
Own Hardware
Own Inference

User B
Own Hardware
Own Inference

User C
Institutional Infrastructure
Own Inference Environment
```
Each user is responsible for ensuring that their environment has sufficient resources for the selected model.
---
Privacy
When the application is run locally:
```text
Web Browser
     |
     v
Local Gradio Server
     |
     v
Local Python Process
     |
     v
User's Computing Hardware
```
Users should review the privacy and security policies of any external computing infrastructure before processing sensitive or proprietary source code.
---
Reproducibility
For reproducible experiments, users should record:
Python version.
PyTorch version.
Transformers version.
PEFT version.
CUDA version.
GPU model.
Model type.
Model version.
Input configuration.
These details can affect inference performance and generated results.
---
Development Workflow
The recommended development workflow is:
```text
Modify Code
      |
      v
Run Syntax Checks
      |
      v
Test Gradio Interface
      |
      v
Test Model Loading
      |
      v
Run Inference
      |
      v
Review Output
      |
      v
Commit Changes
      |
      v
Push to GitHub
```
---
GitHub Actions
GitHub Actions can be used to automatically validate the repository.
The workflow can verify the presence of required files such as:
```text
analyze.py

model_service.py

requirements.txt

input/sample.txt
```
The workflow can also perform Python syntax checks.
GPU model inference can be kept separate from lightweight repository validation.
---
Limitations
This project provides AI-assisted code analysis.
Generated results should be reviewed before being used in:
Production systems.
Safety-critical software.
Security-sensitive applications.
Research experiments requiring formal verification.
AI-generated results may contain errors.
The generated analysis should complement, rather than replace:
Software testing.
Code review.
Expert review.
Security analysis.
Validation and verification.
---
License
A license should be added before public distribution.
The license defines how other users may:
Use the software.
Modify the software.
Distribute the software.
See the `LICENSE` file for the applicable terms.
---
Citation
If you use this software in academic research, please cite the associated publication.
Citation information can be added here when the relevant publication is available.
---
Contributing
Contributions, suggestions, and bug reports are welcome.
Recommended contribution process:
```text
Create Branch
      |
      v
Make Changes
      |
      v
Test Changes
      |
      v
Commit Changes
      |
      v
Push Branch
      |
      v
Create Pull Request
```
---
Future Development
Potential future improvements include:
Full integration between Gradio and the model service.
Additional model versions.
Additional LoRA adapters.
Batch processing.
Improved structured output validation.
Additional code analysis tasks.
Enhanced output visualisation.
Additional deployment options.
Support for additional computing environments.
---
Acknowledgements
This project uses software and machine learning libraries including:
PyTorch.
Hugging Face Transformers.
PEFT.
Gradio.
Users should refer to the documentation and licenses of these dependencies.
---
Disclaimer
This software is provided for research and development purposes.
Users are responsible for:
Selecting appropriate computing infrastructure.
Ensuring hardware compatibility.
Installing compatible dependencies.
Reviewing generated results.
Complying with applicable software licenses.
Protecting sensitive or proprietary source code.
The authors do not guarantee that generated analysis or refactored code is free from errors.
