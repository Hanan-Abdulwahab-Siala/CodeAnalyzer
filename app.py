import os
import uuid

import gradio as gr

from model_service import (
    load_model,
    generate_inference_output,
    extract_clean_dict,
    format_output,
    get_hardware_info,
    is_model_loaded,
    get_loaded_model_info,
)


OUTPUT_DIRECTORY = "output"


# ============================================================================
# Utility
# ============================================================================

def format_time(seconds):

    hours = int(seconds // 3600)
    minutes = int(
        (seconds % 3600) // 60
    )
    remaining_seconds = seconds % 60

    result = ""

    if hours > 0:
        result += f"{hours}h "

    if minutes > 0 or hours > 0:
        result += f"{minutes}m "

    result += f"{remaining_seconds:.6f}s"

    return result


# ============================================================================
# File upload
# ============================================================================

def load_file_into_textbox(
    uploaded_file,
    language,
):

    if uploaded_file is None:
        return ""

    try:

        file_path = uploaded_file

        if language == "Python":

            if not file_path.lower().endswith(".py"):

                return (
                    "# ERROR: Python input file must "
                    "have a .py extension."
                )

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            code = file.read()

        return code

    except Exception as e:

        return (
            "# ERROR: Could not read uploaded file.\n\n"
            f"# Reason: {e}"
        )


# ============================================================================
# Language change
# ============================================================================

def language_changed(language):

    if language == "Mamba":

        return (
            gr.update(
                choices=["Flaws + Refactoring"],
                value="Flaws + Refactoring",
                interactive=False,
                visible=True,
            ),
            gr.update(
                choices=["1", "2"],
                value="2",
            ),
            gr.update(
                label="Upload Mamba Code File",
            ),
            gr.update(
                label="Mamba Code",
                placeholder=(
                    "Upload a Mamba file above or "
                    "paste your Mamba code here..."
                ),
            ),
        )

    # Python

    return (
        gr.update(
            choices=[
                "Flaw Detection",
                "Refactoring",
            ],
            value="Flaw Detection",
            interactive=True,
            visible=True,
        ),
        gr.update(
            choices=["1", "2"],
            value="2",
        ),
        gr.update(
            label="Upload Python Code File (.py)",
        ),
        gr.update(
            label="Python Code",
            placeholder=(
                "Upload a Python .py file above or "
                "paste your Python code here..."
            ),
        ),
    )


# ============================================================================
# Task change
# ============================================================================

def task_changed(
    language,
    task,
):

    if language == "Python" and task == "Refactoring":

        return gr.update(
            choices=["1"],
            value="1",
        )

    return gr.update(
        choices=["1", "2"],
        value="2",
    )


# ============================================================================
# Model status
# ============================================================================

def get_model_status_text():

    if not is_model_loaded():

        return (
            "No model is currently loaded.\n\n"
            "Select Language, Task, Model Type and "
            "Model Version, then click Load Model."
        )

    info = get_loaded_model_info()

    return (
        "MODEL CURRENTLY LOADED\n\n"
        f"Language: {info['language']}\n"
        f"Task: {info['task']}\n"
        f"Model type: {info['model_type']}\n"
        f"Model version: {info['version']}\n"
        f"Checkpoint: {info['checkpoint']}"
    )


# ============================================================================
# Load selected model
# ============================================================================

def load_selected_model(
    language,
    task,
    model_version,
    model_type,
):

    try:

        version = int(model_version)

    except Exception:

        return (
            "ERROR: Invalid model version."
        )

    # ------------------------------------------------------------------------
    # Mamba has one task.
    # ------------------------------------------------------------------------

    if language == "Mamba":

        task = "Flaws + Refactoring"

    # ------------------------------------------------------------------------
    # Python refactoring only has version 1.
    # ------------------------------------------------------------------------

    if (
        language == "Python"
        and task == "Refactoring"
    ):

        version = 1

    try:

        result = load_model(
            language=language,
            task=task,
            version=version,
            model_type=model_type,
        )

    except Exception as e:

        return (
            "ERROR: Model loading failed.\n\n"
            f"Reason: {e}"
        )

    if result == "already_loaded":

        return (
            "MODEL ALREADY LOADED\n\n"
            f"Language: {language}\n"
            f"Task: {task}\n"
            f"Model type: {model_type}\n"
            f"Model version: {version}\n\n"
            "The existing GPU model is being reused."
        )

    return (
        "MODEL LOADED SUCCESSFULLY\n\n"
        f"Language: {language}\n"
        f"Task: {task}\n"
        f"Model type: {model_type}\n"
        f"Model version: {version}\n\n"
        "The model remains loaded in GPU memory "
        "until another model is explicitly loaded."
    )


# ============================================================================
# Analyze
# ============================================================================

def analyze_code(
    code,
    language,
    task,
):

    if code is None:

        return (
            "ERROR: No code was provided.",
            None,
        )

    if not code.strip():

        return (
            "ERROR: Code input is empty.",
            None,
        )

    if not is_model_loaded():

        return (
            "ERROR: No model is loaded.\n\n"
            "Select the configuration and click "
            "Load Model first.",
            None,
        )

    loaded_info = get_loaded_model_info()

    # ------------------------------------------------------------------------
    # Make sure UI configuration matches loaded model.
    # ------------------------------------------------------------------------

    if language == "Mamba":

        expected_task = "Flaws + Refactoring"

    else:

        expected_task = task

    if (
        loaded_info["language"] != language
        or loaded_info["task"] != expected_task
    ):

        return (
            "ERROR: The selected analysis configuration "
            "does not match the loaded model.\n\n"
            f"Loaded: "
            f"{loaded_info['language']} / "
            f"{loaded_info['task']}\n"
            f"Selected: "
            f"{language} / "
            f"{expected_task}\n\n"
            "Load the selected model first.",
            None,
        )

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    try:

        (
            output,
            input_tokens,
            generated_tokens,
            inference_time,
        ) = generate_inference_output(code)

    except Exception as e:

        return (
            "ERROR: Inference failed.\n\n"
            f"Reason: {e}",
            None,
        )

    if output is None or not str(output).strip():

        return (
            "ERROR: Model returned empty output.",
            None,
        )

    # ------------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------------

    try:

        output_dict = extract_clean_dict(
            output
        )

    except Exception as e:

        return (
            "ERROR: Could not extract the result dictionary.\n\n"
            f"Reason: {e}\n\n"
            "Raw model output:\n\n"
            f"{output}",
            None,
        )

    # ------------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------------

    try:

        formatted_output = format_output(
            output_dict
        )

    except Exception as e:

        return (
            "ERROR: Could not format the result.\n\n"
            f"Reason: {e}",
            None,
        )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    if input_tokens > 0:

        time_per_input_token = (
            inference_time / input_tokens
        )

    else:

        time_per_input_token = 0.0

    if generated_tokens > 0:

        time_per_generated_token = (
            inference_time / generated_tokens
        )

    else:

        time_per_generated_token = 0.0

    model_info = get_loaded_model_info()

    final_output = formatted_output

    final_output += (
        "========================================\n"
        "Inference Metrics\n"
        "========================================\n"
        f"Language:                  "
        f"{model_info['language']}\n"
        f"Task:                      "
        f"{model_info['task']}\n"
        f"Model type:                "
        f"{model_info['model_type']}\n"
        f"Model version:             "
        f"{model_info['version']}\n"
        f"Input tokens:              "
        f"{input_tokens}\n"
        f"Generated tokens:          "
        f"{generated_tokens}\n"
        f"Inference time:            "
        f"{format_time(inference_time)}\n"
        f"Time per input token:      "
        f"{format_time(time_per_input_token)}\n"
        f"Time per generated token:  "
        f"{format_time(time_per_generated_token)}\n"
        "========================================\n"
    )

    # ------------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------------

    try:

        os.makedirs(
            OUTPUT_DIRECTORY,
            exist_ok=True,
        )

        unique_id = uuid.uuid4().hex

        output_file_path = os.path.join(
            OUTPUT_DIRECTORY,
            f"output_{unique_id}.txt",
        )

        with open(
            output_file_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                final_output
            )

    except Exception as e:

        return (
            "ERROR: Could not save output file.\n\n"
            f"Reason: {e}",
            None,
        )

    return (
        final_output,
        output_file_path,
    )


# ============================================================================
# Clear
# ============================================================================

def clear_program():

    return (
        None,     # uploaded file
        "",       # code
        "",       # result
        None,     # output file
    )


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(
    title="Unified Code Analyzer"
) as demo:

    gr.Markdown(
        """
# Unified Code Analyzer

This analyzer supports **Mamba** and **Python** code.

### Mamba
Mamba has one combined task:

**Flaws + Refactoring**

### Python
Python supports:

- **Flaw Detection** — Version 1 or 2
- **Refactoring** — Version 1 only

Select the model configuration and click **Load Model**.

The selected model remains in GPU memory until another
model is explicitly loaded.
"""
    )

    # ------------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------------

    hardware_info = gr.Textbox(
        label="Hardware",
        value=get_hardware_info(),
        interactive=False,
        lines=12,
    )

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    with gr.Row():

        language = gr.Dropdown(
            choices=[
                "Mamba",
                "Python",
            ],
            value="Mamba",
            label="Language",
        )

        task = gr.Dropdown(
            choices=[
                "Flaws + Refactoring",
            ],
            value="Flaws + Refactoring",
            label="Task",
            interactive=False,
        )

    with gr.Row():

        model_type = gr.Dropdown(
            choices=[
                "LoRA Adapter",
                "Full Model",
            ],
            value="LoRA Adapter",
            label="Model Type",
        )

        model_version = gr.Dropdown(
            choices=[
                "1",
                "2",
            ],
            value="2",
            label="Model Version",
        )

    # ------------------------------------------------------------------------
    # Load button
    # ------------------------------------------------------------------------

    load_model_button = gr.Button(
        "Load Model",
        variant="primary",
    )

    model_status = gr.Textbox(
        label="Model Status",
        value=get_model_status_text(),
        interactive=False,
        lines=9,
    )

    # ------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------

    uploaded_file = gr.File(
        label="Upload Mamba Code File",
        type="filepath",
    )

    code_input = gr.Textbox(
        label="Mamba Code",
        placeholder=(
            "Upload a Mamba file above or "
            "paste your Mamba code here..."
        ),
        lines=25,
    )

    uploaded_file.change(
        fn=load_file_into_textbox,
        inputs=[
            uploaded_file,
            language,
        ],
        outputs=code_input,
    )

    # ------------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------------

    with gr.Row():

        analyze_button = gr.Button(
            "Analyze Code",
            variant="primary",
        )

        clear_button = gr.Button(
            "Clear / New Program",
            variant="secondary",
        )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    result_output = gr.Textbox(
        label="Analysis Result",
        lines=30,
        interactive=False,
    )

    output_file = gr.File(
        label="Download Analysis Output",
        interactive=False,
    )

    # ------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------

    language.change(
        fn=language_changed,
        inputs=[language],
        outputs=[
            task,
            model_version,
            uploaded_file,
            code_input,
        ],
    )

    task.change(
        fn=task_changed,
        inputs=[
            language,
            task,
        ],
        outputs=model_version,
    )

    load_model_button.click(
        fn=load_selected_model,
        inputs=[
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=model_status,
    )

    analyze_button.click(
        fn=analyze_code,
        inputs=[
            code_input,
            language,
            task,
        ],
        outputs=[
            result_output,
            output_file,
        ],
    )

    clear_button.click(
        fn=clear_program,
        inputs=[],
        outputs=[
            uploaded_file,
            code_input,
            result_output,
            output_file,
        ],
    )


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1
    )

    port = int(
        os.environ.get(
            "GRADIO_SERVER_PORT",
            "7860",
        )
    )

    print("=" * 60)
    print("Starting Unified Code Analyzer")
    print("Model loading is manual.")
    print(
        f"Gradio server: http://0.0.0.0:{port}"
    )
    print(f"Port: {port}")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
