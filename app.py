import os
import gradio as gr

from model_service import (
    load_model,
    GenerateInferenceOutput,
    ExtractCleanDict,
    FormatOutput
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIRECTORY = "output"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "output.txt"
)


# ============================================================
# FORMAT TIME
# ============================================================

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


# ============================================================
# LOAD UPLOADED FILE
# ============================================================

def load_file(uploaded_file):

    if uploaded_file is None:
        return ""

    try:

        with open(
            uploaded_file,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except Exception as e:

        return (
            "# ERROR: Could not read file.\n\n"
            f"# Reason: {e}"
        )


# ============================================================
# ANALYZE CODE
# ============================================================

def analyze_code(
    code,
    model_version,
    model_type
):

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if code is None:

        return (
            "ERROR: No code was provided.",
            None
        )

    if not code.strip():

        return (
            "ERROR: Code input is empty.\n\n"
            "Please upload a file or paste code.",
            None
        )


    # --------------------------------------------------------
    # MODEL VERSION
    # --------------------------------------------------------

    try:

        version = int(model_version)

    except Exception:

        return (
            "ERROR: Invalid model version.",
            None
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        load_model(
            version=version,
            model_type=model_type
        )

    except Exception as e:

        return (
            "ERROR: Model loading failed.\n\n"
            f"Reason: {e}",
            None
        )


    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    try:

        (
            output,
            input_tokens,
            generated_tokens,
            inference_time
        ) = GenerateInferenceOutput(code)

    except Exception as e:

        return (
            "ERROR: Inference failed.\n\n"
            f"Reason: {e}",
            None
        )


    # --------------------------------------------------------
    # CHECK OUTPUT
    # --------------------------------------------------------

    if output is None:

        return (
            "ERROR: Model did not return a response.",
            None
        )


    # --------------------------------------------------------
    # EXTRACT RESULT
    # --------------------------------------------------------

    try:

        output_dict = ExtractCleanDict(output)

    except Exception as e:

        return (
            "ERROR: Could not extract model result.\n\n"
            f"Reason: {e}\n\n"
            "Raw model output:\n\n"
            f"{output}",
            None
        )


    # --------------------------------------------------------
    # FORMAT RESULT
    # --------------------------------------------------------

    formatted_output = FormatOutput(
        output_dict
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    final_output = formatted_output

    final_output += (
        "\n"
        "========================================\n"
        "Inference Metrics\n"
        "========================================\n"
    )

    final_output += (
        f"Model type:                 {model_type}\n"
    )

    final_output += (
        f"Model version:              {version}\n"
    )

    final_output += (
        f"Input tokens:               {input_tokens}\n"
    )

    final_output += (
        f"Generated tokens:           {generated_tokens}\n"
    )

    final_output += (
        f"Inference time:             "
        f"{format_time(inference_time)}\n"
    )

    final_output += (
        f"Time per input token:       "
        f"{format_time(time_per_input_token)}\n"
    )

    final_output += (
        f"Time per generated token:   "
        f"{format_time(time_per_generated_token)}\n"
    )

    final_output += (
        "========================================\n"
    )


    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            final_output
        )


    # --------------------------------------------------------
    # RETURN TO GRADIO
    # --------------------------------------------------------

    return (
        final_output,
        OUTPUT_FILE
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Mamba Code Analyzer"
) as demo:

    gr.Markdown(
        """
# Mamba Code Analyzer

Upload a Mamba code file or paste code manually.

The model runs **locally on your computer**.

Your code and model inference are not sent to
my GPU or to a central inference server.
"""
    )


    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    with gr.Row():

        model_type = gr.Dropdown(

            choices=[
                "LoRA Adapter",
                "Full Model"
            ],

            value="LoRA Adapter",

            label="Model Type"
        )


        model_version = gr.Dropdown(

            choices=[
                "1",
                "2"
            ],

            value="2",

            label="Model Version"
        )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = gr.File(

        label="Upload Mamba Code File",

        type="filepath"
    )


    # --------------------------------------------------------
    # CODE INPUT
    # --------------------------------------------------------

    code_input = gr.Textbox(

        label="Mamba Code",

        placeholder=(
            "Upload a file or paste "
            "your Mamba code here..."
        ),

        lines=25
    )


    # --------------------------------------------------------
    # LOAD FILE AUTOMATICALLY
    # --------------------------------------------------------

    uploaded_file.change(

        fn=load_file,

        inputs=uploaded_file,

        outputs=code_input
    )


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analyze_button = gr.Button(

        "Analyze Code",

        variant="primary"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_output = gr.Textbox(

        label="Analysis Result",

        lines=30,

        interactive=False
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output_file = gr.File(

        label="Download Analysis Output",

        interactive=False
    )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(

        fn=analyze_code,

        inputs=[
            code_input,
            model_version,
            model_type
        ],

        outputs=[
            result_output,
            output_file
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    demo.launch()
