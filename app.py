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

OUTPUT_DIRECTORY = (
    "output"
)


OUTPUT_FILE = (

    os.path.join(

        OUTPUT_DIRECTORY,

        "output.txt"
    )
)


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(
    seconds
):

    hours = int(
        seconds // 3600
    )


    minutes = int(

        (
            seconds % 3600
        )

        // 60
    )


    remaining_seconds = (

        seconds % 60
    )


    result = ""


    if hours > 0:

        result += (
            f"{hours}h "
        )


    if minutes > 0 or hours > 0:

        result += (
            f"{minutes}m "
        )


    result += (
        f"{remaining_seconds:.6f}s"
    )


    return result


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

            "ERROR: Code input is empty.",

            None
        )


    # --------------------------------------------------------
    # CONVERT MODEL VERSION
    # --------------------------------------------------------

    version = int(
        model_version
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

        ) = (

            GenerateInferenceOutput(
                code
            )
        )


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

            "ERROR: Model did not return "
            "a valid response.",

            None
        )


    # --------------------------------------------------------
    # EXTRACT DICTIONARY
    # --------------------------------------------------------

    try:

        output_dict = (

            ExtractCleanDict(
                output
            )
        )


    except Exception as e:

        return (

            "ERROR: Could not extract "
            "the result dictionary.\n\n"

            f"Reason: {e}\n\n"

            "Raw model output:\n\n"

            f"{output}",

            None
        )


    # --------------------------------------------------------
    # FORMAT OUTPUT
    # --------------------------------------------------------

    formatted_output = (

        FormatOutput(
            output_dict
        )
    )


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    if input_tokens > 0:

        time_per_input_token = (

            inference_time

            / input_tokens
        )


    else:

        time_per_input_token = (
            0.0
        )


    if generated_tokens > 0:

        time_per_generated_token = (

            inference_time

            / generated_tokens
        )


    else:

        time_per_generated_token = (
            0.0
        )


    # --------------------------------------------------------
    # BUILD FINAL OUTPUT
    # --------------------------------------------------------

    final_output = (

        formatted_output
    )


    final_output += (

        "========================================\n"

    )


    final_output += (

        "Inference Metrics\n"

    )


    final_output += (

        "========================================\n"

    )


    final_output += (

        f"Model type:                 "
        f"{model_type}\n"

    )


    final_output += (

        f"Model version:              "
        f"{version}\n"

    )


    final_output += (

        f"Input tokens:               "
        f"{input_tokens}\n"

    )


    final_output += (

        f"Generated tokens:           "
        f"{generated_tokens}\n"

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
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(

        OUTPUT_DIRECTORY,

        exist_ok=True
    )


    # --------------------------------------------------------
    # WRITE REAL OUTPUT FILE
    # --------------------------------------------------------

    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            final_output
        )


    # --------------------------------------------------------
    # RETURN RESULT
    #
    # First value -> Gradio result box
    # Second value -> downloadable output/output.txt
    # --------------------------------------------------------

    return (

        final_output,

        OUTPUT_FILE
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(

    title="Mamba Code Analyzer"

) as demo:


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.Markdown(

        """
# Mamba Code Analyzer

Select a model configuration, paste your Mamba code,
and run the analysis.

The analysis result is displayed below and saved as:

`output/output.txt`
        """
    )


    # --------------------------------------------------------
    # MODEL CONFIGURATION
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
    # CODE INPUT
    # --------------------------------------------------------

    code_input = gr.Textbox(

        label="Mamba Code",

        placeholder=(
            "Paste your Mamba code here..."
        ),

        lines=20
    )


    # --------------------------------------------------------
    # BUTTON
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

        lines=25,

        interactive=False
    )


    # --------------------------------------------------------
    # DOWNLOAD FILE
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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    demo.launch()
