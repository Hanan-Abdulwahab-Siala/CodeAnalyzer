import os

import gradio as gr


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIRECTORY = "output"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "output.txt"
)


# ============================================================
# TEST ANALYSIS FUNCTION
#
# This does NOT load Mistral.
# This does NOT require a GPU.
# It only tests the Gradio interface.
# ============================================================

def test_analyze_code(
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
    # CREATE DEMO OUTPUT
    # --------------------------------------------------------

    final_output = f"""
========================================
GRADIO INTERFACE TEST
========================================

The Gradio interface is working.

No GPU was used.

No Mistral model was loaded.

Selected configuration:

Model Type: {model_type}

Model Version: {model_version}

========================================
INPUT CODE
========================================

{code}

========================================
TEST RESULT
========================================

The following interface components worked:

- Model Type selection
- Model Version selection
- Code input
- Analyze button
- Output display
- Output file creation
- Download file

The real AI inference was NOT executed.

========================================
"""


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )


    # --------------------------------------------------------
    # WRITE OUTPUT FILE
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
    # --------------------------------------------------------

    return (
        final_output,
        OUTPUT_FILE
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Mamba Code Analyzer - Interface Test"
) as demo:


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.Markdown(
        """
# Mamba Code Analyzer

## Gradio Interface Test Mode

This version tests the interface only.

- No GPU required
- No Mistral model loaded
- No LoRA model loaded
- No real inference performed

You can test all menus and buttons.
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
    # ANALYZE BUTTON
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

        label="Download Analysis Output"
    )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(

        fn=test_analyze_code,

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
