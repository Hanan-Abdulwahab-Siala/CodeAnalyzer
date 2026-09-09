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
# READ UPLOADED FILE
# ============================================================

def read_uploaded_file(
    uploaded_file
):

    # --------------------------------------------------------
    # NO FILE
    # --------------------------------------------------------

    if uploaded_file is None:

        return ""


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    try:

        with open(

            uploaded_file,

            "r",

            encoding="utf-8"

        ) as file:

            content = file.read()


        return content


    except Exception as e:

        return (
            f"ERROR READING FILE:\n"
            f"{e}"
        )


# ============================================================
# TEST ANALYSIS FUNCTION
#
# This function does NOT load Mistral.
# This function does NOT require a GPU.
# It only tests the Gradio interface.
# ============================================================

def test_analyze_code(

    uploaded_file,

    pasted_code,

    model_version,

    model_type

):

    # --------------------------------------------------------
    # DETERMINE INPUT SOURCE
    # --------------------------------------------------------

    code = ""

    input_source = ""


    # --------------------------------------------------------
    # PRIORITY 1:
    # UPLOADED FILE
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            with open(

                uploaded_file,

                "r",

                encoding="utf-8"

            ) as file:

                code = file.read()


            input_source = (
                "Uploaded File"
            )


        except Exception as e:

            return (

                f"ERROR: Could not read "
                f"uploaded file.\n\n"
                f"Reason: {e}",

                None
            )


    # --------------------------------------------------------
    # PRIORITY 2:
    # PASTED CODE
    # --------------------------------------------------------

    elif pasted_code is not None:

        if pasted_code.strip():

            code = (
                pasted_code
            )

            input_source = (
                "Manual Text Input"
            )


    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not code.strip():

        return (

            "ERROR: No code was provided.\n\n"
            "Please either:\n"
            "- Upload a code file, or\n"
            "- Paste code into the text box.",

            None
        )


    # --------------------------------------------------------
    # CREATE TEST OUTPUT
    # --------------------------------------------------------

    final_output = f"""
========================================
MAMBA CODE ANALYZER
GRADIO INTERFACE TEST
========================================

The Gradio interface is working.

No GPU was used.

No Mistral model was loaded.

No LoRA model was loaded.

No real AI inference was performed.

========================================
SELECTED CONFIGURATION
========================================

Model Type:
{model_type}

Model Version:
{model_version}

Input Source:
{input_source}

========================================
INPUT CODE
========================================

{code}

========================================
TEST RESULT
========================================

The following components were tested:

- Model Type selection
- Model Version selection
- File upload
- Manual code input
- Analyze button
- Output display
- output/output.txt creation
- Output file download

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

This version allows you to test the complete interface
without loading the AI model.

You can:

- Select the model type
- Select the model version
- Upload a code file
- Paste code manually
- Click Analyze Code
- View the result
- Download output/output.txt

This test version does NOT require a GPU.
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
    # INPUT SECTION
    # --------------------------------------------------------

    gr.Markdown(

        """
## Input Code

You can use either method:

1. Upload a code file, OR
2. Paste the code manually.

If you provide both, the uploaded file
will be used.
        """
    )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = gr.File(

        label="Upload Input Code File",

        type="filepath"
    )


    # --------------------------------------------------------
    # MANUAL CODE INPUT
    # --------------------------------------------------------

    pasted_code = gr.Textbox(

        label="Or Paste Code Manually",

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
    # RESULT OUTPUT
    # --------------------------------------------------------

    result_output = gr.Textbox(

        label="Analysis Result",

        lines=25,

        interactive=False
    )


    # --------------------------------------------------------
    # DOWNLOAD OUTPUT
    # --------------------------------------------------------

    output_file = gr.File(

        label="Download Output File"
    )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(

        fn=test_analyze_code,

        inputs=[

            uploaded_file,

            pasted_code,

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
