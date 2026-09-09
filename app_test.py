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
# LOAD FILE CONTENT INTO TEXTBOX
# ============================================================

def load_file_into_textbox(
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

            code = (
                file.read()
            )


        # ----------------------------------------------------
        # RETURN FILE CONTENT
        #
        # This content will automatically appear
        # in the manual code textbox.
        # ----------------------------------------------------

        return code


    except Exception as e:

        return (

            f"# ERROR: Could not read "
            f"uploaded file.\n\n"
            f"# Reason: {e}"
        )


# ============================================================
# TEST ANALYSIS FUNCTION
#
# This does NOT load Mistral.
# This does NOT require a GPU.
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

            "ERROR: Code input is empty.\n\n"
            "Please upload a file or paste "
            "code manually.",

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

========================================
CODE TO ANALYZE
========================================

{code}

========================================
TEST RESULT
========================================

The following components were tested:

- Model Type selection
- Model Version selection
- File upload
- Automatic file content loading
- Manual code editing
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

You can either:

1. Upload a code file.
   Its content will automatically appear in the code box.

OR

2. Paste/type code directly into the code box.

You can edit the code before clicking Analyze Code.

This test version does not load a model and does not
require a GPU.
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

        label="Upload Input Code File",

        type="filepath"
    )


    # --------------------------------------------------------
    # CODE TEXTBOX
    # --------------------------------------------------------

    code_input = gr.Textbox(

        label="Code to Analyze",

        placeholder=(
            "Upload a file above or paste "
            "your Mamba code here..."
        ),

        lines=25
    )


    # --------------------------------------------------------
    # AUTOMATICALLY LOAD FILE CONTENT
    # --------------------------------------------------------

    uploaded_file.change(

        fn=load_file_into_textbox,

        inputs=[

            uploaded_file

        ],

        outputs=[

            code_input

        ]
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
