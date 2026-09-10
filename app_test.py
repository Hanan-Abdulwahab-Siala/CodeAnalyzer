import gradio as gr


# ============================================================
# TEST FILE
# ============================================================

def test_file_loading(
    uploaded_file
):

    if uploaded_file is None:

        return (

            "No file selected."
        )


    try:

        with open(

            uploaded_file,

            "r",

            encoding="utf-8"

        ) as file:

            code = (

                file.read()
            )


        return (

            "File upload test passed.\n\n"

            f"Characters: "
            f"{len(code)}\n\n"

            "========================================\n"

            "FILE CONTENT\n"

            "========================================\n\n"

            f"{code}"
        )


    except Exception as e:

        return (

            "File upload test failed.\n\n"

            f"Reason: {e}"
        )


# ============================================================
# UI
# ============================================================

with gr.Blocks(

    title="Mamba Code Analyzer - Test"

) as demo:


    gr.Markdown(

        """
# Mamba Code Analyzer - UI Test

This application only tests Gradio file upload.

No LLM is loaded.

No GPU is required.

No inference is performed.
"""
    )


    uploaded_file = gr.File(

        label="Upload Input Code File",

        type="filepath"
    )


    result = gr.Textbox(

        label="Test Result",

        lines=30,

        interactive=False
    )


    uploaded_file.change(

        fn=test_file_loading,

        inputs=uploaded_file,

        outputs=result
    )


if __name__ == "__main__":

    demo.launch()
