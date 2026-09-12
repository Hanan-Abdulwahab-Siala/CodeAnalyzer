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
    get_loaded_model_info
)
OUTPUT_DIRECTORY = "output"
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    result = ""
    if hours > 0:
        result += f"{hours}h "
    if minutes > 0 or hours > 0:
        result += f"{minutes}m "
    result += f"{remaining_seconds:.6f}s"
    return result
def load_file_into_textbox(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        with open(uploaded_file, "r", encoding="utf-8") as file:
            code = file.read()
        return code
    except Exception as e:
        return (
            "# ERROR: Could not read uploaded file.\n\n"
            f"# Reason: {e}"
        )
def get_model_status_text():
    if not is_model_loaded():
        return (
            "No model is currently loaded.\n\n"
            "Select Model Type and Model Version, "
            "then click Load Model."
        )
    info = get_loaded_model_info()
    return (
        "MODEL CURRENTLY LOADED\n\n"
        f"Model type: {info['model_type']}\n"
        f"Model version: {info['version']}"
    )

def load_selected_model(model_version, model_type):
    try:
        version = int(model_version)
    except Exception:
        return "ERROR: Invalid model version."
    try:
        result = load_model(version=version, model_type=model_type)
    except Exception as e:
        return (
            "ERROR: Model loading failed.\n\n"
            f"Reason: {e}"
        )
    if result == "already_loaded":
        return (
            "MODEL ALREADY LOADED\n\n"
            f"Model type: {model_type}\n"
            f"Model version: {version}\n\n"
            "The existing GPU model is being reused."
        )
    return (
        "MODEL LOADED SUCCESSFULLY\n\n"
        f"Model type: {model_type}\n"
        f"Model version: {version}\n\n"
        "The model remains loaded in GPU memory "
        "until another model is explicitly loaded."
    )

def clear_program():
    return (
        None,     # uploaded file
        "",       # code
        "",       # result
        None      # output file
    )
def analyze_code(code):
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
    if not is_model_loaded():
        return (
            "ERROR: No model is loaded.\n\n"
            "Select Model Type and Model Version, "
            "then click Load Model first.",
            None
        )
    try:
        (output, input_tokens, generated_tokens, inference_time) = generate_inference_output(code)
    except Exception as e:
        return (
            "ERROR: Inference failed.\n\n"
            f"Reason: {e}",
            None
        )
    try:
        output_dict = extract_clean_dict(output)
    except Exception as e:
        return (
            "ERROR: Could not extract the result dictionary.\n\n"
            f"Reason: {e}\n\n"
            "Raw model output:\n\n"
            f"{output}",
            None
        )
    formatted_output = format_output(output_dict)
    if input_tokens > 0:
        time_per_input_token = (inference_time / input_tokens)
    else:
        time_per_input_token = 0.0
    if generated_tokens > 0:
        time_per_generated_token = (inference_time / generated_tokens)
    else:
        time_per_generated_token = 0.0
    model_info = get_loaded_model_info()
    final_output = formatted_output
    final_output += (
        "========================================\n"
        "Inference Metrics\n"
        "========================================\n"
    )
    final_output += (
        f"Model type:                 "
        f"{model_info['model_type']}\n"
    )
    final_output += (
        f"Model version:              "
        f"{model_info['version']}\n"
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
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        unique_id = uuid.uuid4().hex
        output_file_path = os.path.join(OUTPUT_DIRECTORY, f"output_{unique_id}.txt")
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(final_output)
    except Exception as e:
        return (
            "ERROR: Could not save output file.\n\n"
            f"Reason: {e}",
            None
        )
    return (
        final_output,
        output_file_path
    )
with gr.Blocks(title="Mamba Code Analyzer") as demo:
    gr.Markdown(
        """
# Mamba Code Analyzer

Select the model configuration and click **Load Model**.

After the model is loaded, upload or paste Mamba code and
click **Analyze Code**.

The model remains in GPU memory until another model is
explicitly loaded.
"""
    )
    hardware_info = gr.Textbox(label="Hardware", value=get_hardware_info(), interactive=False, lines=12)
    with gr.Row():
        model_type = gr.Dropdown(choices=["LoRA Adapter", "Full Model"], value="LoRA Adapter", label="Model Type")
        model_version = gr.Dropdown(choices=["1", "2"], value="2", label="Model Version")
    load_model_button = gr.Button("Load Model", variant="primary")
    model_status = gr.Textbox(label="Model Status", value=get_model_status_text(), interactive=False, lines=7)
    uploaded_file = gr.File(label="Upload Mamba Code File", type="filepath")
    code_input = gr.Textbox(label="Mamba Code", placeholder=("Upload a file above or paste " "your Mamba code here..."), lines=25)
    uploaded_file.change(fn=load_file_into_textbox, inputs=uploaded_file, outputs=code_input)
    with gr.Row():
        analyze_button = gr.Button("Analyze Code", variant="primary")
        clear_button = gr.Button("Clear / New Program", variant="secondary")
    result_output = gr.Textbox(label="Analysis Result", lines=30, interactive=False)
    output_file = gr.File(label="Download Analysis Output", interactive=False)
    load_model_button.click(fn=load_selected_model, inputs=[model_version, model_type], outputs=[model_status])
    analyze_button.click(fn=analyze_code, inputs=[code_input], outputs=[result_output, output_file])
    clear_button.click(fn=clear_program, inputs=[], outputs=[uploaded_file, code_input, result_output, output_file])
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print("=" * 60)
    print("Starting Mamba Code Analyzer")
    print("Model loading is manual.")
    print(f"Gradio server: http://0.0.0.0:{port}")
    print(f"Port: {port}")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
