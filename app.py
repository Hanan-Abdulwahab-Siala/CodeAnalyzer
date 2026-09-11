import os
import gradio as gr
from model_service import (
    load_model,
    generate_inference_output,
    extract_clean_dict,
    format_output,
    get_hardware_info
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "output.txt")

# ============================================================
# FORMAT TIME
# ============================================================

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

# ============================================================
# READ UPLOADED FILE
# ============================================================

def load_file_into_textbox(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        with open(uploaded_file, "r", encoding="utf-8") as file:
            code = file.read()
        return code
    except Exception as e:
        return f"# ERROR: Could not read uploaded file.\n\n# Reason: {e}"

# ============================================================
# ANALYZE CODE
# ============================================================

def analyze_code(code, model_version, model_type):
    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    if code is None:
        return "ERROR: No code was provided.", None
    if not code.strip():
        return "ERROR: Code input is empty.", None

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    try:
        version = int(model_version)
    except Exception:
        return "ERROR: Invalid model version.", None

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:
        load_model(version=version, model_type=model_type)
    except Exception as e:
        return f"ERROR: Model loading failed.\n\nReason: {e}", None

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    try:
        (output, input_tokens, generated_tokens, inference_time) = generate_inference_output(code)
    except Exception as e:
        return f"ERROR: Inference failed.\n\nReason: {e}", None

    # --------------------------------------------------------
    # PARSE RESULT
    # --------------------------------------------------------

    try:
        output_dict = extract_clean_dict(output)
    except Exception as e:
        return f"ERROR: Could not extract the result dictionary.\n\nReason: {e}\n\nRaw model output:\n\n{output}", None

    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    formatted_output = format_output(output_dict)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if input_tokens > 0:
        time_per_input_token = inference_time / input_tokens
    else:
        time_per_input_token = 0.0
    if generated_tokens > 0:
        time_per_generated_token = inference_time / generated_tokens
    else:
        time_per_generated_token = 0.0

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    final_output = formatted_output
    final_output += "========================================\nInference Metrics\n========================================\n"
    final_output += f"Model type:                 {model_type}\n"
    final_output += f"Model version:              {version}\n"
    final_output += f"Input tokens:               {input_tokens}\n"
    final_output += f"Generated tokens:           {generated_tokens}\n"
    final_output += f"Inference time:             {format_time(inference_time)}\n"
    final_output += f"Time per input token:       {format_time(time_per_input_token)}\n"
    final_output += f"Time per generated token:   {format_time(time_per_generated_token)}\n"
    final_output += "========================================\n"

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(final_output)
    return final_output, OUTPUT_FILE

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="Mamba Code Analyzer") as demo:
    gr.Markdown("""
# Mamba Code Analyzer

Upload a Mamba code file or paste code manually.

The model runs on the GPU of the machine running this application.
""")

    # --------------------------------------------------------
    # HARDWARE
    # --------------------------------------------------------

    hardware_info = gr.Textbox(label="Hardware", value=get_hardware_info(), interactive=False, lines=6)

    # --------------------------------------------------------
    # MODEL SETTINGS
    # --------------------------------------------------------

    with gr.Row():
        model_type = gr.Dropdown(choices=["LoRA Adapter", "Full Model"], value="LoRA Adapter", label="Model Type")
        model_version = gr.Dropdown(choices=["1", "2"], value="2", label="Model Version")

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    uploaded_file = gr.File(label="Upload Mamba Code File", type="filepath")

    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------

    code_input = gr.Textbox(label="Mamba Code", placeholder="Upload a file above or paste your Mamba code here...", lines=25)

    # --------------------------------------------------------
    # LOAD FILE
    # --------------------------------------------------------

    uploaded_file.change(fn=load_file_into_textbox, inputs=uploaded_file, outputs=code_input)

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    analyze_button = gr.Button("Analyze Code", variant="primary")

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_output = gr.Textbox(label="Analysis Result", lines=30, interactive=False)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output_file = gr.File(label="Download Analysis Output", interactive=False)

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    analyze_button.click(fn=analyze_code, inputs=[code_input, model_version, model_type], outputs=[result_output, output_file])

# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    print("=" * 60)
    print("Starting Mamba Code Analyzer")
    print(f"Gradio server: http://0.0.0.0:{port}")
    print(f"Port: {port}")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
