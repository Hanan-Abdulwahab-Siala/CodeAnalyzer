import os
import gradio as gr
from model_service import (
    load_model,
    unload_model,
    generate_inference_output,
    extract_clean_dict,
    format_output,
    append_inference_metrics,
    is_model_loaded,
    get_loaded_model_info,
)
# ------------------------------------------------------------
MODEL_FAMILIES = ["Mistral", "DeepSeek"]
LANGUAGES = ["Mamba", "Python"]
MODEL_TYPES = ["LoRA Adapter", "Full Model"]
# ------------------------------------------------------------
def _loaded_model_text(info):
    if not info:
        return "No model loaded."
    return (
        f"Model: {info['model_family']}\n"
        f"Language: {info['language']}\n"
        f"Task: {info['task']}\n"
        f"Version: {info['version']}\n"
        f"Model Type: {info['model_type']}\n"
        f"Checkpoint: {info['checkpoint']}"
    )
# ------------------------------------------------------------
def _configuration_values(model_family, language, task, model_version):
    if model_family == "DeepSeek":
        task_update = gr.update(
            choices=["Flaws + Refactoring"],
            value="Flaws + Refactoring",
            interactive=False,
        )
        version_update = gr.update(
            choices=["1"],
            value="1",
            interactive=False,
        )
        return (
            task_update,
            version_update,
        )
    if language == "Mamba":
        task_update = gr.update(
            choices=["Flaws + Refactoring"],
            value="Flaws + Refactoring",
            interactive=False,
        )
        version_update = gr.update(
            choices=["1", "2"],
            value=(
                str(model_version)
                if str(model_version) in {"1", "2"}
                else "2"
            ),
            interactive=True,
        )
        return (
            task_update,
            version_update,
        )
# ------------------------------------------------------------
    task_choices = [
        "Flaw Detection",
        "Refactoring",
    ]
    selected_task = (
        task
        if task in task_choices
        else "Flaw Detection"
    )
    if selected_task == "Refactoring":
        version_update = gr.update(
            choices=["1"],
            value="1",
            interactive=False,
        )
    else:
        version_update = gr.update(
            choices=["1", "2"],
            value=(
                str(model_version)
                if str(model_version) in {"1", "2"}
                else "2"
            ),
            interactive=True,
        )
    task_update = gr.update(
        choices=task_choices,
        value=selected_task,
        interactive=True,
    )
    return (
        task_update,
        version_update,
    )
# ------------------------------------------------------------
def configuration_changed(model_family, language, task, model_version, model_type):
    unload_model()
    task_update, version_update = _configuration_values(
        model_family=model_family,
        language=language,
        task=task,
        model_version=model_version,
    )
    return (
        task_update,
        version_update,
        "",
        None,
        "",
        "No model loaded.",
    )
# ------------------------------------------------------------
def model_family_changed(model_family, language, task, model_version, model_type):
    return configuration_changed(
        model_family=model_family,
        language=language,
        task=task,
        model_version=model_version,
        model_type=model_type,
    )
# ------------------------------------------------------------
def language_changed(model_family, language, task, model_version, model_type):
    return configuration_changed(
        model_family=model_family,
        language=language,
        task=task,
        model_version=model_version,
        model_type=model_type,
    )
# ------------------------------------------------------------
def task_changed(model_family, language, task, model_version, model_type):
    unload_model()
    if model_family == "DeepSeek":
        version_update = gr.update(
            choices=["1"],
            value="1",
            interactive=False,
        )
    elif (
        model_family == "Mistral"
        and language == "Python"
        and task == "Refactoring"
    ):
        version_update = gr.update(
            choices=["1"],
            value="1",
            interactive=False,
        )
    else:
        version_update = gr.update(
            choices=["1", "2"],
            value=(
                str(model_version)
                if str(model_version) in {"1", "2"}
                else "2"
            ),
            interactive=True,
        )
    return (
        version_update,
        "",
        None,
        "",
        "No model loaded.",
    )
# ------------------------------------------------------------
def version_changed(model_family, language, task, model_version, model_type):
    unload_model()
    return (
        "",
        None,
        "",
        "No model loaded.",
    )
# ------------------------------------------------------------
def model_type_changed(model_family, language, task, model_version, model_type):
    unload_model()
    return (
        "",
        None,
        "",
        "No model loaded.",
    )
# ------------------------------------------------------------
def load_selected_model(model_family, language, task, model_version, model_type):
    try:
        if model_family == "DeepSeek":
            task = "Flaws + Refactoring"
            model_version = 1
        elif language == "Mamba":
            task = "Flaws + Refactoring"
        elif (
            language == "Python"
            and task == "Refactoring"
        ):
            model_version = 1
        info = load_model(
            language=language,
            task=task,
            version=int(model_version),
            model_type=model_type,
            model_family=model_family,
        )
        status = (
            "Model loaded successfully.\n\n"
            + _loaded_model_text(info)
        )
        return (
            status,
            "",
            None,
            "",
        )
    except Exception as exc:
        return (
            "ERROR: Could not load model.\n\n"
            f"{type(exc).__name__}: {exc}",
            "",
            None,
            "",
        )
# ------------------------------------------------------------
def load_program_file(file_path):
    if not file_path:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return ""
# ------------------------------------------------------------
def analyze_code(code, language, task):
    if not is_model_loaded():
        return (
            "ERROR: No model is loaded.\n"
            "Please select the model configuration "
            "and press Load Model first."
        )
    if not code or not code.strip():
        return "Please enter or load code first."
    try:
        info = get_loaded_model_info()
        if info["language"] != language:
            return (
                "ERROR: The selected language does not "
                "match the loaded model.\n\n"
                f"Loaded language: {info['language']}\n"
                f"Selected language: {language}\n\n"
                "Please press Load Model."
            )
        if info["model_family"] == "Mistral":
            if (
                language == "Python"
                and info["task"] != task
            ):
                return (
                    "ERROR: The selected task does not "
                    "match the loaded model.\n\n"
                    f"Loaded task: {info['task']}\n"
                    f"Selected task: {task}\n\n"
                    "Please press Load Model."
                )
        (
            raw_output,
            input_tokens,
            generated_tokens,
            inference_time,
        ) = generate_inference_output(code, return_metrics=True)
        if raw_output is None or not str(raw_output).strip():
            return (
                "ERROR: Model returned empty output."
            )
        output_dict = extract_clean_dict(raw_output)
        final_output = format_output(output_dict)
        final_output = append_inference_metrics(
            final_output=final_output,
            language=language,
            task=info["task"],
            model_type=info["model_type"],
            model_version=info["version"],
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            inference_time=inference_time,
        )
        os.makedirs("output", exist_ok=True)
        with open("output/output.txt", "w", encoding="utf-8") as file:
            file.write(final_output)
        return final_output
    except Exception as exc:
        return (
            "ERROR during analysis.\n\n"
            f"{type(exc).__name__}: {exc}"
        )
# ------------------------------------------------------------
def clear_program():
    return (
        "",
        None,
        "",
    )
# ------------------------------------------------------------
with gr.Blocks(title="Unified Code Analyzer") as app:
    gr.Markdown(
        "# Unified Code Analyzer"
    )
    gr.Markdown(
        "Select the model configuration, press "
        "**Load Model**, then analyze your code."
    )
    with gr.Row():
        model_family = gr.Dropdown(
            label="Model",
            choices=MODEL_FAMILIES,
            value="Mistral",
            scale=1,
        )
        language = gr.Dropdown(
            label="Language",
            choices=LANGUAGES,
            value="Mamba",
            scale=1,
        )
    with gr.Row():
        task = gr.Dropdown(
            label="Task",
            choices=["Flaws + Refactoring"],
            value="Flaws + Refactoring",
            interactive=False,
            scale=1,
        )
        model_version = gr.Dropdown(
            label="Model Version",
            choices=["1", "2"],
            value="2",
            interactive=True,
            scale=1,
        )
        model_type = gr.Dropdown(
            label="Model Type",
            choices=MODEL_TYPES,
            value="LoRA Adapter",
            scale=1,
        )
    with gr.Row():
        load_button = gr.Button(
            "Load Model",
            variant="primary",
        )
    model_status = gr.Textbox(
        label="Loaded Model",
        value="No model loaded.",
        lines=4,
        max_lines=8,
        interactive=False,
    )
    program_file = gr.File(
        label="Load Program",
        file_types=[
            ".txt",
            ".py",
            ".mamba",
        ],
        type="filepath",
    )
    code_input = gr.Textbox(
        label="Program",
        placeholder=(
            "Paste your code here or select a program file "
            "above. You can modify the code before Analyze."
        ),
        lines=16,
        max_lines=30,
        interactive=True,
    )
    with gr.Row():
        analyze_button = gr.Button(
            "Analyze",
            variant="primary",
        )
        clear_button = gr.Button(
            "Clear",
            variant="secondary",
        )
    output_box = gr.Textbox(
        label="Analysis Output",
        lines=18,
        max_lines=35,
        interactive=False,
    )
    model_family.change(
        fn=model_family_changed,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            task,
            model_version,
            code_input,
            program_file,
            output_box,
            model_status,
        ],
    )
    language.change(
        fn=language_changed,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            task,
            model_version,
            code_input,
            program_file,
            output_box,
            model_status,
        ],
    )
    task.change(
        fn=task_changed,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            model_version,
            code_input,
            program_file,
            output_box,
            model_status,
        ],
    )
    model_version.change(
        fn=version_changed,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            code_input,
            program_file,
            output_box,
            model_status,
        ],
    )
    model_type.change(
        fn=model_type_changed,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            code_input,
            program_file,
            output_box,
            model_status,
        ],
    )
    program_file.change(
        fn=load_program_file,
        inputs=program_file,
        outputs=code_input,
    )
    load_button.click(
        fn=load_selected_model,
        inputs=[
            model_family,
            language,
            task,
            model_version,
            model_type,
        ],
        outputs=[
            model_status,
            code_input,
            program_file,
            output_box,
        ],
    )
    analyze_button.click(
        fn=analyze_code,
        inputs=[
            code_input,
            language,
            task,
        ],
        outputs=output_box,
    )
    clear_button.click(
        fn=clear_program,
        inputs=[],
        outputs=[
            code_input,
            program_file,
            output_box,
        ],
    )
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port)
# ------------------------------------------------------------
