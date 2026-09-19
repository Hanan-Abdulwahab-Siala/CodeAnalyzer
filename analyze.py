import argparse
import os

from model_service import (
    load_model,
    generate_inference_output,
    extract_clean_dict,
    format_output,
    is_model_loaded,
    get_loaded_model_info,
)
# ------------------------------------------------------------
DEFAULT_MODEL_FAMILY = "Mistral"
DEFAULT_LANGUAGE = "Mamba"
DEFAULT_MODEL_VERSION = 2
DEFAULT_MODEL_TYPE = "LoRA Adapter"
OUTPUT_DIRECTORY = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "output.txt")
# ------------------------------------------------------------
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
# ------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("Unified Mamba and Python Code Analyzer.")
    )
    parser.add_argument(
        "input_file",
        help="Path to the input code file.",
    )
    parser.add_argument(
        "--model-family",
        choices=["Mistral", "DeepSeek"],
        default=DEFAULT_MODEL_FAMILY,
        help=(
            "Model family: Mistral or DeepSeek. "
            f"Default: {DEFAULT_MODEL_FAMILY}"
        ),
    )
    parser.add_argument(
        "--language",
        choices=["Mamba", "Python"],
        default=DEFAULT_LANGUAGE,
        help=(
            "Language to analyze: Mamba or Python. "
            f"Default: {DEFAULT_LANGUAGE}"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["Flaw Detection", "Refactoring"],
        default=None,
        help=(
            "Python task. "
            "Mamba and DeepSeek always use "
            "Flaws + Refactoring. "
            "For Python choose Flaw Detection "
            "or Refactoring."
        ),
    )
    parser.add_argument(
        "--model-version",
        type=int,
        choices=[1, 2],
        default=DEFAULT_MODEL_VERSION,
        help=(
            "Model version: 1 or 2. "
            "DeepSeek always uses version 1. "
            "Python Refactoring automatically "
            "uses version 1."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["LoRA Adapter", "Full Model"],
        default=DEFAULT_MODEL_TYPE,
        help=("Model type: LoRA Adapter or Full Model."),
    )
    parser.add_argument(
        "--output-file",
        "--output",
        dest="output_file",
        default=OUTPUT_FILE,
        help=(f"Output file. Default: {OUTPUT_FILE}"),
    )
    return parser.parse_args()
# ------------------------------------------------------------
def main():
    args = parse_arguments()
    model_family = args.model_family
    language = args.language
    task = args.task
    model_version = args.model_version
    model_type = args.model_type
    input_file = args.input_file
    output_file = args.output_file
    if not os.path.dirname(output_file):
       output_file = os.path.join(OUTPUT_DIRECTORY, output_file,)
# ------------------------------------------------------------
    if model_family == "DeepSeek":
        if task is not None:
            print("ERROR: --task should not be used with DeepSeek.")
            raise SystemExit(1)
        task = "Flaws + Refactoring"
        model_version = 1
# ------------------------------------------------------------
    elif language == "Mamba":
        if task is not None:
            print("ERROR: --task should not be used with Mamba.")
            raise SystemExit(1)
        task = "Flaws + Refactoring"
# ------------------------------------------------------------
    elif language == "Python":
        if task is None:
            print("ERROR: Python requires --task.")
            print("Use either:")
            print('  --task "Flaw Detection"')
            print('  --task "Refactoring"')
            raise SystemExit(1)
        if task == "Refactoring":
            model_version = 1
# ------------------------------------------------------------
    if language == "Python":
        if not input_file.lower().endswith(".py"):
            if "." not in os.path.basename(input_file):
                input_file += ".py"
            else:
                print("ERROR: Python input file must have a .py extension.")
                raise SystemExit(1)
# ------------------------------------------------------------
    print("=" * 70)
    print("UNIFIED CODE ANALYZER")
    print("=" * 70)
    print(f"Model family: {model_family}")
    print(f"Input       : {input_file}")
    print(f"Language    : {language}")
    print(f"Task        : {task}")
    print(f"Version     : {model_version}")
    print(f"Model type  : {model_type}")
    print(f"Output      : {output_file}")
    print("=" * 70)
# ------------------------------------------------------------
    if not os.path.isfile(input_file):
        print(f"ERROR: File does not exist: {input_file}")
        raise SystemExit(1)
    if os.path.getsize(input_file) == 0:
        print("ERROR: Input file is empty.")
        raise SystemExit(1)
# ------------------------------------------------------------
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            code = file.read()
    except Exception as e:
        print("ERROR: Could not read input file.")
        print(f"Reason: {e}")
        raise SystemExit(1)
    print(f"Input characters: {len(code)}", flush=True)
# ------------------------------------------------------------
    print("Loading model...", flush=True)
    try:
        load_model(
            language=language,
            task=task,
            version=model_version,
            model_type=model_type,
            model_family=model_family,
        )
    except Exception as e:
        print("ERROR: Model loading failed.")
        print(f"Reason: {e}")
        raise SystemExit(1)
    if not is_model_loaded():
        print("ERROR: Model was not loaded.")
        raise SystemExit(1)
    loaded_info = get_loaded_model_info()
    print("Model loaded successfully.", flush=True)
    if loaded_info:
        print(f"Checkpoint: " f"{loaded_info['checkpoint']}", flush=True)
# ------------------------------------------------------------
    print()
    print("Starting inference...", flush=True)
    print("Maximum new tokens: 32768", flush=True)
    try:
        (
            output,
            input_tokens,
            generated_tokens,
            inference_time,
        ) = generate_inference_output(
            code,
            return_metrics=True,
        )
    except Exception as e:
        print("ERROR: Inference failed.")
        print(f"Reason: {e}")
        raise SystemExit(1)
    if output is None or not str(output).strip():
        print("ERROR: Model returned empty output.")
        raise SystemExit(1)
# ------------------------------------------------------------
    try:
        output_dict = extract_clean_dict(output)
    except Exception as e:
        output_directory = os.path.dirname(output_file)
        if not output_directory:
            output_directory = "."
        os.makedirs(output_directory, exist_ok=True)
        raw_file = os.path.join(output_directory, "raw_output.txt")
        try:
            with open(raw_file, "w", encoding="utf-8") as file:
                file.write(str(output))
        except Exception as write_error:
            print("WARNING: Could not save raw model output.")
            print(f"Reason: {write_error}")
        print("ERROR: Could not parse model output.")
        print(f"Reason: {e}")
        print(f"Raw output saved to: {raw_file}")
        raise SystemExit(1)
# ------------------------------------------------------------
    try:
        formatted_output = format_output(output_dict)
    except Exception as e:
        print("ERROR: Could not format model output.")
        print(f"Reason: {e}")
        raise SystemExit(1)
# ------------------------------------------------------------
    if input_tokens > 0:
        time_per_input_token = (inference_time / input_tokens)
    else:
        time_per_input_token = 0.0
    if generated_tokens > 0:
        time_per_generated_token = (inference_time / generated_tokens)
    else:
        time_per_generated_token = 0.0
# ------------------------------------------------------------
    final_output = formatted_output
    final_output += (
        "\n"
        "\n========================================\n"
        "Inference Metrics\n"
        "========================================\n"
        f"Language:                  {language}\n"
        f"Task:                      {task}\n"
        f"Model type:                {model_type}\n"
        f"Model version:             {model_version}\n"
        f"Input tokens:              {input_tokens}\n"
        f"Generated tokens:          {generated_tokens}\n"
        f"Inference time:            "
        f"{format_time(inference_time)}\n"
        f"Time per input token:      "
        f"{format_time(time_per_input_token)}\n"
        f"Time per generated token:  "
        f"{format_time(time_per_generated_token)}\n"
        "========================================\n"
    )
# ------------------------------------------------------------
    output_directory = os.path.dirname(output_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(final_output)
    except Exception as e:
        print("ERROR: Could not save output file.")
        print(f"Reason: {e}")
        raise SystemExit(1)
# ------------------------------------------------------------
    print()
    print(final_output)
    print(f"Output saved to: {output_file}")
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
# ------------------------------------------------------------

