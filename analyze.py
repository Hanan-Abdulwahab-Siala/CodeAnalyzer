import argparse
import os

from model_service import (
    load_model,
    generate_inference_output,
    extract_clean_dict,
    format_output,
    is_model_loaded,
)
DEFAULT_MODEL_VERSION = 2
DEFAULT_MODEL_TYPE = "LoRA Adapter"
OUTPUT_DIRECTORY = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "output.txt")

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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Mamba code using the "
            "selected model."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the input code file.",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        choices=[1, 2],
        default=DEFAULT_MODEL_VERSION,
        help=(
            "Model version to use: 1 or 2. "
            f"Default: {DEFAULT_MODEL_VERSION}"
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=[
            "LoRA Adapter",
            "Full Model",
        ],
        default=DEFAULT_MODEL_TYPE,
        help=(
            "Model type to use: "
            "'LoRA Adapter' or 'Full Model'. "
            f"Default: '{DEFAULT_MODEL_TYPE}'"
        ),
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILE,
        help=(
            f"Path for the output file. "
            f"Default: {OUTPUT_FILE}"
        ),
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_file = args.input_file
    model_version = args.model_version
    model_type = args.model_type
    output_file = args.output_file
    print("Starting analyzer...", flush=True,)
    print(f"Model version: {model_version}", flush=True,)
    print(f"Model type: {model_type}", flush=True,)

    if not os.path.isfile(input_file):
        print(f"ERROR: File does not exist: " f"{input_file}")
        raise SystemExit(1)
    if os.path.getsize(input_file) == 0:
        print("ERROR: Input file is empty.")
        raise SystemExit(1)

    try:
        with open(input_file, "r", encoding="utf-8",) as file:
            code = file.read()
    except Exception as e:
        print("ERROR: Could not read input file.")
        print(f"Reason: {e}")
        raise SystemExit(1)
    print(f"Input characters: {len(code)}", flush=True,)

    print("Loading model...", flush=True,)
    try:
        load_model(version=model_version, model_type=model_type,)
    except Exception as e:
        print("ERROR: Model loading failed.")
        print(f"Reason: {e}")
        raise SystemExit(1)

    if not is_model_loaded():
        print("ERROR: Model was not loaded.")
        raise SystemExit(1)
    print("Model loaded successfully.", flush=True,)

    print("Starting inference...", flush=True,)
    try:
        (output, input_tokens, generated_tokens, inference_time,) = generate_inference_output(code)
    except Exception as e:
        print("ERROR: Inference failed.")
        print(f"Reason: {e}")
        raise SystemExit(1)
    if output is None or not str(output).strip():
        print("ERROR: Model returned empty output.")
        raise SystemExit(1)

    try:
        output_dict = extract_clean_dict(output)
    except Exception as e:
        output_directory = (os.path.dirname(output_file))
        if not output_directory:
            output_directory = "."
        os.makedirs(output_directory, exist_ok=True,)
        raw_file = os.path.join(output_directory, "raw_output.txt",)
        try:
            with open(raw_file, "w", encoding="utf-8",) as file:
                file.write(str(output))
        except Exception as write_error:
            print("WARNING: Could not save raw " "model output.")
            print(f"Reason: {write_error}")
        print("ERROR: Could not parse model output.")
        print(f"Reason: {e}")
        print(f"Raw output saved to: {raw_file}")
        raise SystemExit(1)

    formatted_output = format_output(output_dict)

    time_per_input_token = (inference_time / input_tokens if input_tokens > 0 else 0.0)
    time_per_generated_token = (inference_time / generated_tokens if generated_tokens > 0 else 0.0)

    final_output = formatted_output
    final_output += (
        "========================================\n"
        "Inference Metrics\n"
        "========================================\n"
        f"Model type:                 {model_type}\n"
        f"Model version:              {model_version}\n"
        f"Input tokens:               {input_tokens}\n"
        f"Generated tokens:           {generated_tokens}\n"
        f"Inference time:             "
        f"{format_time(inference_time)}\n"
        f"Time per input token:       "
        f"{format_time(time_per_input_token)}\n"
        f"Time per generated token:   "
        f"{format_time(time_per_generated_token)}\n"
        "========================================\n"
    )    

    output_directory = (os.path.dirname(output_file))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True,)
    try:
        with open(output_file, "w", encoding="utf-8",) as file:
            file.write(final_output)
    except Exception as e:
        print("ERROR: Could not save output file.")
        print(f"Reason: {e}")
        raise SystemExit(1)

    print()
    print(final_output)
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
