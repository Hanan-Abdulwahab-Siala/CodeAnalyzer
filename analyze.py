import os
import sys

from model_service import (
    load_model,
    GenerateInferenceOutput,
    ExtractCleanDict,
    FormatOutput
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION = 2

MODEL_TYPE = (
    "LoRA Adapter"
)

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

def format_time(seconds):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (
            seconds % 3600
        ) // 60
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
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # START MESSAGE
    # --------------------------------------------------------

    print(
        "Starting analyzer...",
        flush=True
    )

    # --------------------------------------------------------
    # CHECK COMMAND-LINE ARGUMENTS
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            f"Usage: "
            f"python {sys.argv[0]} "
            f"<input_code_file>"
        )

        sys.exit(1)

    input_file = (
        sys.argv[1]
    )

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not os.path.isfile(
        input_file
    ):

        print(
            f"Error: input file "
            f"does not exist: "
            f"{input_file}"
        )

        sys.exit(1)

    if os.path.getsize(
        input_file
    ) == 0:

        print(
            f"Error: input file "
            f"is empty: "
            f"{input_file}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # READ INPUT FILE
    # --------------------------------------------------------

    print(
        f"Reading input file: "
        f"{input_file}",
        flush=True
    )

    try:

        with open(

            input_file,

            "r",

            encoding="utf-8"

        ) as file:

            code = (
                file.read()
            )

    except Exception as e:

        print(
            f"Error reading file: "
            f"{e}"
        )

        sys.exit(1)

    print(
        "Successfully read input file.",
        flush=True
    )

    print(
        f"Input characters: "
        f"{len(code)}",
        flush=True
    )

    print()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading model...",
        flush=True
    )

    try:

        load_model(

            version=MODEL_VERSION,

            model_type=MODEL_TYPE
        )

    except Exception as e:

        print()

        print(
            "ERROR: Model loading failed."
        )

        print(
            f"Reason: {e}"
        )

        sys.exit(1)

    print()

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    print(
        "Running inference...",
        flush=True
    )

    try:

        (

            output,

            input_tokens,

            generated_tokens,

            inference_time

        ) = GenerateInferenceOutput(
            code
        )

    except KeyboardInterrupt:

        print()

        print(
            "Inference interrupted "
            "by user."
        )

        sys.exit(1)

    except Exception as e:

        print()

        print(
            "ERROR: Inference failed."
        )

        print(
            f"Reason: {e}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # CHECK MODEL OUTPUT
    # --------------------------------------------------------

    if output is None:

        print()

        print(
            "ERROR: Model did not return "
            "a valid response."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # EXTRACT DICTIONARY
    # --------------------------------------------------------

    print()

    print(
        "Extracting model result...",
        flush=True
    )

    try:

        output_dict = (
            ExtractCleanDict(
                output
            )
        )

    except Exception as e:

        print()

        print(
            "ERROR: Could not extract "
            "the result dictionary."
        )

        print()

        print(
            f"Reason: {e}"
        )

        # ----------------------------------------------------
        # SAVE RAW OUTPUT
        # ----------------------------------------------------

        os.makedirs(

            OUTPUT_DIRECTORY,

            exist_ok=True
        )

        raw_output_file = (
            os.path.join(

                OUTPUT_DIRECTORY,

                "raw_output.txt"
            )
        )

        with open(

            raw_output_file,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(
                output
            )

        print()

        print(
            f"Raw model output "
            f"saved to: "
            f"{raw_output_file}"
        )

        sys.exit(1)

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

        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------

        file.write(

            formatted_output
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        file.write(

            "========================================\n"

        )

        file.write(

            "Inference Metrics\n"

        )

        file.write(

            "========================================\n"

        )

        file.write(

            f"Input tokens:               "
            f"{input_tokens}\n"

        )

        file.write(

            f"Generated tokens:           "
            f"{generated_tokens}\n"

        )

        file.write(

            f"Inference time:             "
            f"{format_time(inference_time)}\n"

        )

        file.write(

            f"Time per input token:       "
            f"{format_time(time_per_input_token)}\n"

        )

        file.write(

            f"Time per generated token:   "
            f"{format_time(time_per_generated_token)}\n"

        )

        file.write(

            "========================================\n"

        )

    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print()

    print(
        "========================================"
    )

    print(
        "Inference complete"
    )

    print(
        "========================================"
    )

    print(
        f"Input tokens:               "
        f"{input_tokens}"
    )

    print(
        f"Generated tokens:           "
        f"{generated_tokens}"
    )

    print(
        f"Inference time:             "
        f"{format_time(inference_time)}"
    )

    print(
        f"Time per input token:       "
        f"{format_time(time_per_input_token)}"
    )

    print(
        f"Time per generated token:   "
        f"{format_time(time_per_generated_token)}"
    )

    print()

    print(
        f"Output saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

