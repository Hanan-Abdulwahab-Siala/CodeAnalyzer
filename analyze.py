import os
import sys

from model_service import (
    load_model,
    generate_inference_output,
    extract_clean_dict,
    format_output
)


MODEL_VERSION = 2

MODEL_TYPE = (
    "LoRA Adapter"
)


OUTPUT_DIRECTORY = (
    "output"
)


OUTPUT_FILE = os.path.join(

    OUTPUT_DIRECTORY,

    "output.txt"
)


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(
    seconds
):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
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

    print(
        "Starting analyzer...",
        flush=True
    )


    if len(sys.argv) != 2:

        print(

            f"Usage: python "
            f"{sys.argv[0]} "
            f"<input_code_file>"
        )

        sys.exit(1)


    input_file = (
        sys.argv[1]
    )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.isfile(
        input_file
    ):

        print(

            f"ERROR: File does not exist: "
            f"{input_file}"
        )

        sys.exit(1)


    if os.path.getsize(
        input_file
    ) == 0:

        print(

            "ERROR: Input file is empty."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    with open(

        input_file,

        "r",

        encoding="utf-8"

    ) as file:

        code = (
            file.read()
        )


    print(

        f"Input characters: "
        f"{len(code)}"
    )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    load_model(

        version=MODEL_VERSION,

        model_type=MODEL_TYPE
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    (

        output,

        input_tokens,

        generated_tokens,

        inference_time

    ) = (

        generate_inference_output(
            code
        )
    )


    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    try:

        output_dict = (

            extract_clean_dict(
                output
            )
        )


    except Exception as e:

        os.makedirs(

            OUTPUT_DIRECTORY,

            exist_ok=True
        )


        raw_file = (

            os.path.join(

                OUTPUT_DIRECTORY,

                "raw_output.txt"
            )
        )


        with open(

            raw_file,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(
                output
            )


        print(

            "ERROR: Could not parse model output."
        )

        print(
            f"Reason: {e}"
        )

        print(
            f"Raw output saved to: "
            f"{raw_file}"
        )

        sys.exit(1)


    formatted_output = (

        format_output(
            output_dict
        )
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    time_per_input_token = (

        inference_time / input_tokens

        if input_tokens > 0

        else 0.0
    )


    time_per_generated_token = (

        inference_time / generated_tokens

        if generated_tokens > 0

        else 0.0
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    final_output = (

        formatted_output
    )


    final_output += (

        "========================================\n"

        "Inference Metrics\n"

        "========================================\n"

        f"Model type:                 "
        f"{MODEL_TYPE}\n"

        f"Model version:              "
        f"{MODEL_VERSION}\n"

        f"Input tokens:               "
        f"{input_tokens}\n"

        f"Generated tokens:           "
        f"{generated_tokens}\n"

        f"Inference time:             "
        f"{format_time(inference_time)}\n"

        f"Time per input token:       "
        f"{format_time(time_per_input_token)}\n"

        f"Time per generated token:   "
        f"{format_time(time_per_generated_token)}\n"

        "========================================\n"
    )


    os.makedirs(

        OUTPUT_DIRECTORY,

        exist_ok=True
    )


    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            final_output
        )


    print()

    print(
        final_output
    )

    print(

        f"Output saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()
