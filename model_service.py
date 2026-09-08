import time
import re
import ast

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel


# ============================================================
# GPU / DEVICE CONFIGURATION
# ============================================================

DEVICE = "cpu"
DTYPE = torch.float32

GPU_NAME = "CPU"
GPU_CAPABILITY = None
GPU_FAMILY = "CPU"


if torch.cuda.is_available():

    DEVICE = "cuda:0"

    GPU_NAME = (
        torch.cuda.get_device_name(0)
    )

    GPU_CAPABILITY = (
        torch.cuda.get_device_capability(0)
    )

    GPU_MAJOR = GPU_CAPABILITY[0]
    GPU_MINOR = GPU_CAPABILITY[1]

    # --------------------------------------------------------
    # A100 / Ampere
    #
    # A100 is generally compute capability sm_80.
    # --------------------------------------------------------

    if GPU_MAJOR == 8:

        GPU_FAMILY = "Ampere (A100 compatible)"

    # --------------------------------------------------------
    # B200 / Blackwell
    #
    # B200 systems can report sm_100.
    # --------------------------------------------------------

    elif GPU_MAJOR >= 10:

        GPU_FAMILY = "Blackwell (B200 compatible)"

    # --------------------------------------------------------
    # Other CUDA GPUs
    # --------------------------------------------------------

    else:

        GPU_FAMILY = "Other CUDA GPU"

    # --------------------------------------------------------
    # Select dtype based on actual hardware capability.
    #
    # A100 and B200 both support BF16.
    # --------------------------------------------------------

    if torch.cuda.is_bf16_supported():

        DTYPE = torch.bfloat16

    else:

        DTYPE = torch.float16


print(
    "========================================",
    flush=True
)

print(
    "GPU configuration",
    flush=True
)

print(
    "========================================",
    flush=True
)

print(
    f"GPU:                {GPU_NAME}",
    flush=True
)

print(
    f"GPU family:         {GPU_FAMILY}",
    flush=True
)

if GPU_CAPABILITY is not None:

    print(
        f"Compute capability: "
        f"sm_{GPU_CAPABILITY[0]}"
        f"{GPU_CAPABILITY[1]}",
        flush=True
    )

print(
    f"PyTorch version:    "
    f"{torch.__version__}",
    flush=True
)

print(
    f"CUDA version:       "
    f"{torch.version.cuda}",
    flush=True
)

bf16_supported = (
    torch.cuda.is_bf16_supported()
    if torch.cuda.is_available()
    else False
)

print(
    f"BF16 supported:     {bf16_supported}",
    flush=True
)

print(
    f"Using dtype:        "
    f"{DTYPE}",
    flush=True
)

print(
    "========================================",
    flush=True
)


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

# Keep the same maximum generation size as the original program.
MAX_NEW_TOKENS = 32768

# Original program uses deterministic generation.
DO_SAMPLE = False


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

FULL_MODEL_V1 = (
    "HA-Siala/Mamba-full-v0.1"
)

FULL_MODEL_V2 = (
    "HA-Siala/Mamba-full-v0.2"
)

LORA_MODEL_V1 = (
    "HA-Siala/Mamba-v0.1"
)

LORA_MODEL_V2 = (
    "HA-Siala/Mamba-v0.2"
)


BASE_MODEL = (
    "mistralai/Mistral-7B-v0.3"
)


# ============================================================
# GLOBAL MODEL / TOKENIZER
# ============================================================

model = None
tokenizer = None


# ============================================================
# MODEL LOADING HELPER
# ============================================================

def LoadCausalLM(
    model_path,
    **kwargs
):

    """
    Load a causal language model.

    Newer Transformers versions use `dtype`.
    Older versions may use `torch_dtype`.

    Try dtype first, then fall back if necessary.
    """

    try:

        return (
            AutoModelForCausalLM
            .from_pretrained(
                model_path,
                dtype=DTYPE,
                **kwargs
            )
        )

    except TypeError:

        return (
            AutoModelForCausalLM
            .from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                **kwargs
            )
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    version=2,
    model_type="LoRA Adapter"
):

    global model
    global tokenizer

    print(
        f"Loading model "
        f"(version={version}, "
        f"type={model_type})...",
        flush=True
    )

    # --------------------------------------------------------
    # LoRA ADAPTER
    # --------------------------------------------------------

    if model_type == "LoRA Adapter":

        if version == 1:

            adapter_path = (
                LORA_MODEL_V1
            )

        elif version == 2:

            adapter_path = (
                LORA_MODEL_V2
            )

        else:

            raise ValueError(
                f"Unsupported model version: "
                f"{version}"
            )

        print(
            f"Base model:   "
            f"{BASE_MODEL}",
            flush=True
        )

        print(
            f"LoRA adapter: "
            f"{adapter_path}",
            flush=True
        )

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        print(
            "Loading tokenizer...",
            flush=True
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                BASE_MODEL,
                use_fast=True
            )
        )

        # Preserve original behavior.
        if tokenizer.unk_token is not None:

            tokenizer.pad_token = (
                tokenizer.unk_token
            )

        elif tokenizer.pad_token is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        tokenizer.padding_side = "left"

        print(
            "Tokenizer loaded.",
            flush=True
        )

        # ----------------------------------------------------
        # BASE MODEL
        # ----------------------------------------------------

        print(
            "Loading base model...",
            flush=True
        )

        model = LoadCausalLM(

            BASE_MODEL,

            device_map="auto",

            low_cpu_mem_usage=True
        )

        print(
            "Base model loaded.",
            flush=True
        )

        # ----------------------------------------------------
        # LORA ADAPTER
        # ----------------------------------------------------

        print(
            "Loading LoRA adapter...",
            flush=True
        )

        model = (
            PeftModel
            .from_pretrained(

                model,

                adapter_path,

                is_trainable=False
            )
        )

        print(
            "LoRA adapter loaded.",
            flush=True
        )

    # --------------------------------------------------------
    # FULL MODEL
    # --------------------------------------------------------

    elif model_type == "Full Model":

        if version == 1:

            model_path = (
                FULL_MODEL_V1
            )

        elif version == 2:

            model_path = (
                FULL_MODEL_V2
            )

        else:

            raise ValueError(
                f"Unsupported model version: "
                f"{version}"
            )

        print(
            f"Full model: "
            f"{model_path}",
            flush=True
        )

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        print(
            "Loading tokenizer...",
            flush=True
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path,
                use_fast=True
            )
        )

        if tokenizer.unk_token is not None:

            tokenizer.pad_token = (
                tokenizer.unk_token
            )

        elif tokenizer.pad_token is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        tokenizer.padding_side = "left"

        print(
            "Tokenizer loaded.",
            flush=True
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print(
            "Loading full model...",
            flush=True
        )

        model = LoadCausalLM(

            model_path,

            device_map="auto",

            low_cpu_mem_usage=True
        )

        print(
            "Full model loaded.",
            flush=True
        )

    # --------------------------------------------------------
    # INVALID MODEL TYPE
    # --------------------------------------------------------

    else:

        raise ValueError(

            f"Unsupported model type: "
            f"{model_type}. "

            f"Use 'LoRA Adapter' "
            f"or 'Full Model'."
        )

    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()

    print(
        "Model loaded successfully.",
        flush=True
    )

    return model


# ============================================================
# ORIGINAL PROMPT
# ============================================================

def GeneratePrompt(content):

    Instruction = (
        "Analyze the following Mamba code, "
        "detect any flaws, with a short explanation, "
        "and give one or more corrected versions of "
        "the code. Ensure that none of the refactored "
        "options repeat or reproduce the original code; "
        "only the improved version should be shown:"
    )

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task:

### Instruction:
{Instruction}

### Code:
{content}
### Response:
"""

    return prompt


# ============================================================
# INFERENCE
# ============================================================

def GenerateInferenceOutput(text):

    global model
    global tokenizer

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        raise RuntimeError(
            "Model has not been loaded."
        )

    if tokenizer is None:

        raise RuntimeError(
            "Tokenizer has not been loaded."
        )

    # --------------------------------------------------------
    # ORIGINAL TOKENIZER BEHAVIOR
    # --------------------------------------------------------

    if tokenizer.unk_token is not None:

        tokenizer.pad_token = (
            tokenizer.unk_token
        )

    elif tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # GENERATE ORIGINAL PROMPT
    # --------------------------------------------------------

    prompt = GeneratePrompt(
        text
    )

    # --------------------------------------------------------
    # TOKENIZE
    #
    # Same style as the original program.
    # --------------------------------------------------------

    inputs = tokenizer(

        prompt,

        return_tensors="pt"

    ).to(DEVICE)

    input_tokens = (
        inputs["input_ids"]
        .shape[1]
    )

    print(
        f"Input tokens: "
        f"{input_tokens}",
        flush=True
    )

    # --------------------------------------------------------
    # CLEAR CUDA CACHE
    #
    # Same as original program.
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.time()

    print(
        "Starting generation...",
        flush=True
    )

    # --------------------------------------------------------
    # GENERATION ARGUMENTS
    #
    # Original program:
    #
    # do_sample=False
    # temperature=0.0
    # top_p=1.0
    #
    # In deterministic generation, temperature and top_p
    # do not affect the result. New Transformers versions
    # warn when they are passed with do_sample=False.
    #
    # Therefore they are intentionally omitted here.
    # --------------------------------------------------------

    generation_kwargs = {

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "do_sample":
            DO_SAMPLE,

        "pad_token_id":
            tokenizer.eos_token_id
    }

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            **generation_kwargs
        )

    # --------------------------------------------------------
    # SYNCHRONIZE CUDA
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # CALCULATE TIME
    # --------------------------------------------------------

    inference_time = (

        time.time()

        - start_time
    )

    print(
        "Generation finished.",
        flush=True
    )

    # --------------------------------------------------------
    # COUNT GENERATED TOKENS
    # --------------------------------------------------------

    generated_tokens = (

        outputs.shape[1]

        - input_tokens
    )

    # --------------------------------------------------------
    # DECODE
    #
    # Same approach as original program.
    # --------------------------------------------------------

    output_p = (
        tokenizer.batch_decode(

            outputs,

            skip_special_tokens=True
        )
    )

    # --------------------------------------------------------
    # EXTRACT RESPONSE
    #
    # Same behavior as original program.
    # --------------------------------------------------------

    output = None

    if output_p:

        output_text = (
            output_p[0]
        )

        split_text = (
            output_text.split(
                "Response:"
            )
        )

        if len(split_text) > 1:

            output = (

                split_text[1]

                .strip()
            )

    return (

        output,

        input_tokens,

        generated_tokens,

        inference_time
    )


# ============================================================
# EXTRACT CLEAN DICTIONARY
#
# Based on the original program.
# ============================================================

# ============================================================
# EXTRACT CLEAN DICTIONARY
#
# Extracts only the dictionary returned by the model.
# The original model prompt and inference behavior are kept.
# ============================================================

def ExtractCleanDict(text):

    if text is None:

        raise ValueError(
            "Model output is None."
        )

    text = str(text).strip()

    # --------------------------------------------------------
    # Remove response marker if present.
    # --------------------------------------------------------

    if "### Response:" in text:

        text = (
            text.split(
                "### Response:",
                1
            )[1]
            .strip()
        )

    # --------------------------------------------------------
    # Find the beginning of the dictionary.
    # --------------------------------------------------------

    start = text.find(
        "{'Flaws'"
    )

    if start == -1:

        raise ValueError(
            "No dictionary starting with "
            "{'Flaws' was found."
        )

    # --------------------------------------------------------
    # Start scanning from the dictionary.
    # --------------------------------------------------------

    candidate = text[start:]

    # --------------------------------------------------------
    # Find the matching closing brace.
    #
    # Braces inside strings are ignored.
    # This is important because the refactored Mamba code
    # itself can contain { and }.
    # --------------------------------------------------------

    depth = 0

    in_string = False

    string_quote = None

    escaped = False

    end_position = None

    for index, char in enumerate(candidate):

        # ----------------------------------------------------
        # Handle escaped characters.
        # ----------------------------------------------------

        if escaped:

            escaped = False

            continue

        if char == "\\" and in_string:

            escaped = True

            continue

        # ----------------------------------------------------
        # Handle strings.
        # ----------------------------------------------------

        if char in ("'", '"'):

            if not in_string:

                in_string = True

                string_quote = char

            elif char == string_quote:

                in_string = False

                string_quote = None

            continue

        # ----------------------------------------------------
        # Ignore braces inside strings.
        # ----------------------------------------------------

        if in_string:

            continue

        # ----------------------------------------------------
        # Track dictionary braces.
        # ----------------------------------------------------

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                end_position = index + 1

                break

    # --------------------------------------------------------
    # Make sure the dictionary was closed.
    # --------------------------------------------------------

    if end_position is None:

        raise ValueError(
            "Could not find the end of the "
            "model output dictionary."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT replace newlines here.
    #
    # The model already correctly returns \\n inside the
    # Refactored Versions string.
    # --------------------------------------------------------

    candidate = (
        candidate[
            :end_position
        ]
        .strip()
    )

    # --------------------------------------------------------
    # Parse dictionary.
    # --------------------------------------------------------

    try:

        result = ast.literal_eval(
            candidate
        )

    except Exception as e:

        raise ValueError(
            "Could not parse model output "
            f"as a Python dictionary: {e}"
        ) from e

    # --------------------------------------------------------
    # Validate dictionary.
    # --------------------------------------------------------

    if not isinstance(result, dict):

        raise ValueError(
            "Model output is not a dictionary."
        )

    if "Flaws" not in result:

        raise ValueError(
            "Model dictionary does not contain "
            "'Flaws'."
        )

    if "Refactored Versions" not in result:

        raise ValueError(
            "Model dictionary does not contain "
            "'Refactored Versions'."
        )

    return result

# ============================================================
# FORMAT OUTPUT
# ============================================================

# ============================================================
# FORMAT OUTPUT
#
# Produces the same output format as the original program.
# ============================================================

def FormatOutput(output_dict):

    result = ""

    # --------------------------------------------------------
    # FLAWS
    # --------------------------------------------------------

    result += "Flaws:\n"

    for entry in output_dict["Flaws"]:

        result += (
            f"   - {entry['Flaw']}: "
            f"{entry['Explanation']}\n"
        )

    # --------------------------------------------------------
    # REFACTORED VERSIONS
    # --------------------------------------------------------

    result += (
        "\n"
        "Refactored versions code:\n\n"
    )

    result += (
        output_dict["Refactored Versions"]
    )

    result += "\n\n"

    return result