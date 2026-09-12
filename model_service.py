import gc
import re
import ast
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

MAX_NEW_TOKENS = 32768

# Global model state
model = None
tokenizer = None

loaded_version = None
loaded_model_type = None


# ============================================================
# DEVICE / DTYPE
# ============================================================

def get_dtype():
    """
    Select the best dtype available on the current machine.
    """

    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


DTYPE = get_dtype()


# ============================================================
# HARDWARE INFORMATION
# ============================================================

def get_hardware_info():
    """
    Return basic information about available hardware.
    """

    if not torch.cuda.is_available():
        return (
            "CPU mode\n"
            f"Dtype: {DTYPE}"
        )

    gpu_count = torch.cuda.device_count()

    lines = [
        f"CUDA available: Yes",
        f"GPU count: {gpu_count}",
        f"Dtype: {DTYPE}",
    ]

    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        total_gb = total_memory / (1024 ** 3)

        lines.append(
            f"GPU {i}: {name} "
            f"({total_gb:.2f} GB)"
        )

    return "\n".join(lines)


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded():
    """
    Return True if a model and tokenizer are currently loaded.
    """

    return (
        model is not None
        and tokenizer is not None
    )


def get_loaded_model_info():
    """
    Return information about the currently loaded model.
    """

    if not is_model_loaded():
        return {
            "loaded": False,
            "version": None,
            "model_type": None,
        }

    return {
        "loaded": True,
        "version": loaded_version,
        "model_type": loaded_model_type,
    }


# ============================================================
# TOKENIZER CONFIGURATION
# ============================================================

def configure_tokenizer(tok):
    """
    Configure tokenizer to match the original program.
    """

    if tok.unk_token is not None:
        tok.pad_token = tok.unk_token
    elif tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    return tok


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

def get_checkpoint(version, model_type):
    """
    Return the correct checkpoint based on model version
    and model type.
    """

    if version not in [1, 2]:
        raise ValueError(
            "Model version must be 1 or 2."
        )

    if model_type not in [
        "LoRA Adapter",
        "Full Model",
    ]:
        raise ValueError(
            "Model type must be "
            "'LoRA Adapter' or 'Full Model'."
        )

    if model_type == "LoRA Adapter":

        if version == 1:
            return "HA-Siala/Mamba-v0.1"

        return "HA-Siala/Mamba-v0.2"

    # Full model
    if version == 1:
        return "HA-Siala/Mamba-full-v0.1"

    return "HA-Siala/Mamba-full-v0.2"


# ============================================================
# CLEAR CURRENT MODEL
# ============================================================

def clear_model():
    """
    Completely remove the currently loaded model.
    """

    global model
    global tokenizer
    global loaded_version
    global loaded_model_type

    model = None
    tokenizer = None

    loaded_version = None
    loaded_model_type = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(version, model_type):
    """
    Load the requested model.

    If the requested model is already loaded, it is reused.

    If a different version or model type is requested,
    the existing model is cleared first.
    """

    global model
    global tokenizer
    global loaded_version
    global loaded_model_type

    version = int(version)

    checkpoint = get_checkpoint(
        version,
        model_type,
    )

    # --------------------------------------------------------
    # REUSE EXISTING MODEL
    # --------------------------------------------------------

    if (
        model is not None
        and tokenizer is not None
        and loaded_version == version
        and loaded_model_type == model_type
    ):
        print(
            "Requested model is already loaded.",
            flush=True,
        )

        return {
            "status": "already_loaded",
            "version": version,
            "model_type": model_type,
            "checkpoint": checkpoint,
        }

    # --------------------------------------------------------
    # CLEAR DIFFERENT MODEL
    # --------------------------------------------------------

    if model is not None:
        print(
            "A different model is loaded. "
            "Clearing it first...",
            flush=True,
        )

        clear_model()

    # --------------------------------------------------------
    # LOAD TOKENIZER / MODEL
    # --------------------------------------------------------

    print(
        f"Loading model version {version} "
        f"({model_type})...",
        flush=True,
    )

    print(
        f"Checkpoint: {checkpoint}",
        flush=True,
    )

    print(
        f"Dtype: {DTYPE}",
        flush=True,
    )

    if model_type == "LoRA Adapter":

        # ----------------------------------------------------
        # LOAD BASE MODEL
        # ----------------------------------------------------

        print(
            "Loading base Mistral model...",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
        )

        tokenizer = configure_tokenizer(
            tokenizer
        )

        base_model = (
            AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                dtype=DTYPE,
                device_map="auto",
            )
        )

        # ----------------------------------------------------
        # LOAD LORA ADAPTER
        # ----------------------------------------------------

        print(
            "Loading LoRA adapter...",
            flush=True,
        )

        model = PeftModel.from_pretrained(
            base_model,
            checkpoint,
            dtype=DTYPE,
            is_trainable=False,
        )

    else:

        # ----------------------------------------------------
        # LOAD FULL MODEL
        # ----------------------------------------------------

        print(
            "Loading full Mamba model...",
            flush=True,
        )

        model = (
            AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=DTYPE,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
        )

        tokenizer = configure_tokenizer(
            tokenizer
        )

    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()

    loaded_version = version
    loaded_model_type = model_type

    print(
        "Model loaded successfully.",
        flush=True,
    )

    return {
        "status": "loaded",
        "version": version,
        "model_type": model_type,
        "checkpoint": checkpoint,
    }


# ============================================================
# ORIGINAL PROMPT
# ============================================================

def generate_prompt(content):
    """
    This is the same prompt used by the original program.
    """

    instruction = """Analyze the following Mamba code, detect any flaws, with a short explanation, and give one or more corrected versions of the code. Ensure that none of the refactored options repeat or reproduce the original code; only the improved version should be shown:"""

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task:

### Instruction:
{instruction}

### Code:
{content}
### Response:
"""

    return prompt


# ============================================================
# GET MODEL INPUT DEVICE
# ============================================================

def get_input_device():
    """
    Find a suitable device for input tensors when the model
    uses device_map='auto'.
    """

    if model is None:
        raise RuntimeError(
            "Model is not loaded."
        )

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


# ============================================================
# INFERENCE
# ============================================================

def generate_inference_output(text):
    """
    Run inference using the already-loaded model.

    Returns:

        output_text
        input_tokens
        generated_tokens
        inference_time
    """

    if not is_model_loaded():
        raise RuntimeError(
            "Model is not loaded. "
            "Call load_model() before inference."
        )

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    prompt = generate_prompt(text)

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_tokens = inputs["input_ids"].shape[1]

    input_device = get_input_device()

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    print(
        f"Input tokens: {input_tokens}",
        flush=True,
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --------------------------------------------------------
    # CUDA SYNCHRONIZATION
    #
    # This happens AFTER generation so timing represents
    # actual completed GPU work.
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    # --------------------------------------------------------
    # SEPARATE GENERATED TOKENS
    #
    # We decode only the newly generated part instead of
    # decoding the prompt again.
    # --------------------------------------------------------

    generated_tokens = (
        outputs.shape[1] - input_tokens
    )

    generated_tokens = max(
        0,
        generated_tokens,
    )

    generated_ids = outputs[
        :,
        input_tokens:
    ]

    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    if decoded:
        output_text = decoded[0].strip()
    else:
        output_text = ""

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    reached_limit = (
        generated_tokens >= MAX_NEW_TOKENS
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "GENERATION DIAGNOSTICS",
        flush=True,
    )

    print(
        f"Input tokens: {input_tokens}",
        flush=True,
    )

    print(
        f"Generated tokens: {generated_tokens}",
        flush=True,
    )

    print(
        f"Maximum generated tokens: "
        f"{MAX_NEW_TOKENS}",
        flush=True,
    )

    print(
        f"Reached token limit: "
        f"{reached_limit}",
        flush=True,
    )

    print(
        f"Output characters: "
        f"{len(output_text)}",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    return (
        output_text,
        input_tokens,
        generated_tokens,
        inference_time,
    )


# ============================================================
# EXTRACT MODEL DICTIONARY
# ============================================================

def extract_clean_dict(text):
    """
    Extract the dictionary returned by the model.

    Handles:
      - ### Response:
      - Python dictionaries
      - nested dictionaries/lists
      - braces inside strings
      - markdown fences
    """

    if text is None:
        raise ValueError(
            "Model output is None."
        )

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Model output is empty."
        )

    # --------------------------------------------------------
    # REMOVE RESPONSE HEADER
    # --------------------------------------------------------

    if "### Response:" in text:
        text = text.split(
            "### Response:",
            1,
        )[1].strip()

    # --------------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # --------------------------------------------------------

    text = re.sub(
        r"^```(?:python|json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # FIND FIRST DICT
    # --------------------------------------------------------

    start_positions = []

    python_start = text.find("{")
    json_start = text.find("{")

    if python_start >= 0:
        start_positions.append(
            python_start
        )

    if json_start >= 0:
        start_positions.append(
            json_start
        )

    if not start_positions:
        raise ValueError(
            "No dictionary found in model output."
        )

    start = min(start_positions)

    # --------------------------------------------------------
    # BRACE MATCHING
    # --------------------------------------------------------

    depth = 0
    in_string = False
    string_char = None
    escaped = False

    end = None

    for i in range(start, len(text)):

        char = text[i]

        # ----------------------------------------------------
        # STRING HANDLING
        # ----------------------------------------------------

        if in_string:

            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == string_char:
                in_string = False

            continue

        # ----------------------------------------------------
        # STRING START
        # ----------------------------------------------------

        if char in ("'", '"'):
            in_string = True
            string_char = char
            continue

        # ----------------------------------------------------
        # BRACES
        # ----------------------------------------------------

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                end = i + 1
                break

    # --------------------------------------------------------
    # TRUNCATED OUTPUT
    # --------------------------------------------------------

    if end is None:
        raise ValueError(
            "No complete dictionary found. "
            "The model output appears to be truncated."
        )

    candidate = text[
        start:end
    ].strip()

    # --------------------------------------------------------
    # PARSE PYTHON DICTIONARY
    # --------------------------------------------------------

    try:
        result = ast.literal_eval(
            candidate
        )

    except Exception as e:

        # Try JSON as a fallback.
        try:
            import json

            result = json.loads(
                candidate
            )

        except Exception:
            raise ValueError(
                "Could not parse model output "
                f"as a dictionary: {e}"
            )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not isinstance(result, dict):
        raise ValueError(
            "Parsed model output is not a dictionary."
        )

    if "Flaws" not in result:
        raise ValueError(
            "Model output does not contain "
            "'Flaws'."
        )

    if "Refactored Versions" not in result:
        raise ValueError(
            "Model output does not contain "
            "'Refactored Versions'."
        )

    return result


# ============================================================
# FORMAT OUTPUT
# ============================================================

def format_output(output_dict):
    """
    Format the parsed dictionary in the same general
    presentation as the original program.
    """

    result = ""

    result += "Flaws:\n"

    flaws = output_dict.get(
        "Flaws",
        [],
    )

    if isinstance(flaws, list):

        for entry in flaws:

            if isinstance(entry, dict):

                flaw = entry.get(
                    "Flaw",
                    "",
                )

                explanation = entry.get(
                    "Explanation",
                    "",
                )

                result += (
                    f"   - {flaw}: "
                    f"{explanation}\n"
                )

            else:

                result += (
                    f"   - {entry}\n"
                )

    else:

        result += (
            f"   - {flaws}\n"
        )

    result += (
        "\n"
        "Refactored versions code:\n"
        "\n"
    )

    refactored = output_dict.get(
        "Refactored Versions",
        "",
    )

    result += str(refactored)

    result += "\n\n"

    return result


# ============================================================
# END OF FILE
# ============================================================
