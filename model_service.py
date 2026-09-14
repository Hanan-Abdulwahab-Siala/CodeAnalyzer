import ast
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================================
# General configuration
# ============================================================================

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

MAX_NEW_TOKENS = 32768
DTYPE = torch.bfloat16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model state
# ============================================================================

_model = None
_tokenizer = None

_loaded_language = None
_loaded_task = None
_loaded_version = None
_loaded_model_type = None
_loaded_checkpoint = None


# ============================================================================
# Checkpoints
# ============================================================================

MAMBA_CHECKPOINTS = {
    ("LoRA Adapter", 1): "HA-Siala/Mamba-v0.1",
    ("LoRA Adapter", 2): "HA-Siala/Mamba-v0.2",
    ("Full Model", 1): "HA-Siala/Mamba-full-v0.1",
    ("Full Model", 2): "HA-Siala/Mamba-full-v0.2",
}


PYTHON_FLAW_CHECKPOINTS = {
    ("LoRA Adapter", 1): "HA-Siala/Detect-Flaws-v0.1",
    ("LoRA Adapter", 2): "HA-Siala/Detect-Flaws-v0.2",
    ("Full Model", 1): "HA-Siala/Detect-Flaws-full-v0.1",
    ("Full Model", 2): "HA-Siala/Detect-Flaws-full-v0.2",
}


PYTHON_REFACTOR_CHECKPOINTS = {
    ("LoRA Adapter", 1): "HA-Siala/RefactoringPy-v0.1",
    ("Full Model", 1): "HA-Siala/RefactoringPy-full-v0.1",
}


# ============================================================================
# Hardware information
# ============================================================================

def get_hardware_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        lines = []
        lines.append("CUDA available: Yes")
        lines.append(f"GPU count: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)

            lines.append(
                f"GPU {i}: {props.name} "
                f"({memory_gb:.2f} GB)"
            )

        lines.append(f"PyTorch CUDA version: {torch.version.cuda}")
        lines.append(f"PyTorch version: {torch.__version__}")

        return "\n".join(lines)

    return (
        "CUDA available: No\n"
        f"PyTorch version: {torch.__version__}\n"
        "The analyzer will use CPU."
    )


# ============================================================================
# Model status
# ============================================================================

def is_model_loaded():
    return _model is not None and _tokenizer is not None


def get_loaded_model_info():
    if not is_model_loaded():
        return {
            "language": None,
            "task": None,
            "version": None,
            "model_type": None,
            "checkpoint": None,
        }

    return {
        "language": _loaded_language,
        "task": _loaded_task,
        "version": _loaded_version,
        "model_type": _loaded_model_type,
        "checkpoint": _loaded_checkpoint,
    }


# ============================================================================
# Checkpoint selection
# ============================================================================

def get_checkpoint(language, task, version, model_type):

    if language == "Mamba":

        # Mamba has ONE task only.
        task = "Flaws + Refactoring"

        key = (model_type, version)

        if key not in MAMBA_CHECKPOINTS:
            raise ValueError(
                f"Invalid Mamba configuration: "
                f"model_type={model_type}, version={version}"
            )

        return MAMBA_CHECKPOINTS[key]

    if language == "Python":

        if task == "Flaw Detection":

            key = (model_type, version)

            if key not in PYTHON_FLAW_CHECKPOINTS:
                raise ValueError(
                    f"Invalid Python Flaw Detection configuration: "
                    f"model_type={model_type}, version={version}"
                )

            return PYTHON_FLAW_CHECKPOINTS[key]

        if task == "Refactoring":

            # There is only version 1 for Python refactoring.
            if version != 1:
                raise ValueError(
                    "Python Refactoring only supports Version 1."
                )

            key = (model_type, version)

            if key not in PYTHON_REFACTOR_CHECKPOINTS:
                raise ValueError(
                    f"Invalid Python Refactoring configuration: "
                    f"model_type={model_type}, version={version}"
                )

            return PYTHON_REFACTOR_CHECKPOINTS[key]

        raise ValueError(
            f"Unknown Python task: {task}"
        )

    raise ValueError(
        f"Unknown language: {language}"
    )


# ============================================================================
# Model loading
# ============================================================================

def load_model(language, version, model_type, task=None):

    global _model
    global _tokenizer

    global _loaded_language
    global _loaded_task
    global _loaded_version
    global _loaded_model_type
    global _loaded_checkpoint

    version = int(version)

    # ------------------------------------------------------------------------
    # Mamba has only one task.
    # ------------------------------------------------------------------------

    if language == "Mamba":
        task = "Flaws + Refactoring"

    # ------------------------------------------------------------------------
    # Python refactoring has only version 1.
    # ------------------------------------------------------------------------

    if language == "Python" and task == "Refactoring":
        version = 1

    checkpoint = get_checkpoint(
        language=language,
        task=task,
        version=version,
        model_type=model_type,
    )

    # ------------------------------------------------------------------------
    # Reuse currently loaded model if configuration is identical.
    # ------------------------------------------------------------------------

    if (
        is_model_loaded()
        and _loaded_language == language
        and _loaded_task == task
        and _loaded_version == version
        and _loaded_model_type == model_type
        and _loaded_checkpoint == checkpoint
    ):
        print("=" * 70)
        print("MODEL ALREADY LOADED")
        print("=" * 70)
        print(
            f"Language    : {_loaded_language}"
        )
        print(
            f"Task        : {_loaded_task}"
        )
        print(
            f"Version     : {_loaded_version}"
        )
        print(
            f"Model type  : {_loaded_model_type}"
        )
        print(
            f"Checkpoint  : {_loaded_checkpoint}"
        )
        print("=" * 70)

        return "already_loaded"

    # ------------------------------------------------------------------------
    # If another model is loaded, release it.
    # ------------------------------------------------------------------------

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Print loading information.
    # ------------------------------------------------------------------------

    print("=" * 70)
    print("Loading model")
    print("=" * 70)
    print(f"Language    : {language}")
    print(f"Task        : {task}")
    print(f"Version     : {version}")
    print(f"Model type  : {model_type}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Hardware    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dtype       : {DTYPE}")
    print("=" * 70)

    # =========================================================================
    # LoRA
    # =========================================================================

    if model_type == "LoRA Adapter":

        print(f"Loading base model: {BASE_MODEL}")

        _tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
        )

        _tokenizer.pad_token = _tokenizer.unk_token
        _tokenizer.padding_side = "left"

        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=DTYPE,
            device_map="auto",
        )

        print(f"Loading LoRA adapter: {checkpoint}")

        _model = PeftModel.from_pretrained(
            _model,
            checkpoint,
            torch_dtype=DTYPE,
            is_trainable=False,
        )

    # =========================================================================
    # Full model
    # =========================================================================

    elif model_type == "Full Model":

        print(f"Loading full model: {checkpoint}")

        _model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=DTYPE,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
        )

        _tokenizer.pad_token = _tokenizer.unk_token
        _tokenizer.padding_side = "left"

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    _model.eval()

    # ------------------------------------------------------------------------
    # Save current configuration.
    # ------------------------------------------------------------------------

    _loaded_language = language
    _loaded_task = task
    _loaded_version = version
    _loaded_model_type = model_type
    _loaded_checkpoint = checkpoint

    print("=" * 70)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 70)
    print(
        f"Loaded: {language} | "
        f"Task: {task} | "
        f"Version: {version} | "
        f"Type: {model_type} | "
        f"Checkpoint: {checkpoint}"
    )
    print("=" * 70)

    return "loaded"


# ============================================================================
# PROMPTS
# ============================================================================

def generate_mamba_prompt(content):

    # KEEPING THE ORIGINAL MAMBA PROMPT
    instruction = (
        "Analyze the following Mamba code, detect any flaws, "
        "with a short explanation, and give one or more corrected "
        "versions of the code. Ensure that none of the refactored "
        "options repeat or reproduce the original code; only the "
        "improved version should be shown:"
    )

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task:

### Instruction:
{instruction}

### Code:
{content}
### Response:
"""

    return prompt


def generate_python_prompt(content):

    # KEEPING YOUR ORIGINAL PYTHON PROMPT EXACTLY
    instruction = """Analyze the following Python code, identify any flaws, with a short explanation, and give one or more corrected versions of the code, including a brief explanation of each correction approach:"""

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task: 

### Instruction: 
{instruction} 

### Code: 
{content} 
### Response: 
"""

    return prompt


def generate_prompt(language, task, content):

    if language == "Mamba":
        return generate_mamba_prompt(content)

    if language == "Python":
        return generate_python_prompt(content)

    raise ValueError(
        f"Unknown language: {language}"
    )


# ============================================================================
# INFERENCE
#
# IMPORTANT:
# This mechanism is intentionally the same mechanism used by your
# working Mamba implementation.
# ============================================================================

def generate_inference_output(text):

    if not is_model_loaded():
        raise RuntimeError(
            "No model is loaded."
        )

    tokenizer = _tokenizer
    model = _model

    prompt = generate_prompt(
        language=_loaded_language,
        task=_loaded_task,
        content=text,
    )

    # ------------------------------------------------------------------------
    # Same tokenizer configuration as the original programs.
    # ------------------------------------------------------------------------

    tokenizer.pad_token = tokenizer.unk_token

    # ------------------------------------------------------------------------
    # SAME inference mechanism
    # ------------------------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    input_tokens = inputs["input_ids"].shape[1]

    # Same cache clearing as the working program.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Same timing mechanism.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            # SAME VALUE AS YOUR WORKING PROGRAMS
            max_new_tokens=32768,

            # SAME SETTINGS AS YOUR ORIGINAL PYTHON PROGRAM
            # and the Mamba mechanism.
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            top_p=1.0,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    inference_time = end_time - start_time

    # ------------------------------------------------------------------------
    # Same decoding mechanism.
    # ------------------------------------------------------------------------

    output_p = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    # ------------------------------------------------------------------------
    # Same generated-token concept.
    # ------------------------------------------------------------------------

    if output_p:

        output_text = output_p[0]

        generated_tokens = (
            outputs.shape[1] - input_tokens
        )

        # Same Response extraction mechanism.
        split_text = output_text.split("Response:")

        if len(split_text) > 1:
            output_text = split_text[-1].strip()
        else:
            output_text = output_text.strip()

        return (
            output_text,
            input_tokens,
            generated_tokens,
            inference_time,
        )

    return (
        None,
        input_tokens,
        0,
        inference_time,
    )


# ============================================================================
# Dictionary extraction
# ============================================================================

def extract_clean_dict(text):

    if text is None:
        raise ValueError("Model returned no output.")

    text = str(text).strip()

    # Remove prompt if the model returned it.
    if "### Response:" in text:
        text = text.split("### Response:")[-1].strip()

    # ------------------------------------------------------------------------
    # Find dictionary starting at {'Flaws'
    # ------------------------------------------------------------------------

    start = text.find("{'Flaws'")

    if start == -1:
        start = text.find('{"Flaws"')

    if start == -1:
        raise ValueError(
            "No dictionary beginning with 'Flaws' was found."
        )

    # ------------------------------------------------------------------------
    # Brace-aware extraction.
    # ------------------------------------------------------------------------

    opening = text[start]

    if opening != "{":
        raise ValueError(
            "Invalid dictionary start."
        )

    depth = 0
    in_string = False
    string_char = None
    escaped = False
    end = None

    for i in range(start, len(text)):

        char = text[i]

        if escaped:
            escaped = False
            continue

        if in_string:

            if char == "\\":
                escaped = True
                continue

            if char == string_char:
                in_string = False
                string_char = None

            continue

        if char in ("'", '"'):
            in_string = True
            string_char = char
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(
            "Could not find the end of the result dictionary."
        )

    candidate = text[start:end]

    # ------------------------------------------------------------------------
    # First try literal_eval directly.
    # ------------------------------------------------------------------------

    try:
        result = ast.literal_eval(candidate)

        if not isinstance(result, dict):
            raise ValueError(
                "Parsed result is not a dictionary."
            )

        return result

    except Exception:
        pass

    # ------------------------------------------------------------------------
    # Compatibility with your original Python parser:
    # escape actual newlines inside the generated dictionary.
    # ------------------------------------------------------------------------

    candidate = candidate.replace("\n", "\\n")

    try:
        result = ast.literal_eval(candidate)

        if not isinstance(result, dict):
            raise ValueError(
                "Parsed result is not a dictionary."
            )

        return result

    except Exception as e:
        raise ValueError(
            f"Could not parse model dictionary: {e}"
        )


# ============================================================================
# Output formatting
# ============================================================================

def format_output(output_dict):

    language = _loaded_language
    task = _loaded_task

    if language == "Mamba":

        if "Flaws" not in output_dict:
            raise ValueError(
                "Mamba output does not contain 'Flaws'."
            )

        if "Refactored Versions" not in output_dict:
            raise ValueError(
                "Mamba output does not contain "
                "'Refactored Versions'."
            )

        result = "Flaws:\n"

        for entry in output_dict["Flaws"]:

            if isinstance(entry, dict):

                flaw = entry.get("Flaw", "")
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

        result += (
            "\n"
            "Refactored versions code:\n\n"
        )

        result += str(
            output_dict["Refactored Versions"]
        )

        result += "\n\n"

        return result

    # =========================================================================
    # Python - Flaw Detection
    # =========================================================================

    if language == "Python" and task == "Flaw Detection":

        if "Flaws" not in output_dict:
            raise ValueError(
                "Python Flaw Detection output does not "
                "contain 'Flaws'."
            )

        result = "Flaws:\n"

        for entry in output_dict["Flaws"]:

            if isinstance(entry, dict):

                flaw = entry.get("Flaw", "")
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

        result += (
            "\n"
            "Corrected code recommendation:\n\n"
        )

        corrections = output_dict.get(
            "Corrections"
        )

        if isinstance(corrections, list) and corrections:

            for item in corrections:

                if isinstance(item, dict):

                    explanation = item.get(
                        "Explanation",
                        "",
                    )

                    correction = item.get(
                        "Correction",
                        "",
                    )

                    # Same behavior as your original:
                    # explanation is not printed here.
                    result += (
                        str(correction)
                        + "\n\n"
                    )

                else:
                    result += (
                        str(item)
                        + "\n\n"
                    )

        elif "Correction" in output_dict:

            result += str(
                output_dict["Correction"]
            )

        return result

    # =========================================================================
    # Python - Refactoring
    # =========================================================================

    if language == "Python" and task == "Refactoring":

        if "Flaws" not in output_dict:
            raise ValueError(
                "Python Refactoring output does not "
                "contain 'Flaws'."
            )

        if "Refactored Versions" not in output_dict:
            raise ValueError(
                "Python Refactoring output does not "
                "contain 'Refactored Versions'."
            )

        result = "Flaws:\n"

        for entry in output_dict["Flaws"]:

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

        result += (
            "\n"
            "Refactored versions code:\n\n"
        )

        result += str(
            output_dict["Refactored Versions"]
        )

        result += "\n\n"

        return result

    raise ValueError(
        f"Unsupported configuration: "
        f"{language} / {task}"
    )
