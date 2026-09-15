# -----------------------------------------------------------------------------
# Unified model service
#
# Inference behavior is kept as close as possible to the original standalone
# Mistral and DeepSeek programs.
# -----------------------------------------------------------------------------

import ast
import re
import threading

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# -----------------------------------------------------------------------------
# Device / dtype
# -----------------------------------------------------------------------------

# Match the original programs.
DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# Match the original programs:
# bfloat16 on CUDA.
# float32 only when CUDA is unavailable.
TORCH_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available()
    else torch.float32
)


# -----------------------------------------------------------------------------
# Base models
# -----------------------------------------------------------------------------

MISTRAL_BASE = (
    "mistralai/Mistral-7B-v0.3"
)

DEEPSEEK_BASE = (
    "deepseek-ai/deepseek-coder-6.7b-base"
)


# -----------------------------------------------------------------------------
# Mistral checkpoints
# -----------------------------------------------------------------------------

MISTRAL_CHECKPOINTS = {

    "Mamba": {

        "LoRA Adapter": {
            1: "HA-Siala/Mamba-v0.1",
            2: "HA-Siala/Mamba-v0.2",
        },

        "Full Model": {
            1: "HA-Siala/Mamba-full-v0.1",
            2: "HA-Siala/Mamba-full-v0.2",
        },
    },

    "Python": {

        "LoRA Adapter": {

            "Flaw Detection": {
                1: "HA-Siala/Detect-Flaws-v0.1",
                2: "HA-Siala/Detect-Flaws-v0.2",
            },

            "Refactoring": {
                1: "HA-Siala/RefactoringPy-v0.1",
            },
        },

        "Full Model": {

            "Flaw Detection": {
                1: "HA-Siala/Detect-Flaws-full-v0.1",
                2: "HA-Siala/Detect-Flaws-full-v0.2",
            },

            "Refactoring": {
                1: "HA-Siala/RefactoringPy-full-v0.1",
            },
        },
    },
}


# -----------------------------------------------------------------------------
# DeepSeek checkpoints
# -----------------------------------------------------------------------------

DEEPSEEK_CHECKPOINTS = {

    "Python": {

        "LoRA Adapter":
            "HA-Siala/RefactoringPy-DeepSeek-v0.1",

        "Full Model":
            "HA-Siala/RefactoringPy-DeepSeek-full-v0.1",
    },

    "Mamba": {

        "LoRA Adapter":
            "HA-Siala/Mamba-DeepSeek-v0.1",

        "Full Model":
            "HA-Siala/Mamba-DeepSeek-full-v0.1",
    },
}


# -----------------------------------------------------------------------------
# Global model state
# -----------------------------------------------------------------------------

MODEL = None
TOKENIZER = None

LOADED_MODEL_FAMILY = None
LOADED_LANGUAGE = None
LOADED_TASK = None
LOADED_VERSION = None
LOADED_MODEL_TYPE = None
LOADED_CHECKPOINT = None

MODEL_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Prompt generation
# -----------------------------------------------------------------------------

def generate_mistral_prompt(
    content,
    language,
    task,
):
    """
    Exact Mistral prompt structure from the original programs.
    """

    if language == "Mamba":

        instruction = (
            "Analyze the following Mamba code, detect any flaws, "
            "with a short explanation, and give one or more corrected "
            "versions of the code. Ensure that none of the refactored "
            "options repeat or reproduce the original code; only the "
            "improved version should be shown:"
        )

    elif language == "Python":

        instruction = (
            "Analyze the following Python code, identify any flaws, "
            "with a short explanation, and give one or more corrected "
            "versions of the code, including a brief explanation of "
            "each correction approach:"
        )

    else:

        raise ValueError(
            f"Unsupported Mistral language: {language}"
        )

    # Exact wrapper from the original Mistral programs.
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task:

### Instruction:
{instruction}

### Code:
{content}
### Response:
"""

    return prompt


def generate_deepseek_prompt(
    content,
    language,
):
    """
    Exact DeepSeek prompt structure from the original program.
    """

    if language == "Python":

        instruction = (
            "Analyze the following Python code, detect any flaws, "
            "with a short explanation, and give one or more corrected "
            "versions of the code. Ensure that none of the refactored "
            "options repeat or reproduce the original code; only the "
            "improved version should be shown."
        )

    elif language == "Mamba":

        instruction = (
            "Analyze the following Mamba code, detect any flaws, "
            "with a short explanation, and give one or more corrected "
            "versions of the code. Ensure that none of the refactored "
            "options repeat or reproduce the original code; only the "
            "improved version should be shown."
        )

    else:

        raise ValueError(
            f"Unsupported DeepSeek language: {language}"
        )

    # Exact wrapper from the original DeepSeek program.
    prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model.
### Instruction:
{instruction}

### Input:
{content}

### Response:
"""

    return prompt


# -----------------------------------------------------------------------------
# Public prompt function used by app.py
# -----------------------------------------------------------------------------

def generate_prompt(
    language,
    task,
    content,
    model_family,
):
    """
    Generate the prompt corresponding to the selected model.

    This function intentionally uses the supplied configuration instead of
    requiring the model to be loaded first.
    """

    if model_family == "Mistral":

        return generate_mistral_prompt(
            content=content,
            language=language,
            task=task,
        )

    if model_family == "DeepSeek":

        return generate_deepseek_prompt(
            content=content,
            language=language,
        )

    raise ValueError(
        f"Unsupported model family: {model_family}"
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _load_mistral(
    language,
    task,
    version,
    model_type,
):
    """
    Load Mistral using the same mechanism as the original programs.
    """

    if language not in MISTRAL_CHECKPOINTS:

        raise ValueError(
            f"Unsupported Mistral language: {language}"
        )

    if model_type not in MISTRAL_CHECKPOINTS[language]:

        raise ValueError(
            f"Unsupported Mistral model type: {model_type}"
        )

    checkpoint_group = (
        MISTRAL_CHECKPOINTS[language][model_type]
    )

    # ---------------------------------------------------------
    # Mamba
    # ---------------------------------------------------------

    if language == "Mamba":

        checkpoint = checkpoint_group.get(version)

    # ---------------------------------------------------------
    # Python
    # ---------------------------------------------------------

    else:

        if task not in checkpoint_group:

            raise ValueError(
                f"Unsupported Mistral Python task: {task}"
            )

        checkpoint = (
            checkpoint_group[task].get(version)
        )

    if checkpoint is None:

        raise ValueError(
            "No Mistral checkpoint for "
            f"language={language}, "
            f"task={task}, "
            f"version={version}, "
            f"model_type={model_type}"
        )

    # ---------------------------------------------------------
    # LoRA Adapter
    # ---------------------------------------------------------

    if model_type == "LoRA Adapter":

        # Match original Mistral LoRA tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            MISTRAL_BASE,
            use_fast=True,
        )

        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"

        # Match original Mistral base model.
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_BASE,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
        )

        # Match original Mistral adapter loading.
        model = PeftModel.from_pretrained(
            model,
            checkpoint,
            torch_dtype=TORCH_DTYPE,
            is_trainable=False,
        )

    # ---------------------------------------------------------
    # Full Model
    # ---------------------------------------------------------

    else:

        # Match original Mistral full-model loading.
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
        )

        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"

    model.eval()

    return (
        model,
        tokenizer,
        checkpoint,
    )


def _load_deepseek(
    language,
    model_type,
):
    """
    Load DeepSeek using the same mechanism as the original program.
    """

    if language not in DEEPSEEK_CHECKPOINTS:

        raise ValueError(
            f"Unsupported DeepSeek language: {language}"
        )

    if model_type not in DEEPSEEK_CHECKPOINTS[language]:

        raise ValueError(
            f"Unsupported DeepSeek model type: {model_type}"
        )

    checkpoint = (
        DEEPSEEK_CHECKPOINTS[language][model_type]
    )

    # ---------------------------------------------------------
    # LoRA Adapter
    # ---------------------------------------------------------

    if model_type == "LoRA Adapter":

        # Match original DeepSeek LoRA tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        # Match original DeepSeek base model.
        base_model = (
            AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_BASE,
                torch_dtype=TORCH_DTYPE,
                device_map="auto",
                trust_remote_code=True,
            )
        )

        # Match original DeepSeek PEFT loading.
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint,
        )

    # ---------------------------------------------------------
    # Full Model
    # ---------------------------------------------------------

    else:

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
        )

        if tokenizer.pad_token is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

    model.eval()

    return (
        model,
        tokenizer,
        checkpoint,
    )


# -----------------------------------------------------------------------------
# Public model loading function used by app.py
# -----------------------------------------------------------------------------

def load_model(
    language,
    task,
    version,
    model_type,
    model_family,
):
    """
    Load exactly the configuration selected in app.py.
    """

    global MODEL
    global TOKENIZER

    global LOADED_MODEL_FAMILY
    global LOADED_LANGUAGE
    global LOADED_TASK
    global LOADED_VERSION
    global LOADED_MODEL_TYPE
    global LOADED_CHECKPOINT

    with MODEL_LOCK:

        # Unload previous model first.
        _unload_model_internal()

        # -----------------------------------------------------
        # Mistral
        # -----------------------------------------------------

        if model_family == "Mistral":

            if language == "Mamba":

                task = "Flaws + Refactoring"

            elif (
                language == "Python"
                and task == "Refactoring"
            ):

                version = 1

            (
                model,
                tokenizer,
                checkpoint,
            ) = _load_mistral(
                language=language,
                task=task,
                version=version,
                model_type=model_type,
            )

        # -----------------------------------------------------
        # DeepSeek
        # -----------------------------------------------------

        elif model_family == "DeepSeek":

            # DeepSeek always version 1.
            version = 1

            # DeepSeek always Flaws + Refactoring.
            task = "Flaws + Refactoring"

            (
                model,
                tokenizer,
                checkpoint,
            ) = _load_deepseek(
                language=language,
                model_type=model_type,
            )

        else:

            raise ValueError(
                f"Unsupported model family: {model_family}"
            )

        MODEL = model
        TOKENIZER = tokenizer

        LOADED_MODEL_FAMILY = model_family
        LOADED_LANGUAGE = language
        LOADED_TASK = task
        LOADED_VERSION = version
        LOADED_MODEL_TYPE = model_type
        LOADED_CHECKPOINT = checkpoint

        return {
            "model_family":
            LOADED_MODEL_FAMILY,

            "language":
            LOADED_LANGUAGE,

            "task":
            LOADED_TASK,

            "version":
            LOADED_VERSION,

            "model_type":
            LOADED_MODEL_TYPE,

            "checkpoint":
            LOADED_CHECKPOINT,
        }


# -----------------------------------------------------------------------------
# Internal unload
# -----------------------------------------------------------------------------

def _unload_model_internal():

    global MODEL
    global TOKENIZER

    global LOADED_MODEL_FAMILY
    global LOADED_LANGUAGE
    global LOADED_TASK
    global LOADED_VERSION
    global LOADED_MODEL_TYPE
    global LOADED_CHECKPOINT

    MODEL = None
    TOKENIZER = None

    LOADED_MODEL_FAMILY = None
    LOADED_LANGUAGE = None
    LOADED_TASK = None
    LOADED_VERSION = None
    LOADED_MODEL_TYPE = None
    LOADED_CHECKPOINT = None

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Public unload
# -----------------------------------------------------------------------------

def unload_model():

    with MODEL_LOCK:

        _unload_model_internal()


# -----------------------------------------------------------------------------
# Model state
# -----------------------------------------------------------------------------

def is_model_loaded():

    return (
        MODEL is not None
        and TOKENIZER is not None
    )


def get_loaded_model_info():

    if not is_model_loaded():

        return None

    return {
        "model_family":
        LOADED_MODEL_FAMILY,

        "language":
        LOADED_LANGUAGE,

        "task":
        LOADED_TASK,

        "version":
        LOADED_VERSION,

        "model_type":
        LOADED_MODEL_TYPE,

        "checkpoint":
        LOADED_CHECKPOINT,
    }


# -----------------------------------------------------------------------------
# Mistral inference
# -----------------------------------------------------------------------------

def _generate_mistral_output(
    prompt,
):
    """
    Original Mistral inference mechanism.

    Important:
      - max_new_tokens = 32768
      - do_sample = False
      - top_p = 1.0
      - complete sequence is decoded
      - split at "Response:"
    """

    # Match original Mistral inference.
    TOKENIZER.pad_token = TOKENIZER.unk_token

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    input_tokens = (
        inputs["input_ids"].shape[1]
    )

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    with torch.inference_mode():

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=32768,
            temperature=0.0,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
            top_p=1.0,
        )

        # IMPORTANT:
        # Decode the complete generated sequence.
        output_p = TOKENIZER.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

    if output_p:

        output_text = output_p[0]

        split_text = output_text.split(
            "Response:"
        )

        if len(split_text) > 1:

            return (
                split_text[1].strip(),
                input_tokens,
            )

        return (
            None,
            input_tokens,
        )

    return (
        None,
        0,
    )


# -----------------------------------------------------------------------------
# DeepSeek inference
# -----------------------------------------------------------------------------

def _generate_deepseek_output(
    prompt,
):
    """
    Original DeepSeek inference mechanism, with max_new_tokens=32768
    as requested.

    Important:
      - max_new_tokens = 32768
      - do_sample = False
      - num_beams = 1
      - top_p = 1.0
      - repetition_penalty = 1.1
      - only generated tokens are decoded
    """

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    input_tokens = (
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=32768,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )

    # IMPORTANT:
    # Decode only generated tokens.
    response = TOKENIZER.decode(
        outputs[0][
            inputs["input_ids"].shape[1]:
        ],
        skip_special_tokens=True,
    )

    return (
        response,
        input_tokens,
    )


# -----------------------------------------------------------------------------
# Public inference function used by app.py
# -----------------------------------------------------------------------------

def generate_inference_output(
    prompt,
):
    """
    Generate the raw response using the currently loaded model.

    app.py passes the already-created prompt here.
    """

    if not is_model_loaded():

        raise RuntimeError(
            "No model is loaded."
        )

    if LOADED_MODEL_FAMILY == "Mistral":

        raw_output, _ = (
            _generate_mistral_output(
                prompt
            )
        )

        return raw_output

    if LOADED_MODEL_FAMILY == "DeepSeek":

        raw_output, _ = (
            _generate_deepseek_output(
                prompt
            )
        )

        return raw_output

    raise RuntimeError(
        f"Unsupported loaded model family: "
        f"{LOADED_MODEL_FAMILY}"
    )


# -----------------------------------------------------------------------------
# Mistral dictionary extraction
# -----------------------------------------------------------------------------

def _extract_clean_dict(text):

    text = str(text)

    if "### Response:" in text:

        text = (
            text
            .split("### Response:")[-1]
            .strip()
        )

    else:

        text = text.strip()

    match = re.search(
        r"(\{'Flaws'[\s\S]*\})",
        text,
    )

    if not match:

        raise ValueError(
            "No dict found"
        )

    candidate = match.group(1)

    candidate = candidate.replace(
        "\n",
        "\\n",
    )

    return ast.literal_eval(
        candidate
    )


# -----------------------------------------------------------------------------
# DeepSeek cleaning
# -----------------------------------------------------------------------------

def _clean_deepseek_text(text):

    if not text:

        return ""

    text = str(text)

    text = text.split("###")[0]

    return text


# -----------------------------------------------------------------------------
# Mistral output formatting
# -----------------------------------------------------------------------------

def _format_mistral_output(
    raw_output,
):
    """
    Preserve the original Mistral output handling.
    """

    if raw_output is None:

        return "INVALID OUTPUT"

    output_dict = _extract_clean_dict(
        raw_output
    )

    result = "Flaws:\n"

    # ---------------------------------------------------------
    # Flaws
    # ---------------------------------------------------------

    for entry in output_dict["Flaws"]:

        result += (
            f"   - {entry['Flaw']}: "
            f"{entry['Explanation']}\n"
        )

    # ---------------------------------------------------------
    # Python Flaw Detection
    # ---------------------------------------------------------

    if (
        LOADED_LANGUAGE == "Python"
        and LOADED_TASK == "Flaw Detection"
    ):

        result += (
            "\nCorrected code recommendation:\n\n"
        )

        corrections = output_dict.get(
            "Corrections"
        )

        if (
            isinstance(corrections, list)
            and corrections
        ):

            for item in corrections:

                # Same behavior as the original:
                # explanation is not written.
                code = item.get(
                    "Correction",
                    "",
                )

                result += (
                    code
                    + "\n\n"
                )

        elif "Correction" in output_dict:

            result += str(
                output_dict["Correction"]
            )

    # ---------------------------------------------------------
    # Python Refactoring
    # ---------------------------------------------------------

    elif (
        LOADED_LANGUAGE == "Python"
        and LOADED_TASK == "Refactoring"
    ):

        result += (
            "\nRefactored versions code:\n\n"
        )

        # IMPORTANT:
        # Directly use the original dictionary field.
        if "Refactored Versions" in output_dict:

            result += str(
                output_dict[
                    "Refactored Versions"
                ]
            )

        else:

            result += "INVALID OUTPUT"

    # ---------------------------------------------------------
    # Mamba
    # ---------------------------------------------------------

    elif LOADED_LANGUAGE == "Mamba":

        result += (
            "\nRefactored versions code:\n\n"
        )

        # IMPORTANT:
        # Directly use the original dictionary field.
        if "Refactored Versions" in output_dict:

            result += str(
                output_dict[
                    "Refactored Versions"
                ]
            )

        else:

            result += "INVALID OUTPUT"

    else:

        result += (
            "\nRefactored versions code:\n\n"
        )

        if "Refactored Versions" in output_dict:

            result += str(
                output_dict[
                    "Refactored Versions"
                ]
            )

        else:

            result += "INVALID OUTPUT"

    result += "\n\n"

    return result


# -----------------------------------------------------------------------------
# DeepSeek output formatting
# -----------------------------------------------------------------------------

def _format_deepseek_output(
    raw_output,
):
    """
    Preserve the original DeepSeek PrintResult behavior.
    """

    result = "Flaws:\n"

    if not raw_output:

        result += (
            "INVALID OUTPUT\n\n"
        )

        return result

    try:

        text = _clean_deepseek_text(
            str(raw_output)
        )

        flaws = re.findall(
            r'"Flaw"\s*:\s*"([^"]+)"',
            text,
        )

        explanations = re.findall(
            r'"Explanation"\s*:\s*"([^"]+)"',
            text,
        )

        for i in range(
            min(
                len(flaws),
                len(explanations),
            )
        ):

            # Preserve original DeepSeek spacing.
            result += (
                f"   - {flaws[i]} : "
                f"{explanations[i]}\n"
            )

        result += (
            "\nRefactored versions code:\n\n"
        )

        refactored = re.search(
            r'"Refactored Versions"\s*:\s*"([\s\S]+)"',
            text,
        )

        if refactored:

            cleaned_code = (
                refactored.group(1)
            )

            cleaned_code = (
                cleaned_code.replace(
                    '\\"',
                    '"',
                )
            )

            cleaned_code = (
                cleaned_code.replace(
                    "\\n",
                    "\n",
                )
            )

            result += cleaned_code

        else:

            result += "INVALID OUTPUT"

        result += "\n\n"

    except Exception:

        result += (
            "INVALID OUTPUT\n\n"
        )

    return result


# -----------------------------------------------------------------------------
# Public formatting function used by app.py
# -----------------------------------------------------------------------------

def format_output(
    raw_output,
):
    """
    Format the raw output according to the loaded model.
    """

    if not is_model_loaded():

        raise RuntimeError(
            "No model is loaded."
        )

    if LOADED_MODEL_FAMILY == "Mistral":

        return _format_mistral_output(
            raw_output
        )

    if LOADED_MODEL_FAMILY == "DeepSeek":

        return _format_deepseek_output(
            raw_output
        )

    raise RuntimeError(
        f"Unsupported model family: "
        f"{LOADED_MODEL_FAMILY}"
    )


# -----------------------------------------------------------------------------
# Compatibility aliases
# -----------------------------------------------------------------------------

load_selected_model = load_model

