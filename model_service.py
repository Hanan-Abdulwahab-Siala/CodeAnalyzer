import ast
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    PeftModel
)

# ============================================================
# GPU CONFIGURATION
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = "cuda:0"
    GPU_NAME = torch.cuda.get_device_name(0)
    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"
    DTYPE = torch.float32

# ============================================================
# DISPLAY HARDWARE
# ============================================================

print("=" * 60, flush=True)
print("MAMBA CODE ANALYZER - HARDWARE", flush=True)
print("=" * 60, flush=True)
print(f"CUDA available : {CUDA_AVAILABLE}", flush=True)
print(f"Device         : {DEVICE}", flush=True)
print(f"GPU            : {GPU_NAME}", flush=True)
print(f"PyTorch        : {torch.__version__}", flush=True)
print(f"CUDA version   : {torch.version.cuda}", flush=True)
print(f"Dtype          : {DTYPE}", flush=True)
print("=" * 60, flush=True)

# ============================================================
# GENERATION CONFIGURATION
# ============================================================

# Safer than 32768 for public users with different GPUs.

MAX_NEW_TOKENS = 4096
DO_SAMPLE = False

# ============================================================
# MODEL CHECKPOINTS
# ============================================================

FULL_MODEL_V1 = "HA-Siala/Mamba-full-v0.1"
FULL_MODEL_V2 = "HA-Siala/Mamba-full-v0.2"
LORA_MODEL_V1 = "HA-Siala/Mamba-v0.1"
LORA_MODEL_V2 = "HA-Siala/Mamba-v0.2"
BASE_MODEL = "mistralai/Mistral-7B-v0.3"

# ============================================================
# GLOBAL MODEL STATE
# ============================================================

model = None
tokenizer = None
loaded_version = None
loaded_model_type = None

# ============================================================
# LOAD CAUSAL LM
# ============================================================

def load_causal_lm(model_path, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE, **kwargs)

# ============================================================
# CONFIGURE TOKENIZER
# ============================================================

def configure_tokenizer(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

# ============================================================
# CLEAR MODEL
# ============================================================

def clear_model():
    global model
    global tokenizer
    global loaded_version
    global loaded_model_type
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    loaded_version = None
    loaded_model_type = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(version=2, model_type="LoRA Adapter"):
    global model
    global tokenizer
    global loaded_version
    global loaded_model_type

    # --------------------------------------------------------
    # REUSE MODEL
    # --------------------------------------------------------

    if (model is not None and tokenizer is not None and loaded_version == version and loaded_model_type == model_type):
        print("Requested model is already loaded. Reusing it.", flush=True)
        return model

    # --------------------------------------------------------
    # CLEAR DIFFERENT MODEL
    # --------------------------------------------------------

    if model is not None:
        print("Different model requested. Clearing current model.", flush=True)
        clear_model()

    # --------------------------------------------------------
    # VALIDATE VERSION
    # --------------------------------------------------------

    if version not in (1, 2):
        raise ValueError(f"Unsupported model version: {version}")

    # --------------------------------------------------------
    # VALIDATE TYPE
    # --------------------------------------------------------

    if model_type not in ("LoRA Adapter", "Full Model"):
        raise ValueError("Unsupported model type.")

    print(f"Loading model: version={version}, type={model_type}", flush=True)

    # ========================================================
    # LORA
    # ========================================================

    if model_type == "LoRA Adapter":
        if version == 1:
            adapter_path = LORA_MODEL_V1
        else:
            adapter_path = LORA_MODEL_V2
        print(f"Base model: {BASE_MODEL}", flush=True)
        print(f"LoRA adapter: {adapter_path}", flush=True)

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        tokenizer = configure_tokenizer(tokenizer)

        # ----------------------------------------------------
        # BASE MODEL
        # ----------------------------------------------------

        model = load_causal_lm(BASE_MODEL, device_map="auto", low_cpu_mem_usage=True)

        # ----------------------------------------------------
        # LORA ADAPTER
        # ----------------------------------------------------

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    # ========================================================
    # FULL MODEL
    # ========================================================

    else:
        if version == 1:
            model_path = FULL_MODEL_V1
        else:
            model_path = FULL_MODEL_V2
        print(f"Full model: {model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer = configure_tokenizer(tokenizer)
        model = load_causal_lm(model_path, device_map="auto", low_cpu_mem_usage=True)

    # --------------------------------------------------------
    # EVALUATION MODE
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # SAVE CONFIGURATION
    # --------------------------------------------------------

    loaded_version = version
    loaded_model_type = model_type
    print("Model loaded successfully.", flush=True)
    return model

# ============================================================
# HARDWARE INFO
# ============================================================

def get_hardware_info():
    if not torch.cuda.is_available():
        return f"GPU: CPU\nPyTorch: {torch.__version__}\nCUDA: Not available"
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    return f"GPU: {torch.cuda.get_device_name(0)}\nCUDA: {torch.version.cuda}\nPyTorch: {torch.__version__}\nVRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total\nCompute capability: {capability}\nDtype: {DTYPE}"

# ============================================================
# GENERATE PROMPT
# ============================================================

def generate_prompt(content):
    instruction = (
        "Analyze the following Mamba code, detect any flaws, give a short explanation, and give one or more corrected versions of the code. Ensure that the refactored options do not repeat the original code. Return the result as a Python dictionary with exactly these keys: 'Flaws' and 'Refactored Versions'."
    )
    prompt = f"""

Below is an instruction that describes a task,
paired with an input that provides further context.
Write a response that appropriately solves the following Task:

### Instruction:

{instruction}

### Code:

{content}

### Response:

"""
    return prompt

# ============================================================
# INFERENCE
# ============================================================

def generate_inference_output(text):
    global model
    global tokenizer
    if model is None:
        raise RuntimeError("Model has not been loaded.")
    if tokenizer is None:
        raise RuntimeError("Tokenizer has not been loaded.")

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = generate_prompt(text)

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(prompt, return_tensors="pt")

    # --------------------------------------------------------
    # FIND INPUT DEVICE
    # --------------------------------------------------------

    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_tokens = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_tokens}", flush=True)

    # --------------------------------------------------------
    # CUDA CLEANUP
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE, pad_token_id=tokenizer.eos_token_id)

    # --------------------------------------------------------
    # SYNCHRONIZE
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # TOKEN COUNTS
    # --------------------------------------------------------

    generated_tokens = outputs.shape[1] - input_tokens

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    generated_output = outputs[:, input_tokens:]
    output = tokenizer.decode(generated_output[0], skip_special_tokens=True).strip()
    return output, input_tokens, generated_tokens, inference_time

# ============================================================
# EXTRACT DICTIONARY
# ============================================================

def extract_clean_dict(text):
    if text is None:
        raise ValueError("Model output is None.")
    text = str(text).strip()
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()
    starts = [text.find("{'Flaws'"), text.find('{"Flaws"')]
    starts = [value for value in starts if value >= 0]
    if not starts:
        raise ValueError("No result dictionary starting with 'Flaws' was found.")
    start = min(starts)
    candidate = text[start:]
    depth = 0
    in_string = False
    string_quote = None
    escaped = False
    end_position = None
    for index, char in enumerate(candidate):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char in ("'", '"'):
            if not in_string:
                in_string = True
                string_quote = char
            elif char == string_quote:
                in_string = False
                string_quote = None
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_position = index + 1
                break
    if end_position is None:
        raise ValueError("Could not find the end of the result dictionary.")
    candidate = candidate[:end_position].strip()
    try:
        result = ast.literal_eval(candidate)
    except Exception as e:
        raise ValueError(f"Could not parse model output as a Python dictionary: {e}") from e
    if not isinstance(result, dict):
        raise ValueError("Model output is not a dictionary.")
    if "Flaws" not in result:
        raise ValueError("Missing 'Flaws'.")
    if "Refactored Versions" not in result:
        raise ValueError("Missing 'Refactored Versions'.")
    return result

# ============================================================
# FORMAT OUTPUT
# ============================================================

def format_output(output_dict):
    result = "Flaws:\n"
    flaws = output_dict["Flaws"]
    if isinstance(flaws, list):
        for entry in flaws:
            if isinstance(entry, dict):
                flaw = entry.get("Flaw", "Unknown flaw")
                explanation = entry.get("Explanation", "")
                result += f"   - {flaw}: {explanation}\n"
            else:
                result += f"   - {entry}\n"
    else:
        result += f"   - {flaws}\n"
    result += "\nRefactored versions code:\n\n"
    result += str(output_dict["Refactored Versions"])
    result += "\n\n"
    return result
