import gc
import re
import ast
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-v0.3"

MAX_NEW_TOKENS = 32768

model = None
tokenizer = None

loaded_version = None
loaded_model_type = None

def get_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

DTYPE = get_dtype()

def get_hardware_info():
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
        lines.append(f"GPU {i}: {name} " f"({total_gb:.2f} GB)")
    return "\n".join(lines)

def is_model_loaded():
    return (model is not None and tokenizer is not None)

def get_loaded_model_info():
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

def configure_tokenizer(tok):
    if tok.unk_token is not None:
        tok.pad_token = tok.unk_token
    elif tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def get_checkpoint(version, model_type):
    if version not in [1, 2]:
        raise ValueError("Model version must be 1 or 2.")
    if model_type not in ["LoRA Adapter", "Full Model"]:
        raise ValueError("Model type must be " "'LoRA Adapter' or 'Full Model'.")
    if model_type == "LoRA Adapter":
        if version == 1:
            return "HA-Siala/Mamba-v0.1"
        return "HA-Siala/Mamba-v0.2"
    if version == 1:
        return "HA-Siala/Mamba-full-v0.1"
    return "HA-Siala/Mamba-full-v0.2"

def clear_model():
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

def load_model(version, model_type):
    global model
    global tokenizer
    global loaded_version
    global loaded_model_type
    version = int(version)
    checkpoint = get_checkpoint(version, model_type)
    if (model is not None and tokenizer is not None and loaded_version == version and loaded_model_type == model_type):
        print("Requested model is already loaded.", flush=True,)
        return {
            "status": "already_loaded",
            "version": version,
            "model_type": model_type,
            "checkpoint": checkpoint,
        }
    if model is not None:
        print("A different model is loaded. " "Clearing it first...", flush=True,)
        clear_model()
    print(f"Loading model version {version} " f"({model_type})...", flush=True,)
    print(f"Checkpoint: {checkpoint}", flush=True,)
    print(f"Dtype: {DTYPE}", flush=True,)
    if model_type == "LoRA Adapter":
        print("Loading base Mistral model...", flush=True,)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True,)
        tokenizer = configure_tokenizer(tokenizer)
        base_model = (AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE, device_map="auto",))
        print("Loading LoRA adapter...", flush=True,)
        model = PeftModel.from_pretrained(base_model, checkpoint, dtype=DTYPE, is_trainable=False,)
    else:
        print("Loading full Mamba model...", flush=True,)
        model = (AutoModelForCausalLM.from_pretrained(checkpoint, dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True,))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True,)
        tokenizer = configure_tokenizer(tokenizer)
    model.eval()
    loaded_version = version
    loaded_model_type = model_type
    print("Model loaded successfully.", flush=True,)
    return {
        "status": "loaded",
        "version": version,
        "model_type": model_type,
        "checkpoint": checkpoint,
    }

def generate_prompt(content):
    instruction = """Analyze the following Mamba code, detect any flaws, with a short explanation, and give one or more corrected versions of the code. Ensure that none of the refactored options repeat or reproduce the original code; only the improved version should be shown:"""
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately solves the following Task:

### Instruction:
{instruction}

### Code:
{content}
### Response:
"""
    return prompt

def get_input_device():
    if model is None:
        raise RuntimeError("Model is not loaded.")
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def generate_inference_output(text):
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded. " "Call load_model() before inference.")
    prompt = generate_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt",)
    input_tokens = inputs["input_ids"].shape[1]
    input_device = get_input_device()
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    print(f"Input tokens: {input_tokens}", flush=True,)
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id,)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    generated_tokens = (outputs.shape[1] - input_tokens)
    generated_tokens = max(0, generated_tokens,)
    generated_ids = outputs[:, input_tokens:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,)
    if decoded:
        output_text = decoded[0].strip()
    else:
        output_text = ""
    reached_limit = (generated_tokens >= MAX_NEW_TOKENS)
    print("============================================================", flush=True,)
    print("GENERATION DIAGNOSTICS", flush=True,)
    print(f"Input tokens: {input_tokens}", flush=True,)
    print(f"Generated tokens: {generated_tokens}", flush=True,)
    print(f"Maximum generated tokens: " f"{MAX_NEW_TOKENS}", flush=True,)
    print(f"Reached token limit: " f"{reached_limit}", flush=True,)
    print(f"Output characters: " f"{len(output_text)}", flush=True,)
    print("============================================================", flush=True,)
    return (
        output_text,
        input_tokens,
        generated_tokens,
        inference_time,
    )

def extract_clean_dict(text):
    if text is None:
        raise ValueError("Model output is None.")
    text = str(text).strip()
    if not text:
        raise ValueError("Model output is empty.")
    if "### Response:" in text:
        text = text.split("### Response:", 1,)[1].strip()
    text = re.sub(r"^```(?:python|json)?\s*", "", text, flags=re.IGNORECASE,)
    text = re.sub(r"\s*```$", "", text,)
    text = text.strip()
    start_positions = []
    python_start = text.find("{")
    json_start = text.find("{")
    if python_start >= 0:
        start_positions.append(python_start)
    if json_start >= 0:
        start_positions.append(json_start)
    if not start_positions:
        raise ValueError("No dictionary found in model output.")
    start = min(start_positions)
    depth = 0
    in_string = False
    string_char = None
    escaped = False
    end = None
    for i in range(start, len(text)):
        char = text[i]
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
        raise ValueError("No complete dictionary found. " "The model output appears to be truncated.")
    candidate = text[start:end].strip()
    try:
        result = ast.literal_eval(candidate)
    except Exception as e:
        try:
            import json
            result = json.loads(candidate)
        except Exception:
            raise ValueError("Could not parse model output " f"as a dictionary: {e}")
    if not isinstance(result, dict):
        raise ValueError("Parsed model output is not a dictionary.")
    if "Flaws" not in result:
        raise ValueError("Model output does not contain " "'Flaws'.")
    if "Refactored Versions" not in result:
        raise ValueError("Model output does not contain " "'Refactored Versions'.")
    return result

def format_output(output_dict):
    result = ""
    result += "Flaws:\n"
    flaws = output_dict.get("Flaws", [],)
    if isinstance(flaws, list):
        for entry in flaws:
            if isinstance(entry, dict):
                flaw = entry.get("Flaw", "",)
                explanation = entry.get("Explanation", "",)
                result += (f"   - {flaw}: " f"{explanation}\n")
            else:
                result += (f"   - {entry}\n")
    else:
        result += (f"   - {flaws}\n")
    result += ("\n" "Refactored versions code:\n" "\n")
    refactored = output_dict.get("Refactored Versions", "",)
    result += str(refactored)
    result += "\n\n"
    return result
