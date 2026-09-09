import time
import ast

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    PeftModel
)


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
    # A100 / AMPERE
    # --------------------------------------------------------

    if GPU_MAJOR == 8:

        GPU_FAMILY = (
            "Ampere (A100 compatible)"
        )

    # --------------------------------------------------------
    # BLACKWELL
    # --------------------------------------------------------

    elif GPU_MAJOR >= 10:

        GPU_FAMILY = (
            "Blackwell (B200 compatible)"
        )

    # --------------------------------------------------------
    # OTHER CUDA GPU
    # --------------------------------------------------------

    else:

        GPU_FAMILY = (
            "Other CUDA GPU"
        )

    # --------------------------------------------------------
    # SELECT DTYPE
    # --------------------------------------------------------

    if torch.cuda.is_bf16_supported():

        DTYPE = torch.bfloat16

    else:

        DTYPE = torch.float16


# ============================================================
# DISPLAY HARDWARE INFORMATION
# ============================================================

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
    f"BF16 supported:     "
    f"{bf16_supported}",
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

MAX_NEW_TOKENS = 32768

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
# CURRENTLY LOADED CONFIGURATION
#
# Used to avoid loading the same model repeatedly.
# ============================================================

loaded_version = None

loaded_model_type = None


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

    Try dtype first, then fall back.
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
# CLEAR CURRENT MODEL
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

def load_model(
    version=2,
    model_type="LoRA Adapter"
):

    global model

    global tokenizer

    global loaded_version

    global loaded_model_type


    # --------------------------------------------------------
    # REUSE MODEL
    #
    # If exactly the same model is already loaded,
    # do not download/load it again.
    # --------------------------------------------------------

    if (

        model is not None

        and tokenizer is not None

        and loaded_version == version

        and loaded_model_type == model_type

    ):

        print(

            "Requested model is already loaded. "
            "Reusing the existing model.",

            flush=True
        )

        return model


    # --------------------------------------------------------
    # DIFFERENT MODEL REQUESTED
    # --------------------------------------------------------

    if model is not None:

        print(

            "Different model requested. "
            "Clearing current model...",

            flush=True
        )

        clear_model()


    # --------------------------------------------------------
    # START LOADING
    # --------------------------------------------------------

    print(

        f"Loading model "
        f"(version={version}, "
        f"type={model_type})...",

        flush=True
    )


    # ========================================================
    # LORA ADAPTER
    # ========================================================

    if model_type == "LoRA Adapter":


        # ----------------------------------------------------
        # SELECT ADAPTER VERSION
        # ----------------------------------------------------

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


        if tokenizer.unk_token is not None:

            tokenizer.pad_token = (

                tokenizer.unk_token
            )


        elif tokenizer.pad_token is None:

            tokenizer.pad_token = (

                tokenizer.eos_token
            )


        tokenizer.padding_side = (
            "left"
        )


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


        model = (

            LoadCausalLM(

                BASE_MODEL,

                device_map="auto",

                low_cpu_mem_usage=True
            )
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


    # ========================================================
    # FULL MODEL
    # ========================================================

    elif model_type == "Full Model":


        # ----------------------------------------------------
        # SELECT MODEL VERSION
        # ----------------------------------------------------

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


        tokenizer.padding_side = (
            "left"
        )


        print(

            "Tokenizer loaded.",

            flush=True
        )


        # ----------------------------------------------------
        # FULL MODEL
        # ----------------------------------------------------

        print(

            "Loading full model...",

            flush=True
        )


        model = (

            LoadCausalLM(

                model_path,

                device_map="auto",

                low_cpu_mem_usage=True
            )
        )


        print(

            "Full model loaded.",

            flush=True
        )


    # ========================================================
    # INVALID MODEL TYPE
    # ========================================================

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


    # --------------------------------------------------------
    # SAVE LOADED CONFIGURATION
    # --------------------------------------------------------

    loaded_version = version

    loaded_model_type = model_type


    print(

        "Model loaded successfully.",

        flush=True
    )


    return model


# ============================================================
# GENERATE PROMPT
# ============================================================

def GeneratePrompt(
    content
):

    Instruction = (

        "Analyze the following Mamba code, "
        "detect any flaws, with a short explanation, "
        "and give one or more corrected versions of "
        "the code. Ensure that none of the refactored "
        "options repeat or reproduce the original code; "
        "only the improved version should be shown:"
    )


    prompt = f"""

Below is an instruction that describes a task,
paired with an input that provides further context.
Write a response that appropriately solves the following Task:

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

def GenerateInferenceOutput(
    text
):

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
    # TOKENIZER CONFIGURATION
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
    # GENERATE PROMPT
    # --------------------------------------------------------

    prompt = GeneratePrompt(
        text
    )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = (

        tokenizer(

            prompt,

            return_tensors="pt"
        )
    )


    # --------------------------------------------------------
    # MOVE INPUT TO MODEL DEVICE
    #
    # This is safer than forcing cuda:0 when using
    # device_map="auto".
    # --------------------------------------------------------

    model_device = next(
        model.parameters()
    ).device


    inputs = inputs.to(
        model_device
    )


    input_tokens = (

        inputs[
            "input_ids"
        ]
        .shape[1]
    )


    print(

        f"Input tokens: "
        f"{input_tokens}",

        flush=True
    )


    # --------------------------------------------------------
    # CLEAR CUDA CACHE
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
    # GENERATION SETTINGS
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

        outputs = (

            model.generate(

                **inputs,

                **generation_kwargs
            )
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
    # GENERATED TOKENS
    # --------------------------------------------------------

    generated_tokens = (

        outputs.shape[1]

        - input_tokens
    )


    # --------------------------------------------------------
    # DECODE ONLY GENERATED TOKENS
    #
    # This avoids splitting the entire prompt using
    # the string "Response:".
    # --------------------------------------------------------

    generated_output = (

        outputs[
            :,
            input_tokens:
        ]
    )


    output_text = (

        tokenizer.decode(

            generated_output[0],

            skip_special_tokens=True
        )
    )


    output = (
        output_text.strip()
    )


    return (

        output,

        input_tokens,

        generated_tokens,

        inference_time
    )


# ============================================================
# EXTRACT CLEAN DICTIONARY
# ============================================================

def ExtractCleanDict(
    text
):

    if text is None:

        raise ValueError(

            "Model output is None."
        )


    text = str(
        text
    ).strip()


    # --------------------------------------------------------
    # REMOVE RESPONSE MARKER
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
    # FIND DICTIONARY START
    # --------------------------------------------------------

    start = text.find(
        "{'Flaws'"
    )


    if start == -1:

        raise ValueError(

            "No dictionary starting with "
            "{'Flaws' was found."
        )


    candidate = text[
        start:
    ]


    # --------------------------------------------------------
    # FIND MATCHING BRACE
    # --------------------------------------------------------

    depth = 0

    in_string = False

    string_quote = None

    escaped = False

    end_position = None


    for index, char in enumerate(
        candidate
    ):


        if escaped:

            escaped = False

            continue


        if char == "\\" and in_string:

            escaped = True

            continue


        # ----------------------------------------------------
        # STRING HANDLING
        # ----------------------------------------------------

        if char in (

            "'",

            '"'

        ):


            if not in_string:

                in_string = True

                string_quote = char


            elif char == string_quote:

                in_string = False

                string_quote = None


            continue


        # ----------------------------------------------------
        # IGNORE BRACES INSIDE STRINGS
        # ----------------------------------------------------

        if in_string:

            continue


        # ----------------------------------------------------
        # TRACK BRACES
        # ----------------------------------------------------

        if char == "{":

            depth += 1


        elif char == "}":

            depth -= 1


            if depth == 0:

                end_position = (
                    index + 1
                )

                break


    # --------------------------------------------------------
    # VALIDATE END
    # --------------------------------------------------------

    if end_position is None:

        raise ValueError(

            "Could not find the end of the "
            "model output dictionary."
        )


    candidate = (

        candidate[
            :end_position
        ]

        .strip()
    )


    # --------------------------------------------------------
    # PARSE DICTIONARY
    # --------------------------------------------------------

    try:

        result = (

            ast.literal_eval(
                candidate
            )
        )


    except Exception as e:

        raise ValueError(

            "Could not parse model output "
            f"as a Python dictionary: {e}"

        ) from e


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict
    ):

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

def FormatOutput(
    output_dict
):

    result = ""


    # --------------------------------------------------------
    # FLAWS
    # --------------------------------------------------------

    result += (
        "Flaws:\n"
    )


    for entry in output_dict[
        "Flaws"
    ]:

        result += (

            f"   - "
            f"{entry['Flaw']}: "
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

        output_dict[
            "Refactored Versions"
        ]
    )


    result += (
        "\n\n"
    )


    return result
