import ast
import re
import threading
import time
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# ------------------------------------------------------------
MISTRAL_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
DEEPSEEK_BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"
# ------------------------------------------------------------
MISTRAL_CHECKPOINTS = {
   ("Mamba", "Flaws + Refactoring", 1, "LoRA Adapter"):
      "HA-Siala/Mamba-v0.1",
   ("Mamba", "Flaws + Refactoring", 2, "LoRA Adapter"):
      "HA-Siala/Mamba-v0.2",
   ("Mamba", "Flaws + Refactoring", 1, "Full Model"):
      "HA-Siala/Mamba-full-v0.1",
   ("Mamba", "Flaws + Refactoring", 2, "Full Model"):
      "HA-Siala/Mamba-full-v0.2",
   ("Python", "Flaw Detection", 1, "LoRA Adapter"):
      "HA-Siala/Detect-Flaws-v0.1",
   ("Python", "Flaw Detection", 2, "LoRA Adapter"):
      "HA-Siala/Detect-Flaws-v0.2",
   ("Python", "Flaw Detection", 1, "Full Model"):
      "HA-Siala/Detect-Flaws-full-v0.1",
   ("Python", "Flaw Detection", 2, "Full Model"):
      "HA-Siala/Detect-Flaws-full-v0.2",
   ("Python", "Refactoring", 1, "LoRA Adapter"):
      "HA-Siala/RefactoringPy-v0.1",
   ("Python", "Refactoring", 1, "Full Model"):
      "HA-Siala/RefactoringPy-full-v0.1",
}
# ------------------------------------------------------------
DEEPSEEK_CHECKPOINTS = {
   ("Python", 1, "LoRA Adapter"):
      "HA-Siala/RefactoringPy-DeepSeek-v0.1",
   ("Python", 1, "Full Model"):
      "HA-Siala/RefactoringPy-DeepSeek-full-v0.1",
   ("Mamba", 1, "LoRA Adapter"):
      "HA-Siala/Mamba-DeepSeek-v0.1",
   ("Mamba", 1, "Full Model"):
      "HA-Siala/Mamba-DeepSeek-full-v0.1",
}
# ------------------------------------------------------------
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
# ------------------------------------------------------------
MODEL = None
TOKENIZER = None
MODEL_INFO = None
# ------------------------------------------------------------
def _clear_cuda_cache():
   if torch.cuda.is_available():
      torch.cuda.empty_cache()
# ------------------------------------------------------------
def _load_mistral(language, task, version, model_type):
   key = (
      language,
      task,
      version,
      model_type,
   )
   if key not in MISTRAL_CHECKPOINTS:
      raise ValueError(f"No Mistral checkpoint configured for: {key}")
   checkpoint = MISTRAL_CHECKPOINTS[key]
   tokenizer = AutoTokenizer.from_pretrained(
      MISTRAL_BASE_MODEL,
      use_fast=True,
   )
   tokenizer.pad_token = tokenizer.unk_token
   tokenizer.padding_side = "left"
   if model_type == "LoRA Adapter":
      base_model = AutoModelForCausalLM.from_pretrained(
         MISTRAL_BASE_MODEL,
         torch_dtype=torch.bfloat16,
         device_map="auto",
      )
      model = PeftModel.from_pretrained(
         base_model,
         checkpoint,
         torch_dtype=torch.bfloat16,
         is_trainable=False,
      )
   else:
      model = AutoModelForCausalLM.from_pretrained(
         checkpoint,
         torch_dtype=torch.bfloat16,
         device_map="auto",
      )
   model.eval()
   return (
      model,
      tokenizer,
      checkpoint,
   )
# ------------------------------------------------------------
def _load_deepseek(language, version, model_type):
   key = (
      language,
      version,
      model_type,
   )
   if key not in DEEPSEEK_CHECKPOINTS:
      raise ValueError(f"No DeepSeek checkpoint configured for: {key}")
   checkpoint = DEEPSEEK_CHECKPOINTS[key]
# ------------------------------------------------------------
   if model_type == "LoRA Adapter":
      tokenizer = AutoTokenizer.from_pretrained(
         checkpoint,
         trust_remote_code=True,
      )
      if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
      tokenizer.padding_side = "left"
      base_model = AutoModelForCausalLM.from_pretrained(
         DEEPSEEK_BASE_MODEL,
         torch_dtype=torch.bfloat16,
         device_map="auto",
         trust_remote_code=True,
      )
      model = PeftModel.from_pretrained(
         base_model,
         checkpoint,
      )
# ------------------------------------------------------------
   else:
      tokenizer = AutoTokenizer.from_pretrained(
         checkpoint,
         use_fast=True,
         trust_remote_code=True,
      )
      if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
      tokenizer.padding_side = "left"
      model = AutoModelForCausalLM.from_pretrained(
         checkpoint,
         torch_dtype=torch.bfloat16,
         device_map="auto",
         low_cpu_mem_usage=True,
         trust_remote_code=True,
      )
   model.eval()
   return (
      model,
      tokenizer,
      checkpoint,
   )
# ------------------------------------------------------------
def load_model(language, task, version, model_type, model_family="Mistral"):
   global MODEL
   global TOKENIZER
   global MODEL_INFO
   with MODEL_LOCK:
      unload_model()
      if model_family == "Mistral":
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
      elif model_family == "DeepSeek":
         (
            model,
            tokenizer,
            checkpoint,
         ) = _load_deepseek(
            language=language,
            version=version,
            model_type=model_type,
         )
      else:
         raise ValueError(f"Unsupported model family: {model_family}")
      MODEL = model
      TOKENIZER = tokenizer
      MODEL_INFO = {
         "model_family": model_family,
         "language": language,
         "task": task,
         "version": int(version),
         "model_type": model_type,
         "checkpoint": checkpoint,
      }
      return MODEL_INFO.copy()
# ------------------------------------------------------------
def unload_model():
   global MODEL
   global TOKENIZER
   global MODEL_INFO
   MODEL = None
   TOKENIZER = None
   MODEL_INFO = None
   gc.collect()
   if torch.cuda.is_available():
      torch.cuda.empty_cache()
      try:
         torch.cuda.ipc_collect()
      except Exception:
         pass
# ------------------------------------------------------------
def is_model_loaded():
   return MODEL is not None
# ------------------------------------------------------------
def get_loaded_model_info():
   if MODEL_INFO is None:
      return None
   return MODEL_INFO.copy()
# ------------------------------------------------------------
def _mistral_instruction(language, task):
   if language == "Mamba":
      return (
         "Analyze the following Mamba code, detect any flaws, "
         "with a short explanation, and give one or more "
         "corrected versions of the code. Ensure that none "
         "of the refactored options repeat or reproduce the "
         "original code; only the improved version should be shown:"
      )
   return (
      "Analyze the following Python code, identify any flaws, "
      "with a short explanation, and give one or more "
      "corrected versions of the code, including a brief "
      "explanation of each correction approach:"
   )
# ------------------------------------------------------------
def _generate_mistral_prompt(language, task, content):
   instruction = _mistral_instruction(
      language=language,
      task=task,
   )
   return (
      "Below is an instruction that describes a task, paired "
      "with an input that provides further context. Write a "
      "response that appropriately solves the following Task:\n\n"
      "### Instruction:\n"
      f"{instruction}\n\n"
      "### Code:\n"
      f"{content}\n"
      "### Response:"
   )
# ------------------------------------------------------------
def _deepseek_instruction(language):
   if language == "Python":
      return (
         "Analyze the following Python code, detect any flaws, "
         "with a short explanation, and give one or more "
         "corrected versions of the code. Ensure that none "
         "of the refactored options repeat or reproduce the "
         "original code; only the improved version should be shown."
      )
   return (
      "Analyze the following Mamba code, detect any flaws, "
      "with a short explanation, and give one or more "
      "corrected versions of the code. Ensure that none "
      "of the refactored options repeat or reproduce the "
      "original code; only the improved version should be shown."
   )
# ------------------------------------------------------------
def _generate_deepseek_prompt(language, content):
   instruction = _deepseek_instruction(
      language=language,
   )
   return (
      "You are an AI programming assistant, utilizing the "
      "DeepSeek Coder model.\n"
      "### Instruction:\n"
      f"{instruction}\n\n"
      "### Input:\n"
      f"{content}\n\n"
      "### Response:"
   )
# ------------------------------------------------------------
def generate_prompt(language, task, content, model_family="Mistral"):
   if model_family == "DeepSeek":
      return _generate_deepseek_prompt(
         language=language,
         content=content,
      )
   return _generate_mistral_prompt(
      language=language,
      task=task,
      content=content,
   )
# ------------------------------------------------------------
def _generate_mistral_inference(content, return_metrics=False):
   if MODEL is None or TOKENIZER is None:
      raise RuntimeError("No model is loaded.")
   start_time = time.time()
   language = MODEL_INFO["language"]
   task = MODEL_INFO["task"]
   TOKENIZER.pad_token = TOKENIZER.unk_token
   prompt = _generate_mistral_prompt(
      language=language,
      task=task,
      content=content,
   )
   inputs = TOKENIZER(
      prompt,
      return_tensors="pt",
   ).to("cuda:0")
   input_tokens = inputs["input_ids"].shape[1]
   torch.cuda.empty_cache()
   outputs = None
   with torch.inference_mode():
      outputs = MODEL.generate(
         **inputs,
         max_new_tokens=32768,
         temperature=0.0,
         do_sample=False,
         pad_token_id=TOKENIZER.eos_token_id,
         top_p=1.0,
      )
      output_p = TOKENIZER.batch_decode(
         outputs,
         skip_special_tokens=True,
      )
      if output_p:
         output_text = output_p[0]
         split_text = output_text.split("Response:")
         if len(split_text) > 1:
            raw_output = split_text[1].strip()
         else:
            raw_output = None
      else:
         raw_output = None
   inference_time = time.time() - start_time
   if outputs is not None:
      generated_tokens = max(0, outputs.shape[1] - input_tokens)
   else:
      generated_tokens = 0
# ------------------------------------------------------------
   result = (
      raw_output,
      input_tokens,
      generated_tokens,
      inference_time,
   )
# ------------------------------------------------------------
   del inputs
   if outputs is not None:
      del outputs
   gc.collect()
   if torch.cuda.is_available():
      torch.cuda.empty_cache()
   return result
# ------------------------------------------------------------
def _generate_deepseek_inference(content, return_metrics=False):
   if MODEL is None or TOKENIZER is None:
      raise RuntimeError("No model is loaded.")
   start_time = time.time()
   language = MODEL_INFO["language"]
   prompt = _generate_deepseek_prompt(
      language=language,
      content=content,
   )
   inputs = TOKENIZER(
      prompt,
      return_tensors="pt",
   ).to("cuda:0")
   input_tokens = inputs["input_ids"].shape[1]
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
   response = TOKENIZER.decode(
      outputs[0][inputs["input_ids"].shape[1]:],
      skip_special_tokens=True,
   )
   inference_time = time.time() - start_time
   generated_tokens = max(0, outputs.shape[1] - input_tokens)
# ------------------------------------------------------------
   result = (
      response,
      input_tokens,
      generated_tokens,
      inference_time,
   )
# ------------------------------------------------------------
   del inputs
   del outputs
   gc.collect()
   if torch.cuda.is_available():
      torch.cuda.empty_cache()
   return result
# ------------------------------------------------------------
def generate_inference_output(content, return_metrics=False):
   with INFERENCE_LOCK:
      if MODEL is None:
         raise RuntimeError("No model is loaded.")
      if MODEL_INFO is None:
         raise RuntimeError("Model information is unavailable.")
      if MODEL_INFO["model_family"] == "DeepSeek":
         return _generate_deepseek_inference(
            content=content,
            return_metrics=return_metrics,
         )
      return _generate_mistral_inference(
         content=content,
         return_metrics=return_metrics,
      )
# ------------------------------------------------------------
def _extract_mistral_dict(text):
   text = str(text)
   if "### Response:" in text:
      text = text.split("### Response:")[-1].strip()
   else:
      text = text.strip()
   match = re.search(r"(\{'Flaws'[\s\S]*\})", text)
   if not match:
      raise ValueError("No dict found")
   candidate = match.group(1)
   return ast.literal_eval(candidate)
# ------------------------------------------------------------
def _clean_deepseek_text(text):
   text = str(text)
   text = text.split("###")[0]
   return text
# ------------------------------------------------------------
def _extract_deepseek_dict(text):
   if not text:
      raise ValueError("Empty DeepSeek output")
   text = _clean_deepseek_text(text)
   flaws = re.findall(r'"Flaw"\s*:\s*"([^"]+)"', text)
   explanations = re.findall(r'"Explanation"\s*:\s*"([^"]+)"', text)
   result = {
      "Flaws": [],
      "Refactored Versions": "",
   }
   for i in range(min(len(flaws), len(explanations))):
      result["Flaws"].append(
         {
            "Flaw": flaws[i],
            "Explanation": explanations[i],
         }
      )
   refactored = re.search(r'"Refactored Versions"\s*:\s*"([\s\S]+)"', text)
   if refactored:
      cleaned_code = refactored.group(1)
      cleaned_code = cleaned_code.replace('\\"', '"')
      cleaned_code = cleaned_code.replace("\\n", "\n")
      result["Refactored Versions"] = (cleaned_code)
   elif not result["Flaws"]:
      raise ValueError("Invalid DeepSeek output")
   return result
# ------------------------------------------------------------
def extract_clean_dict(text):
   if MODEL_INFO is None:
      raise RuntimeError("No model information is available.")
   if MODEL_INFO["model_family"] == "DeepSeek":
      return _extract_deepseek_dict(text)
   return _extract_mistral_dict(text)
# ------------------------------------------------------------
def ExtractCleanDict(text):
   return extract_clean_dict(text)
# ------------------------------------------------------------
def CleanText(text):
   return _clean_deepseek_text(text)
# ------------------------------------------------------------
def format_output(output_dict):
   if MODEL_INFO is None:
      raise RuntimeError("No model information is available.")
   output_lines = []
   output_lines.append("Flaws:")
   flaws = output_dict.get("Flaws", [])
   for entry in flaws:
      if isinstance(entry, dict):
         flaw = entry.get("Flaw", "")
         explanation = entry.get("Explanation", "")
         output_lines.append(f"   - {flaw}: {explanation}")
      else:
         output_lines.append(f"   - {entry}")
   if (MODEL_INFO["model_family"] == "Mistral" and MODEL_INFO["language"] == "Python" and MODEL_INFO["task"] == "Flaw Detection"):
      output_lines.append("")
      output_lines.append("Corrected code recommendation:")
      output_lines.append("")
      corrections = output_dict.get("Corrections")
      if isinstance(corrections, list) and corrections:
         for item in corrections:
            if isinstance(item, dict):
               code = item.get("Correction", "")
               if code:
                  output_lines.append(str(code))
                  output_lines.append("")
            elif item:
               output_lines.append(str(item))
               output_lines.append("")
      elif "Correction" in output_dict:
         correction = output_dict.get("Correction", "")
         if correction:
            output_lines.append(str(correction))
   else:
      output_lines.append("")
      output_lines.append("Refactored versions code:")
      output_lines.append("")
      refactored = output_dict.get("Refactored Versions", "")
      if isinstance(refactored, str):
         output_lines.append(refactored)
      elif isinstance(refactored, list):
         for item in refactored:
            if isinstance(item, dict):
               code = item.get(
                  "Code",
                  item.get(
                     "Refactored Code",
                     item.get(
                        "Refactored",
                        ""
                     )
                  )
               )
               if code:
                  output_lines.append(str(code))
            elif item:
               output_lines.append(str(item))
            output_lines.append("")
      elif refactored:
         output_lines.append(str(refactored))
   return "\n".join(output_lines)
# ------------------------------------------------------------
def _format_time(seconds):
   hours = int(seconds // 3600)
   minutes = int((seconds % 3600) // 60)
   remaining_seconds = (seconds % 60)
   result = ""
   if hours > 0:
      result += (f"{hours}h ")
   if (minutes > 0 or hours > 0):
      result += (f"{minutes}m ")
   result += (f"{remaining_seconds:.6f}s")
   return result
# ------------------------------------------------------------
def append_inference_metrics(final_output, language, task, model_type, model_version, input_tokens, generated_tokens, inference_time):
   if input_tokens > 0:
      time_per_input_token = (inference_time / input_tokens)
   else:
      time_per_input_token = 0.0
   if generated_tokens > 0:
      time_per_generated_token = (inference_time / generated_tokens)
   else:
      time_per_generated_token = 0.0
   metrics = (
      "\n\n===========================================\n"
      "Inference Metrics\n"
      "===========================================\n"
      f"Language:                  {language}\n"
      f"Task:                      {task}\n"
      f"Model type:                {model_type}\n"
      f"Model version:             {model_version}\n"
      f"Input tokens:              {input_tokens}\n"
      f"Generated tokens:          {generated_tokens}\n"
      f"Inference time:            "
      f"{_format_time(inference_time)}\n"
      f"Time per input token:      "
      f"{_format_time(time_per_input_token)}\n"
      f"Time per generated token:  "
      f"{_format_time(time_per_generated_token)}\n"
      "===========================================\n"
   )
   return (
      final_output
      + metrics
   )
# ------------------------------------------------------------
