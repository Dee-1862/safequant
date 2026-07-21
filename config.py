# SafeQuant model aliases, HF cache paths, and quantized checkpoint registry.
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(PROJECT_ROOT, "phase1_problem")
DATA_DIR = os.path.join(PHASE1_DIR, "data", "prompts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _load_dotenv():
    """Load .env from repo root (does not override variables already in the shell)."""
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding = "utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)


_load_dotenv()

# Set HF_HOME and QM_ROOT in .env (see .env.example). Laptop defaults if unset.
HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
QM_ROOT = os.environ.get("QM_ROOT") or os.path.join(os.path.expanduser("~"), "quantized_models")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("QM_ROOT", QM_ROOT)

_QM = QM_ROOT
# -- Models -------------------------------------------------------------------
# Short aliases -> HuggingFace model IDs.
# Any HF model ID can also be passed directly (no alias needed).
MODELS = {
    # Qwen2.5 series (primary research models)
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b":  "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
    # LLaMA 3.t2 / 3.1 (for cross-architecture validation)
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama7b": "meta-llama/Llama-3.1-8B-Instruct",
    # Mistral (for cross-architecture validation)
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    # Gemma 2 (for cross-architecture validation)
    "gemma2b": "google/gemma-2-2b-it",
    "gemma9b": "google/gemma-2-9b-it",
    # Decoding Compressed Trust paper models
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    # Llama-3-8B (for AQLM pre-quantized checkpoint)
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    # Qwen3 8B (second Qwen architecture, for cross-architecture validation)
    "qwen3-8b": "Qwen/Qwen3-8B",
    # Gemma 7B (4th architecture, for cross-architecture validation)
    "gemma7b": "google/gemma-7b-it",
    # Phi-3-mini (fused-projection architecture: qkv_proj / gate_up_proj)
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

# FP-base revision pins. Some HF repos updated their WEIGHTS after a pre-quantized
# checkpoint was made, so HEAD no longer matches the quantizer's source weights.
# Pin the FP base to the revision the AQLM checkpoint was quantized from, else the
# read/write restoration injects mismatched weights (both_restored != FP32).
#   phi3: AQLM created 2024-05-29; microsoft updated weights on 2024-07-01
#         ("Update model #77"). ff07dc01 (2024-06-11) is the last original-weight commit.
MODEL_REVISIONS = {
    "phi3": "ff07dc01615f8113924aed013115ab2abd32115b",
}

# Pre-quantized checkpoint aliases (for multi-quantizer comparison)
# These map (base_model_alias, quantizer) -> HuggingFace checkpoint ID
QUANTIZED_CHECKPOINTS = {
    # Qwen2.5-1.5B-Instruct
    
    ("qwen1.5b", "gptq4"): "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    ("qwen1.5b", "gptq8"): "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
    ("qwen1.5b", "awq"):  "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    # Qwen2.5-7B-Instruct
    ("qwen7b", "gptq4"):  f"{_QM}/gptq/qwen7b-gptq4bit-g128",   # self-gen (AutoGPTQ)
    ("qwen7b", "gptq3"):  f"{_QM}/gptq/qwen7b-gptq3bit-g128",
    ("qwen7b", "gptq2"):  f"{_QM}/gptq/qwen7b-gptq2bit-g128",
    ("qwen7b", "gptq8"):  "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
    ("qwen7b", "awq"):    f"{_QM}/awq/qwen7b",             # Qwen/Qwen2.5-7B-Instruct-AWQ
    # Qwen3-8B
    ("qwen3-8b", "gptq4"): "JunHowie/Qwen3-8B-GPTQ-Int4",
    ("qwen3-8b", "awq"):  "Qwen/Qwen3-8B-AWQ",
    # Gemma 7B (Gemma-1 generation)
    ("gemma7b", "gptq4"): "TechxGenus/gemma-7b-it-GPTQ",
    ("gemma7b", "awq"):   "casperhansen/gemma-7b-it-awq",
    # Llama-3.1-8B-Instruct GPTQ: self-generated via scripts/quantize_gptq.py using
    # AutoGPTQ (bits/group 128, desc_act, sym, damp 0.1, WikiText-2 calib n=128,
    # seed 0 -- the hugging-quants recipe). AutoGPTQ (not gptqmodel) makes gptq4
    # match the literature (cos >0.97); gptq3/gptq2 give the low-bit degradation
    # curve under the SAME recipe. See docs/MODELS_QUANTIZATION.md.
    ("llama7b", "gptq4"): f"{_QM}/gptq/llama7b-gptq4bit-g128",
    ("llama7b", "gptq3"): f"{_QM}/gptq/llama7b-gptq3bit-g128",
    ("llama7b", "gptq2"): f"{_QM}/gptq/llama7b-gptq2bit-g128",
    ("llama7b", "gptq8"): "ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-8bit",
    # Llama-3.2-1B-Instruct (fits on single V100)
    ("llama1b", "gptq4"): "ModelCloud/Llama-3.2-1B-Instruct-gptq-4bit",
    # AQLM pre-quantized (SOTA 2-bit)
    ("llama3-8b", "aqlm2"): "ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
    ("llama3b", "aqlm2"): "ISTA-DASLab/Llama-3.2-3B-Instruct-AQLM-PV-2Bit-2x8",
    # Llama-3.1-8B multi-quantizer comparison (matching arxiv 2502.15799)
    ("llama7b", "awq"):   f"{_QM}/awq/llama7b",   # hugging-quants/...-AWQ-INT4
    ("llama7b", "aqlm2"): f"{_QM}/aqlm/llama7b",  # ISTA-DASLab/...-AQLM-PV-2Bit-1x16-hf
    ("llama7b", "quip4"): "kharinaev/Llama-3.1-8B-Instruct-quip-sharp-4bit",
    ("llama7b", "quip2"): "kharinaev/Llama-3.1-8B-Instruct-quip-sharp-2bit",
    # Decoding Compressed Trust -- exact models from the paper
    ("llama2-13b-chat", "gptq4"): "compressed-llm/llama-2-13b-chat-gptq",
    ("llama2-13b-chat", "gptq3"): "compressed-llm/llama-2-13b-chat-gptq",
    ("llama2-13b-chat", "gptq8"): "compressed-llm/llama-2-13b-chat-gptq",
    ("llama2-13b-chat", "awq"):  "compressed-llm/llama-2-13b-chat-awq",
    # Mistral-7B-Instruct-v0.3
    ("mistral7b", "gptq4"): "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    # Mistral-7B-Instruct-v0.2 (matches Kharinaev et al. 2025)
    ("mistral7b-v02", "gptq4"): f"{_QM}/gptq/mistral7b-v02-gptq4bit-g128",   # self-gen (AutoGPTQ)
    ("mistral7b-v02", "gptq3"): f"{_QM}/gptq/mistral7b-v02-gptq3bit-g128",
    ("mistral7b-v02", "gptq2"): f"{_QM}/gptq/mistral7b-v02-gptq2bit-g128",
    ("mistral7b-v02", "awq"):  f"{_QM}/awq/mistral7b-v02",   # TheBloke/...-v0.2-AWQ
    ("mistral7b-v02", "aqlm2"): f"{_QM}/aqlm/mistral7b-v02",  # ISTA-DASLab/...-v0.2-AQLM-2Bit-2x8
    ("mistral-nemo", "aqlm2"): "ISTA-DASLab/Mistral-Nemo-Instruct-2407-AQLM-PV-2Bit-1x16-hf",
    # Phi-3-mini-4k-instruct AQLM-PV 2-bit (fused qkv/gate_up projections)
    ("phi3", "gptq4"): f"{_QM}/gptq/phi3-gptq4bit-g128",   # self-gen (AutoGPTQ)
    ("phi3", "gptq3"): f"{_QM}/gptq/phi3-gptq3bit-g128",
    ("phi3", "gptq2"): f"{_QM}/gptq/phi3-gptq2bit-g128",
    ("phi3", "awq"):  f"{_QM}/awq/phi3",             # Sreenington/Phi-3-mini-4k-instruct-AWQ
    ("phi3", "aqlm2"): f"{_QM}/aqlm/phi3",            # ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf
    # Gemma-2-9B-it
    ("gemma9b", "gptq4"): "ModelCloud/gemma-2-9b-it-gptq-4bit",
}

# fall back to cpu if no gpu is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# how many prompts to use at each stage (SafeQuant v2: n=1000 for both ASR and ARC)
N_TRAIN = 128         # used for direction extraction (Arditi protocol)
N_VAL = 32            # used for direction selection and validation
N_EVAL = 1000         # WildJailbreak prompts for ASR evaluation
N_ARC = 1000          # ARC-Challenge prompts for capability evaluation
MAX_NEW_TOKENS = 256  # cap on generation length during ASR measurement

# bitsandbytes NF4 config - note: float32 compute dtype is required on CPU
# float16 compute fails on CPU because there's no hardware float16 matmul support
NF4_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float32,
    "bnb_4bit_use_double_quant": True,
}
