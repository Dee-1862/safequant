# Model / tokenizer loaders for FP32 and common quantization backends.

import os

# Force gptqmodel to use pure-pytorch backend (no triton/exllama kernel issues)
os.environ["GPTQMODEL_FORCE_BACKEND"] = "torch"
# Disable torch.compile which causes PY_SSIZE_T_CLEAN errors on older torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Front Matter
import torch
import torch.nn as nn
from datasets import load_dataset as _load_ds
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import GPTQConfig

from config import MODELS, DEVICE, HF_HOME, QUANTIZED_CHECKPOINTS

try:
    from config import MODEL_REVISIONS
except ImportError:
    MODEL_REVISIONS = {}

try:
    import optimum.gptq.quantizer as _oq
except ImportError:
    _oq = None

try:
    import bitsandbytes.functional as bnb_F
except ImportError:
    bnb_F = None


def _resolve_model_path(model_name):
    """
    Resolve a model alias to a HuggingFace model path.

    Inputs:
        - model_name (str): Short alias or HF model id

    Outputs:
        - model_path (str): Resolved HF path / id
    """
    return MODELS.get(model_name, model_name)


def _use_remote_code(model_path):
    """
    Decide trust_remote_code; Phi-3 uses transformers-native modeling code.

    Inputs:
        - model_path (str): HF model path / id

    Outputs:
        - use_remote_code (bool): Whether to set trust_remote_code=True
    """
    # Phi-3 remote modeling is incompatible with newer transformers DynamicCache
    p = str(model_path).lower()
    return "phi-3" not in p and "phi3" not in p


def load_fp32(model_name, device = None):
    """
    Load model in FP32 precision.

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    if device is None:
        device = DEVICE
    model_path = _resolve_model_path(model_name)

    print(f"Loading FP32 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # Optional revision pin to match quantized checkpoint provenance
    rev = MODEL_REVISIONS.get(model_name)
    if rev:
        print(
            f"  Revision pin: {rev[:12]} (FP weights matched to the quantizer's source)"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir = HF_HOME, revision = rev
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.float32,
        device_map = device,
        trust_remote_code = _use_remote_code(model_path),
        cache_dir = HF_HOME,
        revision = rev,
    )

    print(
        f"  OK Loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)"
    )
    print(f"  Cache: {HF_HOME}")
    return model, tokenizer


def load_nf4(model_name, device = None):
    """
    Load model in NF4 quantization (bitsandbytes 4-bit NormalFloat).

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    if device is None:
        device = DEVICE
    model_path = _resolve_model_path(model_name)

    print(f"Loading NF4 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    # float32 compute on CPU, float16 on GPU
    compute_dtype = torch.float32 if device == "cpu" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant = True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = bnb_config,
        device_map = device,
        trust_remote_code = True,
        cache_dir = HF_HOME,
    )

    print(f"  OK Loaded in 4-bit NF4 (compute dtype: {compute_dtype})")
    print(f"  Cache: {HF_HOME}")
    return model, tokenizer


def load_int8(model_name, device = None):
    """
    Load model in INT8 quantization (bitsandbytes 8-bit).

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    if device is None:
        device = DEVICE
    model_path = _resolve_model_path(model_name)

    print(f"Loading INT8 model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Device: {device}")

    bnb_config = BitsAndBytesConfig(load_in_8bit = True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = bnb_config,
        device_map = device,
        trust_remote_code = True,
        cache_dir = HF_HOME,
    )

    print(f"  OK Loaded in 8-bit INT8")
    print(f"  Cache: {HF_HOME}")
    return model, tokenizer


def load_quantized(model_name, quantizer = "nf4", device = None):
    """
    Unified loader that dispatches to the right quantizer backend.

    Inputs:
        - model_name (str): Model alias or HF id
        - quantizer (str): Quantizer key (fp32, nf4, int8, awq, gptq*, aqlm*, quip*)
        - device (str | None): Device map target (default: DEVICE)

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    quantizer = quantizer.lower()

    if quantizer == "fp32":
        return load_fp32(model_name, device)
    elif quantizer == "nf4":
        return load_nf4(model_name, device)
    elif quantizer == "int8":
        return load_int8(model_name, device)
    elif quantizer == "awq":
        return _load_awq(model_name, device)
    elif quantizer in ("gptq", "gptq4"):
        return _load_gptq(model_name, device, bits = 4)
    elif quantizer == "gptq3":
        return _load_gptq(model_name, device, bits = 3)
    elif quantizer == "gptq2":
        return _load_gptq(model_name, device, bits = 2)
    elif quantizer == "aqlm2":
        return _load_aqlm(model_name, device, bits = 2)
    elif quantizer == "aqlm3":
        return _load_aqlm(model_name, device, bits = 3)
    elif quantizer in ("quip2", "quip4"):
        bits = 2 if quantizer == "quip2" else 4
        return _load_quip(model_name, device, bits = bits)
    else:
        raise ValueError(
            f"Unknown quantizer: '{quantizer}'. "
            f"Choose from: fp32, nf4, int8, awq, gptq4, gptq3, gptq2, aqlm2, aqlm3, quip2, quip4"
        )


def _load_awq(model_name, device = None):
    """
    Load a pre-quantized AWQ model via transformers native AWQ support.

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    if device is None:
        device = DEVICE

    # Prefer an entry in QUANTIZED_CHECKPOINTS when registered
    ckpt_key = (model_name, "awq")
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AWQ model: {model_name} -> {model_path}")
    else:
        model_path = _resolve_model_path(model_name)
        print(f"Loading AWQ model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir = HF_HOME,
        trust_remote_code = _use_remote_code(model_path)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device,
        trust_remote_code = _use_remote_code(model_path),
        cache_dir = HF_HOME,
        torch_dtype = torch.float16,
    )
    model.eval()
    print(f"  OK Loaded AWQ 4-bit model")
    return model, tokenizer


def _load_aqlm(model_name, device = None, bits = 2):
    """
    Load an AQLM-quantized model (SOTA for 2-bit).

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)
        - bits (int): AQLM bitwidth

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer (from base model)
    """
    if device is None:
        device = DEVICE

    # Resolve checkpoint: registry, then local aqlm_cache, else error
    bit_suffix = f"aqlm{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading AQLM-{bits}bit model: {model_name} -> {model_path}")
    else:
        safe_name = model_name.replace("/", "_")
        local_path = os.path.join(
            HF_HOME, "aqlm_cache", f"{safe_name}-aqlm-{bits}bit"
        )
        if os.path.isdir(local_path) and os.path.exists(
            os.path.join(local_path, "config.json")
        ):
            model_path = local_path
            print(
                f"Loading AQLM-{bits}bit model: {model_name} (cached at {local_path})"
            )
        else:
            raise ValueError(
                f"No pre-quantized AQLM checkpoint for ({model_name}, {bit_suffix}). "
                f"AQLM requires pre-quantized checkpoints -- on-the-fly quantization "
                f"takes hours. Add the checkpoint to config.QUANTIZED_CHECKPOINTS."
            )

    # Tokenizer from base model (AQLM repos often lack chat templates)
    base_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        cache_dir = HF_HOME,
        trust_remote_code = _use_remote_code(base_path),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device,
        trust_remote_code = _use_remote_code(model_path),
        cache_dir = HF_HOME,
        torch_dtype = torch.float16,
    )
    model.eval()
    print(f"  OK Loaded AQLM {bits}-bit model")
    return model, tokenizer


def _load_quip(model_name, device = None, bits = 4):
    """
    Load a pre-quantized QuIP# model.

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)
        - bits (int): QuIP# bitwidth

    Outputs:
        - model (torch.nn.Module): Loaded model
        - tokenizer (Tokenizer): Matching tokenizer (from base model)
    """
    if device is None:
        device = DEVICE

    bit_suffix = f"quip{bits}"
    ckpt_key = (model_name, bit_suffix)
    if ckpt_key in QUANTIZED_CHECKPOINTS:
        model_path = QUANTIZED_CHECKPOINTS[ckpt_key]
        print(f"Loading QuIP#-{bits}bit model: {model_name} -> {model_path}")
    else:
        raise ValueError(
            f"No pre-quantized QuIP# checkpoint for ({model_name}, {bit_suffix})"
        )

    # Tokenizer from base model (QuIP# repos don't include tokenizer)
    base_path = _resolve_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, cache_dir = HF_HOME, trust_remote_code = True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device,
        trust_remote_code = True,
        cache_dir = HF_HOME,
        torch_dtype = torch.float16,
    )
    model.eval()
    print(f"  OK Loaded QuIP# {bits}-bit model")
    return model, tokenizer


def _load_gptq(model_name, device = None, bits = 4):
    """
    Load a GPTQ-quantized model (pre-quantized or on-the-fly).

    Inputs:
        - model_name (str): Model alias or HF id
        - device (str | None): Device map target (default: DEVICE)
        - bits (int): Quantization bits (2, 3, 4, or 8)

    Outputs:
        - model (torch.nn.Module): Loaded / quantized model
        - tokenizer (Tokenizer): Matching tokenizer
    """
    if device is None:
        device = DEVICE

    # --- optimum 2.2.0 / gptqmodel 7.1.0 load shim (act-order checkpoints) ---
    # optimum's GPTQQuantizer defaults act_group_aware=True and passes it
    # EXPLICITLY into gptqmodel's QuantizeConfig, which rejects act_group_aware=True
    # together with desc_act=True -- i.e. every standard act-order GPTQ checkpoint.
    # transformers' GPTQConfig.to_dict_optimum() drops the field, so it cannot be
    # overridden through the config. Force it False here. act_group_aware is a
    # quantization-TIME knob (it controls how groups are ORDERED while quantizing);
    # the already-quantized weights load identically, so this changes no results.
    # Idempotent; load-path only.
    if _oq is not None and not getattr(_oq.GPTQQuantizer, "_safequant_agaw_patch", False):
        _orig_gptq_init = _oq.GPTQQuantizer.__init__

        def _agaw_init(self, *a, **kw):
            kw["act_group_aware"] = False
            _orig_gptq_init(self, *a, **kw)

        _oq.GPTQQuantizer.__init__ = _agaw_init
        _oq.GPTQQuantizer._safequant_agaw_patch = True

    # Look up a registered pre-quantized checkpoint when bits map to a suffix
    bit_suffix = {4: "gptq4", 3: "gptq3", 2: "gptq2", 8: "gptq8"}.get(bits)
    ckpt_key_alias = (model_name, bit_suffix) if bit_suffix else None
    pre_quantized = None
    if ckpt_key_alias and ckpt_key_alias in QUANTIZED_CHECKPOINTS:
        pre_quantized = QUANTIZED_CHECKPOINTS[ckpt_key_alias]

    if pre_quantized:
        # Decoding Compressed Trust models use revision branches (e.g. "4bit_128g")
        revision = None
        if "compressed-llm" in pre_quantized:
            revision = f"{bits}bit_128g"
            print(
                f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized} "
                f"(rev: {revision})"
            )
        else:
            print(f"Loading GPTQ-{bits}bit model: {model_name} -> {pre_quantized}")

        # Tokenizer: base model for compressed-llm, else from the checkpoint
        if "compressed-llm" in pre_quantized:
            base_path = _resolve_model_path(model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                base_path, cache_dir = HF_HOME, trust_remote_code = True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pre_quantized,
                cache_dir = HF_HOME,
                trust_remote_code = _use_remote_code(pre_quantized),
                revision = revision,
            )

        gptq_cfg = GPTQConfig(bits = bits, use_exllama = False)
        model = AutoModelForCausalLM.from_pretrained(
            pre_quantized,
            quantization_config = gptq_cfg,
            device_map = device,
            trust_remote_code = _use_remote_code(pre_quantized),
            cache_dir = HF_HOME,
            torch_dtype = torch.float16,
            revision = revision,
        )
        model.eval()
        print(f"  OK Loaded GPTQ {bits}-bit (pre-quantized)")
        return model, tokenizer

    # Cached on-the-fly result, or quantize from scratch
    model_path = _resolve_model_path(model_name)
    safe_name = model_name.replace("/", "_")
    cache_path = os.path.join(HF_HOME, "gptq_cache", f"{safe_name}-gptq-{bits}bit")

    if os.path.isdir(cache_path) and os.path.exists(
        os.path.join(cache_path, "config.json")
    ):
        print(f"Loading GPTQ-{bits}bit model: {model_name} (cached at {cache_path})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cache_path, trust_remote_code = True
            )
        except Exception as e:
            print(
                f"  cache-path tokenizer load failed ({type(e).__name__}); "
                f"loading from base HF repo {model_path}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code = True
            )
        gptq_cfg = GPTQConfig(bits = bits, use_exllama = False)
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            quantization_config = gptq_cfg,
            device_map = device,
            trust_remote_code = True,
            torch_dtype = torch.float16,
        )
        model.eval()
        print(f"  OK Loaded GPTQ {bits}-bit from cache")
        return model, tokenizer

    # Quantize from scratch with wikitext calibration strings
    print(f"Loading GPTQ-{bits}bit model: {model_name} (on-the-fly quantization)")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir = HF_HOME, trust_remote_code = True
    )
    try:
        _wiki = _load_ds(
            "wikitext", "wikitext-2-raw-v1", split = "train", cache_dir = HF_HOME
        )
        calib_texts = [t for t in _wiki["text"] if len(t.strip()) > 100][:128]
    except Exception:
        _wiki = _load_ds("wikitext", "wikitext-2-raw-v1", split = "train")
        calib_texts = [t for t in _wiki["text"] if len(t.strip()) > 100][:128]

    gptq_config = GPTQConfig(
        bits = bits,
        dataset = calib_texts,
        tokenizer = tokenizer,
        group_size = 128,
        use_exllama = False,
    )
    print(f"  Quantizing to {bits}-bit GPTQ (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = gptq_config,
        device_map = device,
        trust_remote_code = True,
        cache_dir = HF_HOME,
    )
    model.eval()
    print(f"  OK Quantized to GPTQ {bits}-bit on-the-fly")

    # Best-effort local cache for future runs
    try:
        os.makedirs(cache_path, exist_ok = True)
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path)
        print(f"  OK Saved to cache: {cache_path}")
    except Exception as e:
        print(f"  Warning: could not cache quantized model: {e}")

    return model, tokenizer


def get_weight(module, as_dtype = torch.float32):
    """
    Extract weight from a module, handling regular and quantized layouts.

    Inputs:
        - module (nn.Module): Linear or quantized linear module
        - as_dtype (torch.dtype): Target dtype for the returned weight

    Outputs:
        - weight (torch.Tensor): Dense (out_features, in_features) weight
    """
    # Dimension attrs differ across quantizer wrappers
    in_f = getattr(module, "in_features", None) or getattr(module, "infeatures", None)
    out_f = getattr(module, "out_features", None) or getattr(
        module, "outfeatures", None
    )

    # auto_gptq QuantLinear -- dequantize via identity forward
    if hasattr(module, "qweight"):
        device = module.qweight.device
        identity = torch.eye(in_f, device = device, dtype = torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Unknown non-Linear module -- identity forward fall-back
    if not isinstance(module, nn.Linear):
        device = next(module.parameters()).device
        identity = torch.eye(in_f, device = device, dtype = torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # BitsAndBytes Params4bit (NF4)
    if hasattr(module.weight, "quant_type"):
        try:
            if bnb_F is None:
                raise ImportError("bitsandbytes is not installed")
            weight = (
                bnb_F.dequantize_4bit(
                    module.weight.data,
                    module.weight.quant_state,
                )
                .to(torch.float32)
                .view(out_f, in_f)
            )
        except Exception:
            identity = torch.eye(
                in_f, device = module.weight.device, dtype = torch.float32
            )
            with torch.no_grad():
                weight = module(identity).T
        return weight.to(as_dtype)

    # BitsAndBytes Int8Params (LLM.int8)
    if hasattr(module.weight, "CB") or module.weight.dtype == torch.int8:
        device = module.weight.device
        identity = torch.eye(in_f, device = device, dtype = torch.float16)
        with torch.no_grad():
            weight = module(identity).T.float()
        return weight.to(as_dtype)

    # Dense nn.Linear
    return module.weight.data.to(as_dtype)
