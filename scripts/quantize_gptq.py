# Reproducible GPTQ quantization via AutoGPTQ (Bridges-2 helper).

# Front Matter
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

try:
    from auto_gptq.modeling._base import BaseGPTQForCausalLM
    from auto_gptq.modeling import _const as _agc, auto as _aga
except ImportError:
    BaseGPTQForCausalLM = None
    _agc = _aga = None

# Repo root must be on path when invoked as scripts/quantize_gptq.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, MODEL_REVISIONS, HF_HOME, QM_ROOT  # noqa: E402
from core.datasets import load_dataset


def _register_phi3_autogptq():
    """Register Phi-3 fused layout for auto-gptq 0.7.1 (no-op on other archs)."""
    if BaseGPTQForCausalLM is None:
        return
    try:
        if "phi3" not in _agc.SUPPORTED_MODELS:
            class Phi3GPTQForCausalLM(BaseGPTQForCausalLM):
                layer_type = "Phi3DecoderLayer"
                layers_block_name = "model.layers"
                outside_layer_modules = ["model.embed_tokens", "model.norm"]
                inside_layer_modules = [
                    ["self_attn.qkv_proj"],
                    ["self_attn.o_proj"],
                    ["mlp.gate_up_proj"],
                    ["mlp.down_proj"],
                ]
            _agc.SUPPORTED_MODELS.append("phi3")
            _aga.GPTQ_CAUSAL_LM_MODEL_MAP["phi3"] = Phi3GPTQForCausalLM
            print("  registered Phi-3 support in auto-gptq (fused qkv/gate_up)")
    except Exception as _e:
        print(f"  WARN: phi3 auto-gptq registration failed: {_e}")


def default_output_dir(model, bits, group_size):
    """
    Ocean path for a self-quantized GPTQ checkpoint (never under $HOME).

    Inputs:
        - model (str): Model alias from config.MODELS
        - bits (int): Quantization bit width
        - group_size (int): GPTQ group size

    Outputs:
        - out_dir (Path): Target directory for saved weights
    """
    return Path(QM_ROOT) / "gptq" / f"{model}-gptq{bits}bit-g{group_size}"


def build_calibration(tokenizer, n_samples, seqlen, seed):
    """
    WikiText-2 calibration windows for AutoGPTQ (hugging-quants recipe).

    Inputs:
        - tokenizer: HuggingFace tokenizer
        - n_samples (int): Number of windows
        - seqlen (int): Tokens per window
        - seed (int): RNG seed for window starts

    Outputs:
        - samples (list): Dicts with input_ids and attention_mask
    """
    # Load pinned WikiText-2-raw train split
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "train")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors = "pt")

    # Pin RNG before drawing window offsets
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    n_tokens = enc.input_ids.shape[1]
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, n_tokens - seqlen - 1)
        ids = enc.input_ids[:, i:i + seqlen]
        samples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return samples


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser(description = "Reproducible GPTQ quantization via AutoGPTQ.")
    ap.add_argument("--model", required = True,
                    help = "alias from config.MODELS (llama7b, ...) or a raw HF id")
    ap.add_argument("--bits", type = int, default = 4, choices = [2, 3, 4, 8])
    ap.add_argument("--group_size", type = int, default = 128)
    ap.add_argument("--n_calib", type = int, default = 128,
                    help = "calibration windows (GPTQ standard = 128)")
    ap.add_argument("--seqlen", type = int, default = 2048,
                    help = "tokens per calibration window (GPTQ standard = 2048)")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--output_dir", default = None)
    args = ap.parse_args()

    # Resolve HF id and optional revision pin
    model_path = MODELS.get(args.model, args.model)
    revision = MODEL_REVISIONS.get(args.model)

    # Pick output directory on Ocean
    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.model, args.bits, args.group_size))
    out_dir.parent.mkdir(parents = True, exist_ok = True)

    print("=" * 72)
    print(f"  AutoGPTQ quantize: {args.model}  ({model_path})")
    print(f"  bits={args.bits}  group_size={args.group_size}  "
          f"n_calib={args.n_calib}  seqlen={args.seqlen}  seed={args.seed}")
    if revision:
        print(f"  FP-base revision pin: {revision[:12]}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    tok_kwargs = {"use_fast": True, "cache_dir": HF_HOME}
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)

    # Build calibration windows
    print(f"\n[1/3] building {args.n_calib} x {args.seqlen}-token calibration "
          f"from WikiText-2 ...")
    calib = build_calibration(tokenizer, args.n_calib, args.seqlen, args.seed)
    print(f"  built {len(calib)} calibration windows")

    # Load FP base and quantize
    print("\n[2/3] loading FP base + quantizing (AutoGPTQ) ...")
    quantize_config = BaseQuantizeConfig(
        bits = args.bits,
        group_size = args.group_size,
        desc_act = True,
        sym = True,
        damp_percent = 0.1,
    )
    init_kwargs = {
        "cache_dir": HF_HOME,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if revision:
        init_kwargs["revision"] = revision

    _register_phi3_autogptq()

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, **init_kwargs)

    # Pin RotaryEmbedding onto position_ids device (Phi-3 / long-context stability)
    def _pin_rotary(mod, args_hook):
        try:
            dev = args_hook[1].device if len(args_hook) > 1 else args_hook[0].device
            mod.to(dev)
        except Exception:
            pass

    for _name, _mod in model.model.named_modules():
        if _mod.__class__.__name__.endswith("RotaryEmbedding"):
            _mod.register_forward_pre_hook(_pin_rotary)

    model.quantize(calib)

    # Save quantized weights + tokenizer
    print(f"\n[3/3] saving to {out_dir} ...")
    model.save_quantized(str(out_dir), use_safetensors = True)
    tokenizer.save_pretrained(str(out_dir))

    print("\nDONE. Registered in config.QUANTIZED_CHECKPOINTS as:")
    print(f'    ("{args.model}", "gptq{args.bits}"): "{out_dir}",')


if __name__ == "__main__":
    main()
