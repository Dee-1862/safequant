# Tables 3-4: per-projection restoration sweep.
import argparse
import gc
from pathlib import Path

import torch

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch, generate_only
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.restoration import (
    extract_fp32_weights,
    measure_arc,
    restore_projections,
)
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)

PROJ_TYPES_LLama = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PROJ_TYPES_PHI3 = ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]


def _proj_types(model):
    """
    Return the projection-type paths for this model architecture.

    Inputs:
        - model (nn.Module): Loaded model (Llama-style or Phi-3)

    Outputs:
        - proj_types (list[str]): Projection paths to restore one at a time
    """
    # Detect Phi-3 fused qkv vs Llama-style separate projections
    submods = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if "qkv_proj" in submods:
        return PROJ_TYPES_PHI3
    return PROJ_TYPES_LLama


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--n_reasoning", type = int, default = 50)
    ap.add_argument("--layer_lo", type = int, default = None)
    ap.add_argument("--layer_hi", type = int, default = None)
    args = ap.parse_args()

    # Resolve runtime device and optional middle-layer band
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_range = None
    variant = None
    if args.layer_lo is not None and args.layer_hi is not None:
        layer_range = (args.layer_lo, args.layer_hi)
        variant = f"L{args.layer_lo}-{args.layer_hi}"

    # Load eval prompts and extract HarmBench behaviors
    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples = args.n_reasoning)

    # FP32: cache weights per projection, then measure baseline ASR/ARC
    all_responses, all_arc = {}, {}
    fp32_model, tokenizer = load_fp32(args.model)
    proj_types = _proj_types(fp32_model)
    fp32_weights_by_proj = {p: extract_fp32_weights(fp32_model, [p]) for p in proj_types}
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc = "FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Quantized baseline without restoration
    q_model, _ = load_quantized(args.model, quantizer = qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc = qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # One-at-a-time projection restore conditions
    for proj in proj_types:
        suffix = "_middle" if layer_range else ""
        cond = f"only_{proj.replace('.', '_')}{suffix}"
        q_model, _ = load_quantized(args.model, quantizer = qtz)
        if layer_range:
            restore_projections(q_model, fp32_weights_by_proj[proj], [proj], layer_range = layer_range)
        else:
            restore_projections(q_model, fp32_weights_by_proj[proj], [proj])
        all_responses[cond] = generate_only(q_model, tokenizer, items, desc = cond)
        all_arc[cond] = measure_arc(q_model, tokenizer, arc_items)
        del q_model
        gc.collect()
        torch.cuda.empty_cache()

    # Classify all conditions in one HarmBench session
    conditions = list(all_responses.keys())
    flat_b, flat_r = [], []
    for cond in conditions:
        flat_b.extend(behaviors)
        flat_r.extend(all_responses[cond])
    flags = classify_batch(flat_b, flat_r, device = device)
    n = args.n_eval
    asr_results = {cond: 100.0 * sum(flags[i * n : (i + 1) * n]) / n for i, cond in enumerate(conditions)}

    # Damage relative to FP32 and per-projection recovery
    fp32_asr = asr_results["fp32"]
    q_asr = asr_results[qtz]
    damage = q_asr - fp32_asr

    # Persist results under the localization mirror path
    cfg = RunConfig(
        model = args.model,
        quantizer = qtz,
        variant = variant,
        extra = {"n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED},
    )
    data = {
        "fp32_asr": fp32_asr,
        "quantized_asr": q_asr,
        "damage_pp": damage,
        "asr_per_condition": asr_results,
        "arc_per_condition": all_arc,
        "layer_range": list(layer_range) if layer_range else None,
        "recovery_pp_per_projection": {
            p: q_asr - asr_results[f"only_{p.replace('.', '_')}{'_middle' if layer_range else ''}"]
            for p in proj_types
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
