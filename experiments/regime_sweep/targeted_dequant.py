# Targeted mixed-precision repair: restore AQLM read-side from AWQ-4 source.
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
    get_proj_sets,
    measure_arc,
    restore_projections,
)
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _role_projs(model, role):
    """Read / write / both projection dotted-paths for this architecture."""
    read, write = get_proj_sets(model)
    if role == "read":
        return read
    if role == "write":
        return write
    return read + write


def main():
    ap = argparse.ArgumentParser(
        description="Targeted 4-bit (AWQ) restoration into the 2-bit (AQLM) cell.")
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2", help="damaged cell to repair")
    ap.add_argument("--source_quant", default="awq",
                    help="4-bit source of restored weights (dequantized)")
    ap.add_argument("--role", default="read", choices=["read", "write", "both"])
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_reasoning", type=int, default=50)
    ap.add_argument("--layer_lo", type=int, default=None)
    ap.add_argument("--layer_hi", type=int, default=None)
    args = ap.parse_args()

    qtz, src = args.quantizer, args.source_quant
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_range, band = None, "all"
    if args.layer_lo is not None and args.layer_hi is not None:
        layer_range = (args.layer_lo, args.layer_hi)
        band = f"L{args.layer_lo}-{args.layer_hi}"
    variant = f"{args.role}_{src}4_{band}"   # e.g. read_awq4_all / read_awq4_L11-22

    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    all_responses, all_arc = {}, {}

    # FP32 reference + upper-bound source weights
    fp32_model, tokenizer = load_fp32(args.model)
    projs = _role_projs(fp32_model, args.role)
    fp32_weights = extract_fp32_weights(fp32_model, projs)
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # 4-bit source: dequantized AWQ weights for the same projections
    src_model, _ = load_quantized(args.model, quantizer=src)
    src_weights = extract_fp32_weights(src_model, projs)
    del src_model
    gc.collect()
    torch.cuda.empty_cache()

    # 2-bit baseline (damaged, no restore)
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # + targeted 4-bit restore (the tested repair) and + FP32 restore (upper bound)
    for tag, weights in ((f"{src}4_restore", src_weights), ("fp32_restore", fp32_weights)):
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        restore_projections(q_model, weights, projs, layer_range=layer_range)
        all_responses[tag] = generate_only(q_model, tokenizer, items, desc=tag)
        all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
        del q_model
        gc.collect()
        torch.cuda.empty_cache()

    # Classify all conditions in one HarmBench session
    conditions = list(all_responses.keys())
    flat_b, flat_r = [], []
    for c in conditions:
        flat_b.extend(behaviors)
        flat_r.extend(all_responses[c])
    flags = classify_batch(flat_b, flat_r, device=device)
    n = args.n_eval
    asr = {c: 100.0 * sum(flags[i * n:(i + 1) * n]) / n for i, c in enumerate(conditions)}

    q_asr = asr[qtz]
    rec_4bit = q_asr - asr[f"{src}4_restore"]
    rec_fp32 = q_asr - asr["fp32_restore"]
    frac = (rec_4bit / rec_fp32) if abs(rec_fp32) > 1e-9 else None

    cfg = RunConfig(
        model=args.model, quantizer=qtz, variant=variant,
        extra={"source_quant": src, "role": args.role, "band": band,
               "n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED},
    )
    data = {
        "fp32_asr": asr["fp32"], "quantized_asr": q_asr,
        "damage_pp": q_asr - asr["fp32"],
        "role": args.role, "layer_range": list(layer_range) if layer_range else None,
        "asr_per_condition": asr, "arc_per_condition": all_arc,
        "recovery_pp_4bit": rec_4bit, "recovery_pp_fp32": rec_fp32,
        "frac_of_fp32_recovery": frac,
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")
    print(f"  damage={data['damage_pp']:.1f}pp  4bit_recovery={rec_4bit:.1f}pp  "
          f"fp32_recovery={rec_fp32:.1f}pp  frac_of_fp32={frac}")


if __name__ == "__main__":
    main()
