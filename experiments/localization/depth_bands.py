# Table 4: depth-band restoration with --projections read|write|all.
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


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--projections", choices = ["read", "write", "all"], default = "all")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--n_reasoning", type = int, default = 50)
    ap.add_argument("--n_layers", type = int, default = None)
    ap.add_argument("--early_end", type = int, default = None)
    ap.add_argument("--mid_end", type = int, default = None)
    args = ap.parse_args()

    # Resolve quantizer label and device
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load FP32 to discover projection sets for this architecture
    fp32_model, tokenizer = load_fp32(args.model)
    read_projs, write_projs = get_proj_sets(fp32_model)
    all_projs = read_projs + write_projs
    proj_set = {"read": read_projs, "write": write_projs, "all": all_projs}[args.projections]
    variant = None if args.projections == "all" else args.projections

    # Load eval prompts and behaviors
    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples = args.n_reasoning)

    # Define early/middle/late depth bands
    all_responses, all_arc = {}, {}
    n_layers = args.n_layers or fp32_model.config.num_hidden_layers
    early_end = args.early_end or (n_layers // 3)
    mid_end = args.mid_end or (2 * n_layers // 3)
    ranges = {"early": (0, early_end), "middle": (early_end, mid_end), "late": (mid_end, n_layers)}

    # Cache FP32 weights for the selected projection set
    fp32_w = extract_fp32_weights(fp32_model, proj_set)

    # FP32 baseline generation and ARC
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

    # Restore the projection set within each depth band
    for name, lo_hi in ranges.items():
        cond = f"layers_{name}"
        q_model, _ = load_quantized(args.model, quantizer = qtz)
        restore_projections(q_model, fp32_w, proj_set, layer_range = lo_hi)
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
    fp32_asr, q_asr = asr_results["fp32"], asr_results[qtz]
    damage = q_asr - fp32_asr

    # Persist results under the localization mirror path
    cfg = RunConfig(
        model = args.model,
        quantizer = qtz,
        variant = variant,
        extra = {
            "n_eval": args.n_eval,
            "projections": args.projections,
            "ranges": {k: list(v) for k, v in ranges.items()},
            "seed": SEED,
        },
    )
    data = {
        "n_layers": n_layers,
        "fp32_asr": fp32_asr,
        "quantized_asr": q_asr,
        "damage_pp": damage,
        "asr_per_condition": asr_results,
        "arc_per_condition": all_arc,
        "recovery_pp_per_range": {
            name: q_asr - asr_results[f"layers_{name}"] for name in ranges
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
