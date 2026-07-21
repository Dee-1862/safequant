# Table 2: 14-cell FP/quant/+W/+R/+Both restoration matrix.
import argparse
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch, generate_only
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.restoration import Restorer, measure_arc
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_SCRIPT = Path(__file__)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--quantizer", default = "aqlm2")
    parser.add_argument("--n_eval", type = int, default = 1000)
    parser.add_argument("--n_reasoning", type = int, default = 50)
    args = parser.parse_args()

    # Resolve quantizer label and device
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load eval prompts and behaviors
    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = [_extract_behavior(item) for item in items]
    arc_items = load_arc_challenge(n_samples = args.n_reasoning)

    # FP32 baseline and Restorer weight cache
    all_responses, all_arc = {}, {}
    fp32_model, tokenizer = load_fp32(args.model)
    fp32_model_type = getattr(fp32_model.config, "model_type", None)
    restorer = Restorer(fp32_model)
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc = "FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Quantized baseline
    q_model, _ = load_quantized(args.model, quantizer = qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc = qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Write / read / both restored cells. If the quantized checkpoint's
    # architecture does not match the FP32 model -- e.g. an "unfused"
    # Mistral-layout requantization of fused Phi-3 (Sreenington's Phi-3 AWQ is
    # model_type=mistral with separate q/k/v vs Phi-3's fused qkv_proj) -- its
    # projection modules don't correspond to the FP32 ones, so restoration is
    # structurally undefined. Catch that and record N/A rather than crash; the
    # FP32 and quant baselines above stay valid.
    restore_status = "ok"
    restore_note = None
    try:
        for cond, restore_fn in [
            ("write_restored", "restore_write"),
            ("read_restored", "restore_read"),
            ("both_restored", "restore_both"),
        ]:
            q_model, _ = load_quantized(args.model, quantizer = qtz)
            getattr(restorer, restore_fn)(q_model)
            all_responses[cond] = generate_only(q_model, tokenizer, items, desc = cond)
            all_arc[cond] = measure_arc(q_model, tokenizer, arc_items)
            del q_model
            gc.collect()
            torch.cuda.empty_cache()
    except (AttributeError, KeyError) as e:
        restore_status = "arch_mismatch_na"
        restore_note = (
            f"restoration undefined: {qtz} checkpoint architecture does not "
            f"expose the FP32 ({fp32_model_type}) projection modules "
            f"({type(e).__name__}: {e})"
        )
        for cond in ("write_restored", "read_restored", "both_restored"):
            all_responses.pop(cond, None)
            all_arc.pop(cond, None)
        try:
            del q_model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  NOTE {restore_status}: {restore_note}")

    # Classify all conditions in one HarmBench session
    conditions = list(all_responses.keys())
    flat_behaviors, flat_responses = [], []
    for cond in conditions:
        flat_behaviors.extend(behaviors)
        flat_responses.extend(all_responses[cond])
    flat_flags = classify_batch(flat_behaviors, flat_responses, device = device)
    n = args.n_eval
    asr_results = {}
    for i, cond in enumerate(conditions):
        flags = flat_flags[i * n : (i + 1) * n]
        asr_results[cond] = 100.0 * sum(flags) / n

    # Swap-validity gate against FP32 with ARC floor
    fp32_asr = asr_results["fp32"]
    q_asr = asr_results[qtz]
    damage = q_asr - fp32_asr
    if restore_status == "ok":
        both_asr = asr_results["both_restored"]
        swap_tol = max(8.0, 2 * math.sqrt(2 * 0.25 / max(args.n_eval, 1)) * 100)
        swap_delta = both_asr - fp32_asr
        arc_floor_ok = (
            min(all_arc["write_restored"], all_arc["read_restored"],
                all_arc["both_restored"])
            >= all_arc[qtz] - 7.0
        )
        swap_valid = abs(swap_delta) <= swap_tol and arc_floor_ok
    else:
        both_asr = swap_tol = swap_delta = None
        swap_valid = False

    # Persist the restoration matrix
    cfg = RunConfig(
        model = args.model,
        quantizer = qtz,
        extra = {"n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED},
    )
    data = {
        "fp32_asr": fp32_asr,
        "quantized_asr": q_asr,
        "damage_pp": damage,
        "write_restored_asr": asr_results.get("write_restored"),
        "read_restored_asr": asr_results.get("read_restored"),
        "both_restored_asr": both_asr,
        "write_recovery_pp": (q_asr - asr_results["write_restored"]
                              if restore_status == "ok" else None),
        "read_recovery_pp": (q_asr - asr_results["read_restored"]
                             if restore_status == "ok" else None),
        "swap_valid": swap_valid,
        "swap_delta_pp": swap_delta,
        "swap_tol_pp": swap_tol,
        "arc": all_arc,
        "asr_per_condition": asr_results,
        "restore_status": restore_status,
        "restore_note": restore_note,
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
