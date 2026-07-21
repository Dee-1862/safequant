# Parameter-matched read restoration control (volume vs role).
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.datasets import load_arc_challenge, load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch, generate_only
from core.manifest import RunConfig, save_result
from core.model_loader import get_weight, load_fp32, load_quantized
from core.residuals import get_inner
from core.restoration import (
    Restorer, _get_submodule, _set_submodule, extract_fp32_weights,
    get_proj_sets, measure_arc, restore_projections,
)
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def proj_params(weights, projs):
    """Total parameter count over all layers for the given projection paths."""
    total = 0
    for L in weights:
        for p in projs:
            out, inf = weights[L][p].shape
            total += out * inf
    return total


def select_whole_subset(weights, read_projs, budget):
    """Greedily pick whole read projections whose total params ~= write budget."""
    n_layers = len(weights)
    L0 = next(iter(weights))
    per = {p: weights[L0][p].shape[0] * weights[L0][p].shape[1] * n_layers
           for p in read_projs}
    chosen, total = [], 0
    for p in sorted(per, key=per.get, reverse=True):
        if total + per[p] <= budget * 1.10: # allow up to 10% over the budget
            chosen.append(p)
            total += per[p]
    return chosen, total


def restore_read_rows(model, fp32_read_weights, read_projs, frac, rng):
    """Restore a random `frac` of each read projection's output rows to FP32,
    leaving the remaining rows at their quantized (dequantized) values."""
    inner = get_inner(model)
    device = next(model.parameters()).device
    act_dtype = inner.norm.weight.dtype
    for L in range(model.config.num_hidden_layers):
        layer = inner.layers[L]
        for p in read_projs:
            orig = _get_submodule(layer, p)
            base = get_weight(orig, as_dtype=torch.float32).cpu()  # dequant [out,in]
            fp = fp32_read_weights[L][p]                           # FP32   [out,in]
            out, in_f = base.shape
            k = int(round(frac * out))
            idx = rng.permutation(out)[:k]
            new_w = base.clone()
            new_w[idx] = fp[idx]
            has_bias = orig.bias is not None
            lin = nn.Linear(in_f, out, bias=has_bias).to(device).to(act_dtype)
            lin.weight = nn.Parameter(new_w.to(device).to(act_dtype),
                                      requires_grad=False)
            if has_bias:
                lin.bias = nn.Parameter(orig.bias.data.clone(), requires_grad=False)
            _set_submodule(layer, p, lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_reasoning", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    # FP32 baseline; cache read/write weights and compute the param budget.
    fp32_model, tokenizer = load_fp32(args.model)
    read_projs, write_projs = get_proj_sets(fp32_model)
    read_w = extract_fp32_weights(fp32_model, read_projs)
    write_budget = proj_params(extract_fp32_weights(fp32_model, write_projs),
                               write_projs)
    read_total = proj_params(read_w, read_projs)
    frac = write_budget / read_total
    whole_subset, whole_total = select_whole_subset(read_w, read_projs, write_budget)
    restorer = Restorer(fp32_model)  # full-write reference
    print(f"write_budget={write_budget:,}  read_total={read_total:,}  "
          f"frac={frac:.3f}  whole_subset={whole_subset} ({whole_total:,})")

    all_resp, all_arc = {}, {}
    all_resp["fp32"] = generate_only(fp32_model, tokenizer, items, desc="fp32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    def eval_condition(name, restore_fn):
        m, _ = load_quantized(args.model, quantizer=args.quantizer)
        if restore_fn is not None:
            restore_fn(m)
        all_resp[name] = generate_only(m, tokenizer, items, desc=name)
        all_arc[name] = measure_arc(m, tokenizer, arc_items)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    eval_condition("quant", None)
    eval_condition("write_full", restorer.restore_write)                    # reference
    eval_condition("read_matched_whole",
                   lambda m: restore_projections(m, read_w, whole_subset))
    for s in range(args.seeds):
        rng = np.random.default_rng(SEED + s)
        eval_condition(f"read_matched_random_s{s}",
                       lambda m, rng=rng: restore_read_rows(m, read_w, read_projs,
                                                            frac, rng))

    # Classify every condition in one HarmBench session for comparability.
    conds = list(all_resp.keys())
    flatb, flatr = [], []
    for c in conds:
        flatb.extend(behaviors)
        flatr.extend(all_resp[c])
    flags = classify_batch(flatb, flatr, device=device)
    n = args.n_eval
    asr = {c: 100.0 * sum(flags[i * n:(i + 1) * n]) / n for i, c in enumerate(conds)}
    rand = [asr[f"read_matched_random_s{s}"] for s in range(args.seeds)]

    # recovery = how much of the damage each restore removes (quant_asr - asr).
    data = {
        "param_budget": {
            "write_params": write_budget, "read_params_total": read_total,
            "match_fraction": frac, "whole_subset": whole_subset,
            "whole_subset_params": whole_total,
        },
        "asr_per_condition": asr,
        "arc_per_condition": all_arc,
        "read_matched_random_asr_mean": float(np.mean(rand)),
        "read_matched_random_asr_std": float(np.std(rand)),
        "recovery_pp": {
            "write_full": asr["quant"] - asr["write_full"],
            "read_matched_whole": asr["quant"] - asr["read_matched_whole"],
            "read_matched_random_mean": asr["quant"] - float(np.mean(rand)),
        },
    }
    cfg = RunConfig(
        model=args.model, quantizer=args.quantizer,
        extra={"n_eval": args.n_eval, "n_reasoning": args.n_reasoning,
               "seeds": args.seeds, "seed": SEED},
    )
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
