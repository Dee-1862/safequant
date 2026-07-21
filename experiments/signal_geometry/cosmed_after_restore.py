# Mediation test: cosmed after read/write/both restoration vs FP32 direction.
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.directions import compute_r_hat, cos_sim
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.residuals import default_token_pos, extract_last_token_residuals
from core.restoration import Restorer
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def directions(model, tokenizer, harmful, harmless, layers, device, token_pos):
    """
    Per-layer unit diff-in-means refusal direction for one model state.

    Inputs:
        - model (nn.Module): Model under test
        - tokenizer: HuggingFace tokenizer
        - harmful (list): Harmful calibration prompts
        - harmless (list): Harmless calibration prompts
        - layers (list[int]): Layer indices to extract
        - device (str): Torch device
        - token_pos (int): Residual token position

    Outputs:
        - vecs (dict): Layer index -> unit direction tensor
    """
    h = extract_last_token_residuals(
        model, tokenizer, harmful, layers, device, token_pos=token_pos
    )
    s = extract_last_token_residuals(
        model, tokenizer, harmless, layers, device, token_pos=token_pos
    )
    return {L: compute_r_hat(h[L], s[L]) for L in layers}


def cosmed_vs(fp32_dirs, cond_dirs, layers):
    """Per-layer cosine of a condition's directions against FP32."""
    cos = [cos_sim(fp32_dirs[L], cond_dirs[L]) for L in layers]
    return {
        "cos_median": float(np.median(cos)),
        "cos_min": float(np.min(cos)),
        "cos_max": float(np.max(cos)),
        "cos_per_layer": {str(L): float(c) for L, c in zip(layers, cos)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--token_pos", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    token_pos = (args.token_pos if args.token_pos is not None
                 else default_token_pos(args.model))

    hi = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmful = [(it.get("behavior") or it.get("prompt") or "")
               if isinstance(it, dict) else str(it) for it in hi]
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    # FP32 directions + Restorer (both need the FP32 model, built before delete).
    fp32_model, tokenizer = load_fp32(args.model)
    layers = list(range(fp32_model.config.num_hidden_layers))
    fp32_dirs = directions(fp32_model, tokenizer, harmful, harmless, layers,
                           device, token_pos)
    restorer = Restorer(fp32_model)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Quant baseline, then read/write/both restored -- recompute directions each.
    results = {}
    conditions = [("quant", None), ("read_restored", "restore_read"),
                  ("write_restored", "restore_write"),
                  ("both_restored", "restore_both")]
    for name, fn in conditions:
        model, _ = load_quantized(args.model, quantizer=args.quantizer)
        if fn is not None:
            getattr(restorer, fn)(model)
        d = directions(model, tokenizer, harmful, harmless, layers, device, token_pos)
        results[name] = cosmed_vs(fp32_dirs, d, layers)
        print(f"[{name}] cosmed_median = {results[name]['cos_median']:.4f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    q = results["quant"]["cos_median"]
    data = {
        "summary": {
            "cosmed_quant": q,
            "cosmed_read_restored": results["read_restored"]["cos_median"],
            "cosmed_write_restored": results["write_restored"]["cos_median"],
            "cosmed_both_restored": results["both_restored"]["cos_median"],
            # positive delta = restoring this side pulls the direction toward FP32
            "delta_read": results["read_restored"]["cos_median"] - q,
            "delta_write": results["write_restored"]["cos_median"] - q,
        },
        "per_condition": results,
    }
    cfg = RunConfig(
        model=args.model, quantizer=args.quantizer,
        extra={"n_calib": args.n_calib, "token_pos": token_pos, "seed": SEED},
    )
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
