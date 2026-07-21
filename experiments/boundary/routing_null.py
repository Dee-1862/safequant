# Appendix: de-confounded routing null (magnitude-normalized operative layer).
import argparse
import gc
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.directions import compute_r_hat, per_layer_separation
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.residuals import default_token_pos, extract_last_token_residuals, get_inner
from core.routing import operative_layer_per_prompt, per_prompt_projections
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _summary(op_list):
    """
    Summarize the distribution of per-prompt operative layers.

    Inputs:
        - op_list (list[int]): Operative layer index per prompt

    Outputs:
        - summary (dict): Mode, mean, std, and distinct-layer count
    """
    # Mode layer and how often it wins
    c = Counter(op_list)
    mode_layer, mode_count = c.most_common(1)[0]

    # Aggregate distribution statistics
    return {
        "mode_layer": int(mode_layer),
        "mode_count": int(mode_count),
        "mode_fraction": float(mode_count / len(op_list)),
        "mean": float(np.mean(op_list)),
        "std": float(np.std(op_list)),
        "n_distinct_layers": len(c),
    }


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--token_pos", type = int, default = None)
    ap.add_argument("--normalize", action = "store_true", default = False)
    args = ap.parse_args()

    # Resolve token position and optional normalize variant label
    token_pos = args.token_pos if args.token_pos is not None else default_token_pos(args.model)
    variant = "norm" if args.normalize else None

    # Load calibration harmful and harmless prompts
    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # FP32 residuals, per-layer separations, and best-layer r_hat
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    layers = list(range(len(get_inner(fp32_model).layers)))
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device, token_pos)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device, token_pos)
    fp_r_hats = {L: compute_r_hat(fp_h[L], fp_s[L]) for L in layers}
    fp_seps = {L: per_layer_separation(fp_h[L], fp_s[L], fp_r_hats[L])[0] for L in layers}
    fp32_best = max(layers, key = lambda L: fp_seps[L])
    r_hat_fp = fp_r_hats[fp32_best]
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Quantized residuals and best-layer r_hat
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    q_device = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device, token_pos)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device, token_pos)
    q_r_hats = {L: compute_r_hat(q_h[L], q_s[L]) for L in layers}
    q_seps = {L: per_layer_separation(q_h[L], q_s[L], q_r_hats[L])[0] for L in layers}
    quant_best = max(layers, key = lambda L: q_seps[L])
    r_hat_q = q_r_hats[quant_best]

    # Project quantized residuals onto FP and quantized directions
    q_proj_fp_h, _ = per_prompt_projections(q_h, r_hat_fp, normalize = args.normalize)
    q_proj_fp_s, _ = per_prompt_projections(q_s, r_hat_fp, normalize = args.normalize)
    q_proj_q_h, _ = per_prompt_projections(q_h, r_hat_q, normalize = args.normalize)
    q_proj_q_s, _ = per_prompt_projections(q_s, r_hat_q, normalize = args.normalize)

    # Operative layer per prompt under each direction
    op_fp_h = operative_layer_per_prompt(q_proj_fp_h, layers)
    op_fp_s = operative_layer_per_prompt(q_proj_fp_s, layers)
    op_q_h = operative_layer_per_prompt(q_proj_q_h, layers)
    op_q_s = operative_layer_per_prompt(q_proj_q_s, layers)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Persist routing summaries
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = variant,
        extra = {
            "n_calib": args.n_calib,
            "token_pos": token_pos,
            "normalize": args.normalize,
            "seed": SEED,
        },
    )
    data = {
        "n_layers": len(layers),
        "fp32_best_layer": fp32_best,
        "quant_best_layer": quant_best,
        "fp32_separations": fp_seps,
        "quant_separations": q_seps,
        "routing_summary": {
            "operative_layer_FP_rhat_harmful": _summary(op_fp_h),
            "operative_layer_FP_rhat_harmless": _summary(op_fp_s),
            "operative_layer_q_rhat_harmful": _summary(op_q_h),
            "operative_layer_q_rhat_harmless": _summary(op_q_s),
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
