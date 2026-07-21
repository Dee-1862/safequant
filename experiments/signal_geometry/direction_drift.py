# Table 1: direction drift (delta_med + cosmed).
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_harmbench_prompts, load_harmless_prompts, load_wildjailbreak_prompts
from core.directions import compute_r_hat, cos_sim, per_layer_separation
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.residuals import default_token_pos, extract_last_token_residuals
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def run_for_model(model, tokenizer, harmful, harmless, layers, device, label, token_pos = -1):
    """
    Extract per-layer refusal directions and separations for one model.

    Inputs:
        - model (nn.Module): Model to extract residuals from
        - tokenizer (PreTrainedTokenizer): Tokenizer matched to model
        - harmful (list): Harmful prompts
        - harmless (list): Harmless prompts
        - layers (list[int]): Layer indices
        - device (str | torch.device): Device for extraction
        - label (str): Log label for this condition
        - token_pos (int): Token position for residuals

    Outputs:
        - r_hats (dict[int, Tensor]): Unit refusal direction per layer
        - seps (dict[int, dict]): Separation statistics per layer
    """
    # Extract last-token residuals for both prompt sets
    print(f"\n[{label}] extracting residuals (token_pos={token_pos})")
    h_caps = extract_last_token_residuals(
        model, tokenizer, harmful, layers, device, token_pos = token_pos
    )
    s_caps = extract_last_token_residuals(
        model, tokenizer, harmless, layers, device, token_pos = token_pos
    )

    # Diff-in-means directions and separation scores
    r_hats, seps = {}, {}
    for layer_idx in layers:
        r = compute_r_hat(h_caps[layer_idx], s_caps[layer_idx])
        r_hats[layer_idx] = r
        sep, hm, sm = per_layer_separation(h_caps[layer_idx], s_caps[layer_idx], r)
        seps[layer_idx] = {"separation_sigma": sep, "harmful_mean": hm, "harmless_mean": sm}
    return r_hats, seps


def analyze(quantizer, fp32_r_hats, fp32_seps, q_r_hats, q_seps, layers, n_calib, model_name):
    """
    Compare FP32 vs quantized directions and package the result payload.

    Inputs:
        - quantizer (str): Quantizer id used for the quantized cell
        - fp32_r_hats (dict): FP32 directions by layer
        - fp32_seps (dict): FP32 separations by layer
        - q_r_hats (dict): Quantized directions by layer
        - q_seps (dict): Quantized separations by layer
        - layers (list[int]): Layer indices
        - n_calib (int): Calibration sample count
        - model_name (str): Model alias

    Outputs:
        - data (dict): Result payload for save_result
    """
    # Cosine similarity and Pearson of sep vs cosine
    cos_per_layer = {L: cos_sim(fp32_r_hats[L], q_r_hats[L]) for L in layers}
    cos_vals = list(cos_per_layer.values())
    fp32_seps_arr = np.array([fp32_seps[L]["separation_sigma"] for L in layers])
    cos_arr = np.array([cos_per_layer[L] for L in layers])
    pearson = (
        float(np.corrcoef(fp32_seps_arr, cos_arr)[0, 1])
        if fp32_seps_arr.std() > 1e-9 and cos_arr.std() > 1e-9
        else 0.0
    )

    # Weak-layer thresholds from the paper screens
    weak = [L for L, c in cos_per_layer.items() if c < 0.95]
    very_weak = [L for L, c in cos_per_layer.items() if c < 0.80]
    return {
        "model": model_name,
        "quantizer": quantizer,
        "n_calib": n_calib,
        "n_layers": len(layers),
        "cos_sim_per_layer": {str(L): cos_per_layer[L] for L in layers},
        "fp32_separation_per_layer": {str(L): fp32_seps[L] for L in layers},
        f"{quantizer}_separation_per_layer": {str(L): q_seps[L] for L in layers},
        "summary": {
            "cos_sim_min": float(min(cos_vals)),
            "cos_sim_median": float(np.median(cos_vals)),
            "cos_sim_max": float(max(cos_vals)),
            "layers_below_0_95": weak,
            "layers_below_0_80": very_weak,
            "pearson_sep_vs_cos": pearson,
        },
    }


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--n_calib", type = int, default = 50)
    ap.add_argument("--dataset", default = "wildjailbreak", choices = ["wildjailbreak", "harmbench"])
    ap.add_argument("--token_pos", type = int, default = None)
    args = ap.parse_args()

    # Resolve quantizer list, device, and token position
    quantizers = [q.strip() for q in args.quantizer.split(",") if q.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token_pos = args.token_pos if args.token_pos is not None else default_token_pos(args.model)

    # Load harmful prompts from the selected dataset
    if args.dataset == "wildjailbreak":
        harmful_items = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
        harmful = [
            (it.get("behavior") or it.get("prompt") or "") if isinstance(it, dict) else str(it)
            for it in harmful_items
        ]
    else:
        harmful = load_harmbench_prompts(n_samples = args.n_calib)
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # FP32 directions once, then compare each quantizer
    fp32_model, tokenizer = load_fp32(args.model)
    n_layers = fp32_model.config.num_hidden_layers
    layers = list(range(n_layers))
    fp32_r_hats, fp32_seps = run_for_model(
        fp32_model, tokenizer, harmful, harmless, layers, device, "FP32", token_pos
    )
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # One output artifact per quantizer cell
    for q in quantizers:
        q_model, _ = load_quantized(args.model, quantizer = q)
        q_r_hats, q_seps = run_for_model(
            q_model, tokenizer, harmful, harmless, layers, device, q, token_pos
        )
        del q_model
        gc.collect()
        torch.cuda.empty_cache()

        data = analyze(q, fp32_r_hats, fp32_seps, q_r_hats, q_seps, layers, args.n_calib, args.model)
        cfg = RunConfig(
            model = args.model,
            quantizer = q,
            extra = {"n_calib": args.n_calib, "token_pos": token_pos, "dataset": args.dataset},
        )
        path = save_result(_SCRIPT, cfg, data)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
