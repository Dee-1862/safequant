# Probe refit noise-floor control for probe-rotation screen.
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _cos(a, b):
    """
    Cosine similarity between two vectors.

    Inputs:
        - a (array-like): First vector
        - b (array-like): Second vector

    Outputs:
        - cos (float): Cosine similarity, or 0.0 if either vector is near-zero
    """
    # Cast to float64 for stable norms
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    # Guard against near-zero norms
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def _fit_w(h_list, s_list):
    """
    Fit a linear probe and return its weight vector.

    Inputs:
        - h_list (list[Tensor]): Harmful residual vectors
        - s_list (list[Tensor]): Harmless residual vectors

    Outputs:
        - weight (ndarray): LogisticRegression coef_[0]
    """
    # Stack features and binary labels (harmful = 1)
    X = np.stack([t.numpy() for t in h_list] + [t.numpy() for t in s_list])
    y = np.array([1] * len(h_list) + [0] * len(s_list))

    # Fit L2-regularized logistic regression
    clf = LogisticRegression(C = 1.0, max_iter = 2000, random_state = SEED)
    clf.fit(X, y)
    return clf.coef_[0]


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--token_pos", type = int, default = -1)
    args = ap.parse_args()

    # Load calibration prompts
    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # Extract last-token residuals from FP32
    fp_model, fp_tok = load_fp32(args.model)
    device = next(fp_model.parameters()).device
    layers = list(range(len(get_inner(fp_model).layers)))
    h = extract_last_token_residuals(fp_model, fp_tok, harmful, layers, device, token_pos = args.token_pos)
    s = extract_last_token_residuals(fp_model, fp_tok, harmless, layers, device, token_pos = args.token_pos)
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    # Random half-split of harmful and harmless indices
    rng = np.random.default_rng(SEED)
    nh, ns = len(h[layers[0]]), len(s[layers[0]])
    hp, sp = rng.permutation(nh), rng.permutation(ns)
    hA, hB = hp[: nh // 2], hp[nh // 2 :]
    sA, sB = sp[: ns // 2], sp[ns // 2 :]

    # Per-layer cosine between independent probe fits
    cos_floor = {}
    for layer_idx in layers:
        wA = _fit_w([h[layer_idx][i] for i in hA], [s[layer_idx][i] for i in sA])
        wB = _fit_w([h[layer_idx][i] for i in hB], [s[layer_idx][i] for i in sB])
        cos_floor[layer_idx] = _cos(wA, wB)
    vals = list(cos_floor.values())

    # Persist noise-floor summary
    cfg = RunConfig(
        model = args.model,
        omit_quantizer = True,
        extra = {"n_calib": args.n_calib, "token_pos": args.token_pos, "seed": SEED},
    )
    data = {
        "summary": {
            "cos_floor_min": float(min(vals)),
            "cos_floor_median": float(np.median(vals)),
            "cos_floor_max": float(max(vals)),
        },
        "cos_floor_per_layer": {str(k): v for k, v in cos_floor.items()},
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
