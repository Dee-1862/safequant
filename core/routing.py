# Per-prompt operative-layer routing from residual projections onto r_hat.

from __future__ import annotations

import numpy as np


def per_prompt_projections(residuals_by_layer, r_hat, normalize = False):
    """
    Per-prompt operative-layer routing analysis (de-confounded null).

    Inputs:
        - residuals_by_layer (dict): Layer -> list of residual tensors
        - r_hat (torch.Tensor): Reference direction
        - normalize (bool): If True, use cosine projection instead of raw dot

    Outputs:
        - projections (np.ndarray): Shape (n_prompts, n_layers)
        - layers (list): Sorted layer keys matching columns
    """
    # Fixed column order and prompt count from the first layer
    layers = sorted(residuals_by_layer.keys())
    n_prompts = len(residuals_by_layer[layers[0]])
    out = np.zeros((n_prompts, len(layers)), dtype = np.float64)

    # Fill projection matrix prompt-by-prompt per layer
    for li, layer_idx in enumerate(layers):
        for pi, t in enumerate(residuals_by_layer[layer_idx]):
            v = float(t @ r_hat)
            if normalize:
                v = v / (float(t.norm()) + 1e-8)
            out[pi, li] = v
    return out, layers


def operative_layer_per_prompt(projections, layers):
    """
    Layer index per prompt where |projection| is maximal.

    Inputs:
        - projections (np.ndarray): Shape (n_prompts, n_layers)
        - layers (list): Layer keys matching columns of projections

    Outputs:
        - operative (list): Per-prompt layer key of max |projection|
    """
    # Argmax over absolute projection magnitude
    abs_proj = np.abs(projections)
    argmax_indices = abs_proj.argmax(axis = 1)
    return [layers[i] for i in argmax_indices]
