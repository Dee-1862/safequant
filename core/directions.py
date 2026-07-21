# Refusal-direction (R-hat) computation and cosine / separation helpers.

from __future__ import annotations

import numpy as np
import torch


def compute_r_hat(harmful_lasts, harmless_lasts):
    """
    Compute the normalized R-hat direction from harmful vs harmless last tokens.

    Inputs:
        - harmful_lasts (list): Harmful last-token residual tensors
        - harmless_lasts (list): Harmless last-token residual tensors

    Outputs:
        - r_hat (torch.Tensor): Unit difference direction (or unnormalized if tiny)
    """
    # Mean activations for each class
    h = torch.stack(harmful_lasts).mean(0)
    s = torch.stack(harmless_lasts).mean(0)

    # Difference vector, normalized when large enough
    r = h - s
    n = r.norm()
    if n > 1e-6:
        return r / n
    else:
        return r


def per_layer_separation(harmful_lasts, harmless_lasts, r_hat):
    """
    Score-class separation along r_hat in pooled-std units.

    Inputs:
        - harmful_lasts (list): Harmful last-token residual tensors
        - harmless_lasts (list): Harmless last-token residual tensors
        - r_hat (torch.Tensor): Precomputed refusal direction

    Outputs:
        - separation_sigma (float): (mean_h - mean_s) / pooled_std
        - harmful_mean (float): Mean projection of harmful residuals
        - harmless_mean (float): Mean projection of harmless residuals
    """
    # Project each residual onto r_hat
    h_scores = []
    s_scores = []
    for t in harmful_lasts:
        h_scores.append(float(t @ r_hat))
    for t in harmless_lasts:
        s_scores.append(float(t @ r_hat))

    # Pooled std and mean-difference separation
    h_scores = np.array(h_scores, dtype = np.float64)
    s_scores = np.array(s_scores, dtype = np.float64)
    pooled = np.sqrt((h_scores.var() + s_scores.var()) / 2 + 1e-12)
    sep = (h_scores.mean() - s_scores.mean()) / pooled

    separation_sigma = float(sep)
    harmful_mean = float(np.mean(h_scores))
    harmless_mean = float(np.mean(s_scores))
    return separation_sigma, harmful_mean, harmless_mean


def cos_sim(a, b):
    """
    Cosine similarity between two tensors.

    Inputs:
        - a (torch.Tensor): First tensor
        - b (torch.Tensor): Second tensor

    Outputs:
        - cos_sim (float): Cosine similarity, or 0.0 if either norm is tiny
    """
    # Work in float for stable norms
    a = a.float()
    b = b.float()

    # Guard near-zero vectors
    na, nb = a.norm(), b.norm()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))
