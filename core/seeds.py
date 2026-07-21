# Global RNG seeding for reproducible SafeQuant experiments.

import os
import random as _python_random

import numpy as np
import torch

# THE SEED. Do not change. Every script in SafeQuant v2 uses this single value.
# If you ever need to vary the seed (example: to estimate variance), pass the variant
# explicitly via the script's --seed argument.
# Never change DEFAULT_SEED itself, as that would invalidate all reproductions.

DEFAULT_SEED = 42


def set_global_seed(seed = DEFAULT_SEED, *, deterministic = False):
    """
    Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    Inputs:
        - seed (int): Integer seed (default 42)
        - deterministic (bool): If True, force deterministic torch/cudnn algorithms

    Outputs:
        - seed (int): The seed that was set
    """
    # Seed host-side RNGs
    _python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Mirror seed across all CUDA devices when present
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional strict determinism (slower; may disable some fast kernels)
    if deterministic:
        # Required before the first CUDA matmul when deterministic algorithms are on
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only = True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
