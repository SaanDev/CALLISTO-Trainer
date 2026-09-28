"""Reproducibility helpers."""

# NOTE: Vendored from H:\Burst Identifier (src/utils/seed.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Set Python, NumPy, and PyTorch random seeds when PyTorch is available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
