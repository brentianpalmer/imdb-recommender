"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np

ENV_VAR = "IMDBREC_SEED"
DEFAULT_SEED = 1234


def set_global_seed(seed: int | None = None) -> int:
    """Set global random seeds for deterministic behaviour.

    Parameters
    ----------
    seed:
        Seed to use. When ``None``, ``IMDBREC_SEED`` environment variable is
        read with a fallback of ``1234``.

    Returns
    -------
    int
        The seed value that was applied.
    """
    if seed is None:
        seed = int(os.getenv(ENV_VAR, DEFAULT_SEED))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


__all__ = ["set_global_seed"]
