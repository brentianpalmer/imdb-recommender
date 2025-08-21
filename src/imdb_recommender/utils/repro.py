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
        Seed to use. When ``IMDBREC_SEED`` environment variable is set, its
        value takes precedence. Otherwise ``seed`` is used, falling back to
        ``1234`` when ``None``.

    Returns
    -------
    int
        The seed value that was applied.
    """
    env_seed = os.getenv(ENV_VAR)
    if env_seed is not None:
        seed = int(env_seed)
    elif seed is None:
        seed = DEFAULT_SEED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


__all__ = ["set_global_seed"]
