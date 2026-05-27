"""Deterministic seeding.

Calling `set_seed(n)` makes torch, numpy, and Python's random module
deterministic for the rest of the process. cuDNN is set to deterministic
mode and benchmark mode is disabled — this trades some speed for
reproducibility, which is the right call when the goal is comparable PPL
numbers across seeds.

For full determinism on CUDA, `torch.use_deterministic_algorithms(True)` is
also called. Some ops (notably certain backward passes) will then raise if
they have no deterministic implementation — that's by design; we want to
know rather than silently get nondeterministic results.

Note: even with all of this, results can differ across GPU models, CUDA
versions, and PyTorch versions. "Deterministic" means "reproducible on the
same hardware/software stack", not "identical everywhere".
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, strict: bool = True) -> None:
    """Seed all RNGs used in training.

    Args:
        seed: integer seed.
        strict: if True, also enables torch's deterministic-algorithms mode.
            Set to False if you hit an op without a deterministic impl and
            decide to accept the nondeterminism for that run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        # CUBLAS workspace config is required for deterministic matmul on
        # some CUDA versions. Set before any CUDA work.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_states() -> dict:
    """Capture the current RNG state for checkpointing.

    Returns a dict suitable for `torch.save`. Pair with `set_rng_states` on
    resume to continue training with the same random stream.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def set_rng_states(states: dict) -> None:
    """Restore RNG state captured by `get_rng_states`."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
