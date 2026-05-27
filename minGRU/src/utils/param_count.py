"""Parameter counting and budget assertions.

The proposal commits to a ~750K parameter budget (±10%) for fair cross-paradigm
comparison. Drift here invalidates the comparison, so we expose a strict
assertion helper that the trainer calls at startup. If you tune the model and
the count drifts outside tolerance, the run fails fast rather than producing
results that look comparable but aren't.

Tied parameters (e.g. tied input/output embeddings) are counted once. This is
done by deduplicating Parameter objects by id.
"""
from __future__ import annotations

import torch.nn as nn


TARGET_PARAM_COUNT = 750_000
DEFAULT_TOLERANCE = 0.10  # ±10% per proposal §5


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count parameters, deduplicating shared Parameters by id.

    Args:
        model: any nn.Module.
        only_trainable: if True, skips params with requires_grad=False.

    Returns:
        Total parameter count (int).
    """
    params = model.parameters()
    if only_trainable:
        params = (p for p in params if p.requires_grad)

    seen: set[int] = set()
    total = 0
    for p in params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def assert_param_budget(
    model: nn.Module,
    target: int = TARGET_PARAM_COUNT,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "model",
) -> int:
    """Check `model`'s param count is within `target` × (1 ± tolerance).

    Returns the actual count on success. Raises ValueError otherwise — the
    intent is to fail fast at trainer startup rather than discover the drift
    after a long run.
    """
    n = count_parameters(model)
    lo = int(target * (1 - tolerance))
    hi = int(target * (1 + tolerance))
    if not (lo <= n <= hi):
        raise ValueError(
            f"{name} has {n:,} parameters; expected {target:,} ± {tolerance:.0%} "
            f"(range [{lo:,}, {hi:,}]). Adjust model config to bring it into "
            f"range — the cross-paradigm comparison assumes matched budgets."
        )
    return n


def param_breakdown(model: nn.Module) -> dict[str, int]:
    """Per-named-parameter count for debugging budget drift.

    Returns a dict mapping parameter name to numel(). Useful when
    `assert_param_budget` fails and you need to see which submodule grew.
    """
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
