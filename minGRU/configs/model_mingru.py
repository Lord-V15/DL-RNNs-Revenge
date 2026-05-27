"""Default minGRU model configuration.

A thin helper to build a `MinGRUConfig` dict with the proposal's defaults
(d_model=128, L=4, expansion=1.0, mlp_mult=4.5). The trainer accepts a dict
and instantiates `MinGRUConfig` from it — keeping the model decoupled from
the training pipeline.
"""
from __future__ import annotations


def default_mingru_config(vocab_size: int, dropout: float = 0.2) -> dict:
    """Return a dict matching `MinGRUConfig`'s field names.

    Args:
        vocab_size: task-specific; set per dataset (TS ~65, copy 29, induction 27).
        dropout: from the TrainConfig — passed through here so a single change
            in the train config propagates to the model.

    Returns:
        dict that `MinGRUConfig(**d)` accepts.
    """
    return {
        "vocab_size": vocab_size,
        "d_model": 128,
        "n_blocks": 4,
        "expansion_factor": 1.0,
        "mlp_mult": 4.5,
        "conv_kernel": 4,
        "dropout": dropout,
        "tie_embeddings": True,
        "rms_norm_eps": 1e-5,
    }
