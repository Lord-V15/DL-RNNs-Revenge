"""Named experiments — one entry per cell of the run matrix.

This file is the single source of truth for which experiments exist. To run
an experiment, `scripts/train.py --exp <name>` looks it up here. Adding a
new run is one dict entry; no YAML files, no CLI sprawl.

Coverage (proposal §6, minGRU owner's share):

  TinyShakespeare × {256, 1024, 2048}        — 3 cells
  Long-range copy × {short, medium, long}    — 3 cells
  Induction       × {short, medium, long}    — 3 cells

Total: 9 cells. The TinyShakespeare-256 entry uses seed 42 by default but
the proposal already has it run at 3 seeds (42, 1337, 2025) — pass --seed
on the command line to override.

Vocab sizes reflect the actual generator output:
  * TinyShakespeare: ~65, derived from corpus at runtime
  * Long-range copy: 55  (26 uppercase keys + 26 lowercase distractors + space + colon + pipe)
  * Induction:       27  (26 lowercase + space)
"""
from __future__ import annotations

from configs.base import TrainConfig
from configs.model_mingru import default_mingru_config


def _ts_config(block_size: int, seed: int = 42) -> TrainConfig:
    return TrainConfig(
        exp_name=f"mingru_tinyshakespeare_{block_size}",
        task="tinyshakespeare",
        seed=seed,
        dropout=0.05,
        block_size=block_size,
        model_config=default_mingru_config(vocab_size=65, dropout=0.05),
    )


def _copy_config(length_tier: str, seed: int = 42) -> TrainConfig:
    return TrainConfig(
        exp_name=f"mingru_longcopy_{length_tier}",
        task="longrange_copy",
        seed=seed,
        dropout=0.05,
        length_tier=length_tier,  # type: ignore[arg-type]
        model_config=default_mingru_config(vocab_size=55, dropout=0.05),
    )


def _induction_config(length_tier: str, seed: int = 42) -> TrainConfig:
    return TrainConfig(
        exp_name=f"mingru_induction_{length_tier}",
        task="induction",
        seed=seed,
        dropout=0.05,
        length_tier=length_tier,  # type: ignore[arg-type]
        total_steps=20000,
        lr_decay_steps=50000,
        model_config=default_mingru_config(vocab_size=27, dropout=0.05),
    )


EXPERIMENTS: dict[str, TrainConfig] = {
    # TinyShakespeare
    "mingru_tinyshakespeare_256":  _ts_config(256),
    "mingru_tinyshakespeare_1024": _ts_config(1024),
    "mingru_tinyshakespeare_2048": _ts_config(2048),
    # Long-range copy
    "mingru_longcopy_short":  _copy_config("short"),
    "mingru_longcopy_medium": _copy_config("medium"),
    "mingru_longcopy_long":   _copy_config("long"),
    # Induction
    "mingru_induction_short":  _induction_config("short"),
    "mingru_induction_medium": _induction_config("medium"),
    "mingru_induction_long":   _induction_config("long"),
}


def get(name: str) -> TrainConfig:
    """Look up an experiment by name. Raises KeyError with a helpful message
    listing the available experiments if `name` is unknown."""
    if name not in EXPERIMENTS:
        avail = "\n  ".join(sorted(EXPERIMENTS))
        raise KeyError(
            f"unknown experiment {name!r}. Available experiments:\n  {avail}"
        )
    return EXPERIMENTS[name]
