"""Checkpointing.

Two checkpoints per run, both inside the run's checkpoints/ directory:

  * best.pt  — written whenever a new lowest val loss is seen.
  * last.pt  — written at every eval. Used by --resume.

Each checkpoint is a dict containing:
  - model_state_dict
  - optimizer_state_dict
  - scheduler_state_dict
  - step (the training step just completed when this checkpoint was saved)
  - best_val_loss (the lowest val loss seen so far)
  - rng_states (from src.utils.seed.get_rng_states)
  - config (a dict of the TrainConfig — repr/asdict)

Resume semantics: `load_checkpoint(path, model, optimizer, scheduler)`
restores all four, returns the step + best_val_loss + rng_states for the
trainer to apply. Trainer is responsible for calling `set_rng_states` after
restoring its own state.

CSV logs are NOT touched here — they're appended to by `logging_utils.py`,
which uses append mode so resume just continues writing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    best_val_loss: float,
    rng_states: dict,
    config: dict,
) -> None:
    """Atomically save a checkpoint.

    Writes to a tempfile and renames, so a crash mid-save doesn't leave a
    corrupted checkpoint behind. Important on shared filesystems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "rng_states": rng_states,
        "config": config,
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device | None = None,
) -> dict:
    """Load a checkpoint, restoring model + optimizer + scheduler in-place.

    Args:
        path: checkpoint path.
        model: module to restore.
        optimizer: optimizer to restore. Pass None to skip (e.g. for inference).
        scheduler: scheduler to restore. Pass None to skip.
        map_location: passed to torch.load. Useful for CPU loading on a
            machine without a GPU.

    Returns:
        dict with keys:
            step (int)
            best_val_loss (float)
            rng_states (dict) — restore these in the trainer via
                src.utils.seed.set_rng_states
            config (dict) — the frozen TrainConfig from the run that saved
                this checkpoint
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return {
        "step": payload["step"],
        "best_val_loss": payload["best_val_loss"],
        "rng_states": payload["rng_states"],
        "config": payload["config"],
    }
