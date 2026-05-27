"""Base training configuration.

`TrainConfig` is a frozen dataclass holding every hyperparameter the trainer
needs. Per-experiment configs (configs/experiments.py) construct instances of
this with task-specific overrides.

Defaults are taken from proposal §5:
  AdamW, lr 3e-4 -> 3e-5 cosine, warmup 200, batch 64, 5000 steps,
  grad clip 1.0, dropout 0.2, bf16.

The config is repr-dumped into meta.txt and config.txt, so changes here are
captured in every run's metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


TaskName = Literal["tinyshakespeare", "longrange_copy", "induction"]
LengthTier = Literal["short", "medium", "long"]


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters and run-shaping options.

    Frozen so a TrainConfig instance can be safely shared / hashed and so
    accidental mutation in the trainer is impossible.
    """
    # -- run identity ----------------------------------------------------
    exp_name: str                       # used as the run directory name root
    task: TaskName
    seed: int = 42

    # -- task-specific length spec --------------------------------------
    # For TinyShakespeare: block_size is the context window (256/1024/2048).
    # For synthetic tasks: length_tier is "short"/"medium"/"long" and
    #   block_size is set from the data at load time. We still record it
    #   in the config (filled in by the trainer) for the metadata.
    block_size: Optional[int] = None    # TS: explicit; synth: derived later
    length_tier: Optional[LengthTier] = None  # synth tasks only

    # -- optimizer ------------------------------------------------------
    optimizer: str = "adamw"
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1           # AdamW default for transformers; harmless for minGRU
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # -- schedule --------------------------------------------------------
    warmup_steps: int = 200
    total_steps: int = 5000
    lr_decay_steps: int | None = None  # cosine decay horizon (defaults to total_steps)

    # -- training loop --------------------------------------------------
    batch_size: int = 64
    grad_clip: float = 1.0
    dropout: float = 0.2                # propagated to the model config
    mixed_precision: Literal["bf16", "fp32"] = "bf16"

    # -- evaluation -----------------------------------------------------
    eval_interval: int = 250            # eval every N training steps
    eval_batch_size: int = 64
    # For TinyShakespeare val, use non-overlapping windows over the val split.
    # `eval_max_batches` caps the number of batches if val is huge; None means
    # use the entire val set, which is what the report expects.
    eval_max_batches: Optional[int] = None

    # -- data paths -----------------------------------------------------
    data_root: str = "data"             # data/{tinyshakespeare,longrange_copy,induction}

    # -- output ---------------------------------------------------------
    runs_root: str = "runs"             # parent dir for all run directories

    # -- model ----------------------------------------------------------
    # Model-specific config is a separate dataclass (configs/model_mingru.py).
    # We store its as-dict here so the meta.txt captures it. The trainer
    # constructs the model from this dict.
    model_config: dict = field(default_factory=dict)

    # -- bookkeeping ----------------------------------------------------
    # Tolerance for parameter-budget assertion. Defaults to proposal's ±10%.
    param_budget_target: int = 750_000
    param_budget_tolerance: float = 0.10

    def to_dict(self) -> dict:
        """Return a dict suitable for JSON / repr / meta.txt dumping."""
        return asdict(self)

    def pretty_repr(self) -> str:
        """Multi-line repr with one field per line, for meta.txt."""
        d = self.to_dict()
        lines = ["TrainConfig("]
        for k, v in d.items():
            lines.append(f"    {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
