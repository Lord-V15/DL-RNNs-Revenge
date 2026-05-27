"""Learning rate schedule: linear warmup then cosine decay.

Per proposal §5: warmup 200 steps from 0 to peak_lr (3e-4), then cosine decay
to min_lr (3e-5) over the remaining steps.

Implemented as a `LambdaLR`-friendly callable so it composes with the standard
PyTorch optimizer interface. The closure returns a scalar multiplier on the
optimizer's base lr — so set the optimizer's `lr` to the peak value and let
the schedule scale it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WarmupCosineSchedule:
    """Linear warmup -> cosine decay schedule.

    Args:
        warmup_steps: number of warmup steps (peak reached at step == warmup_steps).
        total_steps: total training steps (>= warmup_steps).
        min_lr_ratio: floor as a fraction of the peak lr. E.g. min_lr_ratio
            = 0.1 means the lr decays to 0.1 * peak_lr by the end.

    Usage:
        sched = WarmupCosineSchedule(warmup_steps=200, total_steps=5000,
                                     min_lr_ratio=3e-5 / 3e-4)
        lr_lambda = sched   # callable; pass to LambdaLR

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    """
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1

    def __call__(self, step: int) -> float:
        """Return the multiplier on the optimizer's base lr at `step`.

        Step 0 returns 0.0 (or very close to it — linear warmup starts at 0
        and reaches 1.0 at step == warmup_steps).
        """
        if step < self.warmup_steps:
            # Linear from 0 -> 1 over warmup_steps. We use (step + 1) so that
            # step 0 is not exactly zero (a zero lr means no parameter update,
            # which is wasteful for the very first step).
            return (step + 1) / max(self.warmup_steps, 1)

        # Cosine decay from 1.0 down to min_lr_ratio over the remaining steps.
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)  # clamp so post-training calls stay at min
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
