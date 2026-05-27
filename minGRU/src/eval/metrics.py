"""Evaluation metrics.

All metric helpers operate on `logits`, `targets`, and optionally a boolean
`mask`. Shapes follow the trainer's convention:

  * logits:  [B, T, V] float
  * targets: [B, T]    int64
  * mask:    [B, T]    bool   (True at positions to include)

Two kinds of metrics are computed:

  1. Aggregate val PPL: cross-entropy averaged over all positions, then
     exponentiated. Reported per (paradigm, task, length) cell — Table 1
     of the report.

  2. Discriminating-position metrics: cross-entropy and accuracy averaged
     over only the positions where `mask` is True. For long-range copy this
     is the 5 recall positions; for induction it is the 5 pattern-completion
     positions. Table 2 of the report.

A separate helper, `per_position_breakdown`, returns the loss at each masked
position individually (not averaged). This feeds the per-position CSV that
Plot 2 needs, and the metrics for "predicting the 5th char given the first
4" can be extracted from it.

All computations run in fp32 regardless of training dtype, to avoid PPL
estimates being noisy because of bf16.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# A sentinel for "no per-position breakdown" — keeps the EvalResult shape
# stable when masked metrics are not applicable (e.g. TinyShakespeare).
_NO_BREAKDOWN: list[float] = []


@dataclass
class EvalResult:
    """Bundle of metrics from one full evaluation pass.

    Fields are kept flat so CSV writers can dump them without nested logic.
    Masked fields are NaN when no mask was supplied (TinyShakespeare path).
    """
    # Aggregate
    val_loss: float                    # mean cross-entropy over all positions
    val_ppl: float                     # exp(val_loss)
    n_tokens: int                      # total positions counted in val_loss

    # Discriminating-position (NaN if no mask)
    masked_loss: float                 # mean CE over masked positions only
    masked_ppl: float
    masked_acc: float                  # exact-match accuracy at masked positions
    n_masked_tokens: int

    # Per-position breakdown over masked positions (e.g. 5 entries for copy
    # and induction). Empty list if no mask. Order is positional: index 0 =
    # first masked position in each sequence, averaged over the batch.
    masked_loss_per_position: list[float]

    def as_csv_row(self) -> dict[str, float | int]:
        """Flatten to a dict suitable for CSV writing.

        Per-position fields become `masked_loss_pos0`, `masked_loss_pos1`, ...
        """
        row: dict[str, float | int] = {
            "val_loss": self.val_loss,
            "val_ppl": self.val_ppl,
            "n_tokens": self.n_tokens,
            "masked_loss": self.masked_loss,
            "masked_ppl": self.masked_ppl,
            "masked_acc": self.masked_acc,
            "n_masked_tokens": self.n_masked_tokens,
        }
        # Always emit exactly 5 per-position columns so every task's CSV rows
        # have the same schema — the logging_utils column-consistency check
        # requires this. Copy has 5 masked positions; induction has 1.
        # Missing positions are filled with NaN.
        MAX_POS_COLS = 5
        for i in range(MAX_POS_COLS):
            if i < len(self.masked_loss_per_position):
                row[f"masked_loss_pos{i}"] = self.masked_loss_per_position[i]
            else:
                row[f"masked_loss_pos{i}"] = float("nan")
        return row


def _cross_entropy_per_token(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy at every position, no reduction.

    Returns:
        loss: [B, T] float32 tensor.
    """
    B, T, V = logits.shape
    # reduction='none' returns shape [B*T]; reshape back to [B, T].
    flat_loss = F.cross_entropy(
        logits.float().reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    )
    return flat_loss.view(B, T)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches,
    device: str | torch.device,
    has_mask: bool = False,
    mask_positions_per_sequence: Optional[int] = None,
) -> EvalResult:
    """Run a full evaluation pass and aggregate metrics.

    Args:
        model: the model in eval mode (will be set inside this function).
        batches: an iterable yielding either (x, y) or (x, y, mask), depending
            on `has_mask`. Typically `dataset.iter_val_batches(...)` or
            `dataset.iter_batches(...)`.
        device: device for model inputs.
        has_mask: if True, batches yield 3-tuples and masked metrics are
            computed. If False, masked fields will be NaN.
        mask_positions_per_sequence: if `has_mask` is True, the number of
            True entries expected per sequence in `mask` (5 for both copy
            and induction). Used to build the per-position breakdown. If
            None, the breakdown is skipped.

    Returns:
        EvalResult.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_masked_loss = 0.0
    total_masked_correct = 0
    total_masked_tokens = 0

    # For the per-position breakdown: accumulate sum of loss at position k
    # (k = 0..mask_positions_per_sequence-1) and a count, then divide at end.
    pos_sums: Optional[torch.Tensor] = None
    pos_counts: Optional[torch.Tensor] = None
    if has_mask and mask_positions_per_sequence is not None:
        pos_sums = torch.zeros(mask_positions_per_sequence, dtype=torch.float64)
        pos_counts = torch.zeros(mask_positions_per_sequence, dtype=torch.long)

    for batch in batches:
        if has_mask:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        logits = model(x)
        loss_per_tok = _cross_entropy_per_token(logits, y)  # [B, T]

        # Aggregate over all positions.
        total_loss += loss_per_tok.sum().item()
        total_tokens += loss_per_tok.numel()

        if mask is not None:
            masked_losses = loss_per_tok[mask]  # 1D
            total_masked_loss += masked_losses.sum().item()
            total_masked_tokens += masked_losses.numel()

            preds = logits.argmax(dim=-1)  # [B, T]
            correct = (preds == y) & mask
            total_masked_correct += int(correct.sum().item())

            # Per-position breakdown: for each row, the True positions in
            # `mask` are assumed to appear in the same order across the
            # batch (e.g. all 5 recall positions in copy follow "recall: ").
            # We extract them, reshape to [B, K], and accumulate column-wise.
            if pos_sums is not None:
                B = mask.shape[0]
                K = mask.sum(dim=1)  # per-row count
                # Sanity: every sequence must have exactly K masked positions.
                if not torch.all(K == mask_positions_per_sequence):
                    # Either the mask is malformed or the assumption is wrong.
                    # Fail loudly — silent miscounting here would corrupt the
                    # per-position table that feeds Plot 2.
                    raise ValueError(
                        f"expected {mask_positions_per_sequence} masked positions "
                        f"per sequence; saw counts in {K.unique().tolist()}"
                    )
                # masked_losses currently flattens row-major; reshape to [B, K].
                per_row = masked_losses.view(B, mask_positions_per_sequence)
                pos_sums += per_row.sum(dim=0).double().cpu()
                pos_counts += B

    val_loss = total_loss / max(total_tokens, 1)
    val_ppl = float(torch.tensor(val_loss).exp().item())

    if total_masked_tokens > 0:
        masked_loss = total_masked_loss / total_masked_tokens
        masked_ppl = float(torch.tensor(masked_loss).exp().item())
        masked_acc = total_masked_correct / total_masked_tokens
        if pos_sums is not None and pos_counts is not None:
            per_pos = (pos_sums / pos_counts.clamp(min=1)).tolist()
        else:
            per_pos = list(_NO_BREAKDOWN)
    else:
        masked_loss = float("nan")
        masked_ppl = float("nan")
        masked_acc = float("nan")
        per_pos = list(_NO_BREAKDOWN)

    return EvalResult(
        val_loss=val_loss,
        val_ppl=val_ppl,
        n_tokens=total_tokens,
        masked_loss=masked_loss,
        masked_ppl=masked_ppl,
        masked_acc=masked_acc,
        n_masked_tokens=total_masked_tokens,
        masked_loss_per_position=per_pos,
    )
