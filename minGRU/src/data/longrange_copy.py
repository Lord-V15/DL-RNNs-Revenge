"""Long-range copy task data loader.

Reads pre-generated sequences from `data/longrange_copy/{train,val}_{short,medium,long}.txt`.

Each line is one sequence in the format specified by proposal §4.2:

    key: ABCDE | <N distractor chars> | recall: ABCDE

The model sees the entire line (encoded character-by-character) and must
predict the next character at each position. The discriminating signal is the
PPL at the 5 final positions — the `ABCDE` after "recall: ". Those positions
are marked by `recall_mask`, a boolean tensor of shape [B, T] that is True
exactly at the 5 target positions.

Length tiers from proposal §4.2:
  * short:  N=100  → sequence length ~115
  * medium: N=500  → sequence length ~515
  * long:   N=2000 → sequence length ~2015

The file is loaded entirely into memory (a few MB at most) and a torch
Dataset-like API is exposed for batching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from src.data.vocab import CharVocab


# Markers that delimit the recall span. The recall target is the 5 characters
# immediately following the final occurrence of `RECALL_PREFIX` on each line.
RECALL_PREFIX = "recall: "
RECALL_LEN = 5

LengthTier = Literal["short", "medium", "long"]
Split = Literal["train", "val"]


def _path_for(data_dir: Path, split: Split, length: LengthTier) -> Path:
    return data_dir / f"{split}_{length}.txt"


def _parse_line(line: str) -> tuple[str, int]:
    """Parse one line, returning (sequence_string, recall_start_index).

    `recall_start_index` is the position (in characters) where the 5-character
    target begins. We locate it as the position immediately after the LAST
    occurrence of "recall: " — using the last occurrence guards against the
    (extremely unlikely) case where the distractor span contains the literal
    substring "recall: ", though the generator's rejection sampling should
    prevent that.

    Raises ValueError if the line is malformed.
    """
    line = line.rstrip("\n")
    idx = line.rfind(RECALL_PREFIX)
    if idx < 0:
        raise ValueError(f"line missing {RECALL_PREFIX!r} marker: {line[:60]!r}...")
    recall_start = idx + len(RECALL_PREFIX)
    if recall_start + RECALL_LEN > len(line):
        raise ValueError(
            f"line ends before {RECALL_LEN}-char recall target: {line[-40:]!r}"
        )
    return line, recall_start


class LongRangeCopyDataset:
    """In-memory long-range copy dataset.

    All sequences in a single file have the same length by construction (the
    generator pads/truncates the distractor span to the tier's target). We
    verify this at load time and store sequences as a single packed
    [N, seq_len] int64 tensor.

    The recall mask is precomputed once per file and reused across batches.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: Split,
        length: LengthTier,
    ):
        data_dir = Path(data_dir)
        path = _path_for(data_dir, split, length)
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path}; check data/longrange_copy/ layout"
            )

        self.vocab = CharVocab.long_range_copy()
        self.length_tier = length
        self.split = split

        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            raise ValueError(f"no sequences found in {path}")

        # Parse and check uniform length.
        parsed = [_parse_line(ln) for ln in lines]
        seq_len = len(parsed[0][0])
        for ln, _ in parsed:
            if len(ln) != seq_len:
                raise ValueError(
                    f"non-uniform sequence lengths in {path}: saw "
                    f"{len(ln)} and {seq_len}"
                )
        self.seq_len = seq_len

        # Encode all sequences. This is a few MB at most.
        encoded = torch.stack([self.vocab.encode(ln) for ln, _ in parsed])
        self._tokens = encoded  # [N, seq_len] int64

        # Build the recall mask. Same for every line in a given file (by
        # construction the generator places the recall target at the same
        # position), but we build it per-row to be robust to any future
        # variation.
        mask = torch.zeros_like(encoded, dtype=torch.bool)
        for i, (_, recall_start) in enumerate(parsed):
            mask[i, recall_start : recall_start + RECALL_LEN] = True
        self._recall_mask = mask  # [N, seq_len] bool

    # -- size / shape -------------------------------------------------------

    def __len__(self) -> int:
        return self._tokens.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    # -- batching -----------------------------------------------------------

    def get_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of sequences.

        Returns:
            x: [B, T-1] int64 — inputs (sequence with the last char dropped).
            y: [B, T-1] int64 — targets (sequence shifted left by one).
            recall_mask: [B, T-1] bool — True at positions in `y` that are
                part of the 5-char recall target. Note this mask is aligned to
                `y`, NOT to the original sequence: position i in `y` is the
                target for predicting from position i in `x`, i.e. original
                position i+1. So the recall mask is the original `_recall_mask`
                shifted left by one and truncated.
        """
        n = len(self)
        idx = torch.randint(low=0, high=n, size=(batch_size,), generator=generator)
        seqs = self._tokens[idx]              # [B, T]
        masks = self._recall_mask[idx]        # [B, T]

        x = seqs[:, :-1].contiguous()
        y = seqs[:, 1:].contiguous()
        # The mask is over original-sequence positions; align to y by dropping
        # the first column (positions that are never targets — they're the
        # very first input).
        y_mask = masks[:, 1:].contiguous()
        return x.to(device), y.to(device), y_mask.to(device)

    def iter_batches(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ):
        """Deterministic iteration over the entire dataset.

        Used for evaluation so val PPL is reproducible across evals. Yields
        (x, y, recall_mask) tuples; the final batch may be smaller.
        """
        n = len(self)
        for i in range(0, n, batch_size):
            seqs = self._tokens[i : i + batch_size]
            masks = self._recall_mask[i : i + batch_size]
            x = seqs[:, :-1].contiguous()
            y = seqs[:, 1:].contiguous()
            y_mask = masks[:, 1:].contiguous()
            yield x.to(device), y.to(device), y_mask.to(device)
