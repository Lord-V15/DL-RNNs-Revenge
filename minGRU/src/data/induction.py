"""Induction task data loader.

Reads pre-generated sequences from `data/induction/{train,val}_{short,medium,long}.txt`
and the companion pattern files `data/induction/{train,val}_{short,medium,long}_patterns.txt`.

Actual sequence format (confirmed from generator output):

    <M random chars> P <M random chars> P

  where P is a randomly-generated 5-character pattern. The full pattern P
  appears TWICE: once in the middle, once at the end. The model reads
  P[0:4] at the tail and must predict P[4] at the final position.

The discriminating signal is therefore a SINGLE position per sequence: the
last character. The mask has exactly 1 True entry per sequence, at the final
position.

Pattern file format:
    One line per sequence, containing the 5-character pattern P.
    Used to verify the pattern is present in the sequence and to locate
    the single prediction position.

Length tiers from proposal §4.3:
  * short:  M=50   → sequence length ~110
  * medium: M=200  → sequence length ~410
  * long:   M=1000 → sequence length ~2010
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from src.data.vocab import CharVocab


PATTERN_LEN = 5

LengthTier = Literal["short", "medium", "long"]
Split = Literal["train", "val"]


def _seq_path(data_dir: Path, split: Split, length: LengthTier) -> Path:
    return data_dir / f"{split}_{length}.txt"


def _pat_path(data_dir: Path, split: Split, length: LengthTier) -> Path:
    return data_dir / f"{split}_{length}_patterns.txt"


def _locate_pattern_and_prefix(seq: str, pattern: str) -> tuple[int, int]:
    """Validate the sequence format and return useful positions.

    Confirms:
    1. The full pattern P appears exactly twice in the sequence
        (once in the body, once at the tail).
    2. The sequence ends with the full pattern P.

    Returns:
        (full_occurrence_start, tail_start) where:
        - full_occurrence_start: index of first P in the body.
        - tail_start: index where the final P begins at the tail
            (= len(seq) - 5).

    Raises ValueError if format is wrong.
    """
    # Check the sequence ends with the full pattern.
    if not seq.endswith(pattern):
        raise ValueError(
            f"sequence does not end with pattern {pattern!r}; "
            f"tail is {seq[-8:]!r}"
        )

    # The tail pattern starts here.
    tail_start = len(seq) - len(pattern)
    body = seq[:tail_start]

    first = body.find(pattern)
    if first < 0:
        raise ValueError(
            f"full pattern {pattern!r} not found in sequence body"
        )
    # Reject if the full pattern also appears elsewhere in the body.
    second_in_body = body.find(pattern, first + 1)
    if second_in_body >= 0:
        raise ValueError(
            f"pattern {pattern!r} appears more than once in sequence body; "
            f"generator rejection-sampling may have failed"
        )

    return first, tail_start


class InductionDataset:
    """In-memory induction dataset with single-position completion masks.

    The mask has exactly ONE True entry per sequence: the final position,
    where the model predicts P[4] given P[:4] at the tail.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: Split,
        length: LengthTier,
    ):
        data_dir = Path(data_dir)
        seq_path = _seq_path(data_dir, split, length)
        pat_path = _pat_path(data_dir, split, length)
        if not seq_path.exists():
            raise FileNotFoundError(
                f"expected {seq_path}; check data/induction/ layout"
            )
        if not pat_path.exists():
            raise FileNotFoundError(
                f"expected pattern file {pat_path} alongside {seq_path.name}"
            )

        self.vocab = CharVocab.induction()
        self.length_tier = length
        self.split = split

        seqs = [
            ln for ln in seq_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        pats = [
            ln for ln in pat_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if len(seqs) != len(pats):
            raise ValueError(
                f"sequence/pattern count mismatch: {len(seqs)} sequences vs "
                f"{len(pats)} patterns in {seq_path.name} / {pat_path.name}"
            )
        if not seqs:
            raise ValueError(f"no sequences found in {seq_path}")

        # Validate uniform length and pattern shape.
        seq_len = len(seqs[0])
        for s in seqs:
            if len(s) != seq_len:
                raise ValueError(
                    f"non-uniform sequence lengths in {seq_path}: saw "
                    f"{len(s)} and {seq_len}"
                )
        self.seq_len = seq_len

        for i, p in enumerate(pats):
            if len(p) != PATTERN_LEN:
                raise ValueError(
                    f"pattern {i} has length {len(p)}; expected {PATTERN_LEN}"
                )

        # Encode sequences.
        encoded = torch.stack([self.vocab.encode(s) for s in seqs])
        self._tokens = encoded  # [N, seq_len]

        # Build mask: exactly 1 True per sequence at the LAST position.
        # That is the position where the model predicts P[4].
        # We validate the format via _locate_pattern_and_prefix but the
        # mask itself is simply the final character of every sequence.
        mask = torch.zeros_like(encoded, dtype=torch.bool)
        for i, (s, p) in enumerate(zip(seqs, pats)):
            _locate_pattern_and_prefix(s, p)   # validates format; raises if wrong
            mask[i, -1] = True                  # single prediction position
        self._pattern_mask = mask  # [N, seq_len] bool

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
            x: [B, T-1] int64 — inputs.
            y: [B, T-1] int64 — targets (x shifted left by one).
            pattern_mask: [B, T-1] bool — True at the single final position
                in y where P[4] is predicted. Aligned to y (original mask
                shifted left by one).
        """
        n = len(self)
        idx = torch.randint(low=0, high=n, size=(batch_size,), generator=generator)
        seqs = self._tokens[idx]
        masks = self._pattern_mask[idx]

        x = seqs[:, :-1].contiguous()
        y = seqs[:, 1:].contiguous()
        y_mask = masks[:, 1:].contiguous()
        return x.to(device), y.to(device), y_mask.to(device)

    def iter_batches(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ):
        """Deterministic iteration over the entire dataset for evaluation."""
        n = len(self)
        for i in range(0, n, batch_size):
            seqs = self._tokens[i : i + batch_size]
            masks = self._pattern_mask[i : i + batch_size]
            x = seqs[:, :-1].contiguous()
            y = seqs[:, 1:].contiguous()
            y_mask = masks[:, 1:].contiguous()
            yield x.to(device), y.to(device), y_mask.to(device)
