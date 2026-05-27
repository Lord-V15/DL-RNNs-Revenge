"""TinyShakespeare character-level dataset.

On first use, downloads the 1.1M-character corpus to `data/tinyshakespeare/`.
Subsequent calls read from the local cache.

Train/val split is 90/10 by character position — the first 90% of the file is
train, the last 10% is val. This matches the standard char-LM setup in
Karpathy (2022) and the convention used by Feng et al. (2024) so that our
PPL 4.63 reproduction number is directly comparable.

Batches are drawn by uniform random sampling of starting positions within the
split, returning (x, y) where y is x shifted by one position — the standard
next-token-prediction setup.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Literal

import torch

from src.data.vocab import CharVocab


# Karpathy's char-rnn TinyShakespeare URL. Stable, well-mirrored, no auth.
_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_DEFAULT_DIR = Path("data/tinyshakespeare")
_FILENAME = "input.txt"


def _download_if_needed(data_dir: Path) -> Path:
    """Ensure the corpus is on disk; download to `data_dir/input.txt` if not."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / _FILENAME
    if not path.exists():
        # Print rather than logging — this happens once, on first run.
        print(f"[tinyshakespeare] downloading corpus from {_URL}")
        urllib.request.urlretrieve(_URL, path)
        print(f"[tinyshakespeare] saved to {path}")
    return path


class TinyShakespeareDataset:
    """In-memory char-level TinyShakespeare with train/val splits and a
    random-window batch sampler.

    Usage:
        ds = TinyShakespeareDataset(block_size=256)
        x, y = ds.get_batch("train", batch_size=64, device="cuda")

    `x` and `y` are int64 tensors of shape [batch_size, block_size]; `y` is
    `x` shifted by one position. Standard next-token prediction.

    The full corpus is encoded once at construction time and held in CPU
    memory as two int64 tensors (~9 MB total — trivial).
    """

    def __init__(
        self,
        block_size: int,
        data_dir: Path | str = _DEFAULT_DIR,
        train_frac: float = 0.9,
    ):
        data_dir = Path(data_dir)
        path = _download_if_needed(data_dir)
        text = path.read_text(encoding="utf-8")

        self.vocab = CharVocab.from_corpus(text)
        self.block_size = block_size

        full = self.vocab.encode(text)
        n_train = int(len(full) * train_frac)
        self._train = full[:n_train].contiguous()
        self._val = full[n_train:].contiguous()

        if len(self._train) < block_size + 1:
            raise ValueError(
                f"train split has {len(self._train)} tokens but block_size+1 = "
                f"{block_size + 1}. Lower block_size or check the corpus."
            )
        if len(self._val) < block_size + 1:
            raise ValueError(
                f"val split has {len(self._val)} tokens but block_size+1 = "
                f"{block_size + 1}. Lower block_size or check the corpus."
            )

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    def split_data(self, split: Literal["train", "val"]) -> torch.Tensor:
        if split == "train":
            return self._train
        if split == "val":
            return self._val
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'val'")

    def get_batch(
        self,
        split: Literal["train", "val"],
        batch_size: int,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of random windows from the given split.

        Args:
            split: "train" or "val".
            batch_size: number of windows.
            device: target device for the returned tensors.
            generator: optional torch.Generator for reproducible sampling.

        Returns:
            x, y: int64 tensors of shape [batch_size, block_size]. y is x
                shifted right by one position.
        """
        data = self.split_data(split)
        max_start = len(data) - self.block_size - 1  # inclusive upper bound
        # randint upper bound is exclusive, so + 1.
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(batch_size,),
            generator=generator,
        )
        x = torch.stack([data[s : s + self.block_size] for s in starts])
        y = torch.stack([data[s + 1 : s + 1 + self.block_size] for s in starts])
        # pin_memory + non_blocking transfer is a meaningful win on H100s;
        # we keep the simple version here and let the trainer decide.
        return x.to(device), y.to(device)

    def iter_val_batches(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        stride: int | None = None,
    ):
        """Deterministic iterator over the val split with non-overlapping
        (or strided) windows.

        Used for evaluation: random sampling gives noisy val PPL across
        evaluations, whereas a fixed sweep over val gives a stable estimate.

        Args:
            batch_size: number of windows per yielded batch.
            device: target device.
            stride: distance between consecutive window starts. Defaults to
                `block_size` (non-overlapping). Use a smaller stride for a
                denser but slower estimate.

        Yields:
            (x, y) batches as in `get_batch`. The final batch may be smaller
            than `batch_size`.
        """
        if stride is None:
            stride = self.block_size
        data = self._val
        starts = list(range(0, len(data) - self.block_size - 1, stride))
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            x = torch.stack([data[s : s + self.block_size] for s in chunk])
            y = torch.stack([data[s + 1 : s + 1 + self.block_size] for s in chunk])
            yield x.to(device), y.to(device)
