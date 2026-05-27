"""Character-level vocabularies.

A `CharVocab` is a bijection between a fixed set of characters and the integer
ids 0..V-1. It is intentionally minimal — no special tokens, no BPE, no
out-of-vocabulary handling. Any character outside the vocabulary raises
KeyError at encode time, which is the correct behaviour for our setup: the
synthetic datasets are generated from fixed alphabets, and TinyShakespeare's
vocabulary is fully determined by the corpus.

Vocab sizes per the proposal §4:
  * TinyShakespeare: ~65 (derived from the corpus, not hard-coded)
  * Long-range copy: 55 (26 lowercase + 26 uppercase + space + colon + pipe)
  * Induction:       27 (26 lowercase + space)

Note: the long-range copy vocab was originally specified as 29 chars (lowercase
only) but the actual generator produces uppercase 5-letter keys (e.g. "EEUSZ"),
so uppercase A-Z must be included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class CharVocab:
    """A frozen character-to-id mapping.

    Construct via one of the classmethods (`from_corpus`, `long_range_copy`,
    `induction`) rather than instantiating directly, to ensure `chars` is sorted
    and the inverse mapping stays consistent.
    """
    chars: tuple[str, ...]  # ordered; index in tuple == token id

    @property
    def size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.chars)}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_corpus(cls, text: str) -> "CharVocab":
        """Build a vocab from the unique characters in `text`, sorted."""
        return cls(chars=tuple(sorted(set(text))))

    @classmethod
    def from_chars(cls, chars: Iterable[str]) -> "CharVocab":
        """Build a vocab from an explicit, ordered iterable of single chars.

        Order is preserved (not re-sorted). Use this when token ids must match
        a specific layout — for example, when reading a dataset whose ids were
        assigned by the generator.
        """
        chars = tuple(chars)
        assert all(len(c) == 1 for c in chars), "vocab entries must be single chars"
        assert len(set(chars)) == len(chars), "vocab entries must be unique"
        return cls(chars=chars)

    @classmethod
    def long_range_copy(cls) -> "CharVocab":
        """55-character vocab: 26 uppercase + 26 lowercase + space + colon + pipe.

        The distractor span uses lowercase; the 5-character key and recall
        target are uppercase (e.g. "key: EEUSZ | ... | recall: EEUSZ").
        Chars are sorted so ids are stable: space (32) < colon (58) < pipe (124)
        interleave with letters by ASCII order — A-Z (65-90) then a-z (97-122).
        Sorted order: ' '=0, ':'=1, 'A'=2 .. 'Z'=27, '|'=28, 'a'=29 .. 'z'=54.
        """
        chars = tuple(sorted(
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") +
            [" ", ":", "|"]
        ))
        return cls(chars=chars)

    @classmethod
    def induction(cls) -> "CharVocab":
        """27-character vocab: 26 lowercase + space."""
        chars = tuple("abcdefghijklmnopqrstuvwxyz") + (" ",)
        return cls(chars=chars)

    # -- encode / decode ----------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """Encode a string to a 1D int64 tensor of token ids.

        Raises KeyError if `text` contains a character outside the vocab.
        """
        stoi = self.stoi
        try:
            ids = [stoi[c] for c in text]
        except KeyError as e:
            bad = e.args[0]
            raise KeyError(
                f"character {bad!r} not in vocab "
                f"(vocab has {self.size} chars: {''.join(self.chars)!r})"
            ) from None
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode a 1D tensor or list of token ids back to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        itos = self.itos
        return "".join(itos[int(i)] for i in ids)
