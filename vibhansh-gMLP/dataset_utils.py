"""
PyTorch dataset utilities for loading long-range copy and induction tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab: str):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.idx_to_char[idx] for idx in tokens])


class LongRangeCopyDataset(Dataset):
    """Dataset for long-range copy task."""

    def __init__(self, file_path: str, block_size: int = 128):
        self.file_path = Path(file_path)
        self.block_size = block_size

        # Vocabulary for long-range copy (55 chars: 26 lower + 26 upper + space + colon + pipe)
        vocab = " :ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz"
        self.tokenizer = CharacterTokenizer(vocab)

        with open(self.file_path, 'r') as f:
            self.sequences = [ln for ln in f.read().split('\n') if ln]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        tokens = self.tokenizer.encode(sequence)
        seq_len = len(tokens)

        # Pad or truncate to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
            seq_len = min(seq_len, self.block_size)

        # input/target shifted by one
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Recall positions:
        # Format: "key: KEY | distractors | recall: KEY"
        # The 5 recall chars occupy tokens[seq_len-5 : seq_len].
        # To predict tokens[i], the model uses input_ids[i-1]. So the positions
        # to mark in input_ids are seq_len-6, seq_len-5, ..., seq_len-2 (5 positions).
        recall_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        if seq_len >= 6:
            start = seq_len - 6
            end = seq_len - 1  # exclusive
            recall_positions[start:end] = True

        return {
            'input_ids': input_ids,
            'labels': labels,
            'recall_positions': recall_positions,
            'seq_len': seq_len,
        }


class InductionDataset(Dataset):
    """Dataset for induction task."""

    def __init__(self, file_path: str, pattern_file: Optional[str] = None, block_size: int = 128):
        self.file_path = Path(file_path)
        self.block_size = block_size

        # Vocabulary for induction (27 chars: 26 lowercase + space)
        vocab = " abcdefghijklmnopqrstuvwxyz"
        self.tokenizer = CharacterTokenizer(vocab)

        with open(self.file_path, 'r') as f:
            self.sequences = [ln for ln in f.read().split('\n') if ln]

        self.patterns = None
        if pattern_file:
            with open(pattern_file, 'r') as f:
                self.patterns = [ln for ln in f.read().split('\n') if ln]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        tokens = self.tokenizer.encode(sequence)
        seq_len = len(tokens)

        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
            seq_len = min(seq_len, self.block_size)

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Pattern-completion positions:
        # Format: <prefix> <pattern P (5 chars)> <suffix> <P[:4] (4 chars)>
        # The full sequence ends with P[:4], length seq_len. The model must
        # predict the 5th char of P at the final position. Proposal §7 also
        # asks for PPL at "the 5 pattern-completion positions" — the 4 prompt
        # chars plus the final predicted char.
        #
        # The 4 prompt chars are tokens[seq_len-4 : seq_len-1] (P[0:3], known)
        # plus the predicted char at "the position after seq_len-1" — i.e. the
        # final label is tokens[seq_len-1] = P[4]. To mark these 5 label
        # positions in input_ids (which is tokens[:-1]), we use indices
        # seq_len-6 through seq_len-2.
        pattern_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        if seq_len >= 6:
            start = seq_len - 6
            end = seq_len - 1  # exclusive
            pattern_positions[start:end] = True

        result = {
            'input_ids': input_ids,
            'labels': labels,
            'pattern_positions': pattern_positions,
            'seq_len': seq_len,
        }

        if self.patterns:
            result['pattern'] = self.patterns[idx]

        return result


def get_longrange_dataloader(
    length: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, CharacterTokenizer]:
    # Bumped from proposal {128, 528, 2048} to accommodate the actual sequence
    # lengths {129, 529, 2029}. Without this, the final 'N' of the recall key
    # gets truncated and the discriminating metric measures the wrong thing.
    block_sizes = {'short': 144, 'medium': 544, 'long': 2048}
    block_size = block_sizes[length]
    file_path = f"data/longrange_copy/{split}_{length}.txt"
    dataset = LongRangeCopyDataset(file_path, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset.tokenizer


def get_induction_dataloader(
    length: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, CharacterTokenizer]:
    # Induction sequences are 109/409/2009 chars; proposal block sizes
    # {128, 512, 2048} all fit with padding. Kept as-is.
    block_sizes = {'short': 128, 'medium': 512, 'long': 2048}
    block_size = block_sizes[length]
    file_path = f"data/induction/{split}_{length}.txt"
    pattern_file = f"data/induction/{split}_{length}_patterns.txt"
    dataset = InductionDataset(file_path, pattern_file=pattern_file, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset.tokenizer


if __name__ == "__main__":
    # Quick verification that the masks land on real content, not padding
    print("=" * 80)
    print("Verifying recall_positions mask lands on real recall chars")
    print("=" * 80)
    ds = LongRangeCopyDataset("data/longrange_copy/val_short.txt", block_size=128)
    item = ds[0]
    seq_len = item['seq_len']
    marked = item['recall_positions'].nonzero().flatten().tolist()
    print(f"seq_len: {seq_len}, block_size: 128, marked positions: {marked}")
    # Show what's at the marked positions
    itos = ds.tokenizer.idx_to_char
    print("Marked positions in (input_ids -> labels):")
    for i in marked:
        in_ch = itos[item['input_ids'][i].item()]
        out_ch = itos[item['labels'][i].item()]
        print(f"  position {i}: input={in_ch!r} -> predict={out_ch!r}")
    print(f"Sequence (first 50): {ds.sequences[0][:50]!r}")
    print(f"Sequence (last 30):  ...{ds.sequences[0][-30:]!r}")

    print()
    print("=" * 80)
    print("Verifying pattern_positions mask lands on real pattern chars")
    print("=" * 80)
    ds = InductionDataset(
        "data/induction/val_short.txt",
        pattern_file="data/induction/val_short_patterns.txt",
        block_size=128,
    )
    item = ds[0]
    seq_len = item['seq_len']
    marked = item['pattern_positions'].nonzero().flatten().tolist()
    print(f"seq_len: {seq_len}, block_size: 128, marked positions: {marked}")
    itos = ds.tokenizer.idx_to_char
    print("Marked positions in (input_ids -> labels):")
    for i in marked:
        in_ch = itos[item['input_ids'][i].item()]
        out_ch = itos[item['labels'][i].item()]
        print(f"  position {i}: input={in_ch!r} -> predict={out_ch!r}")
    print(f"Pattern: {item['pattern']!r}")
    print(f"Sequence (last 30): ...{ds.sequences[0][-30:]!r}")
