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
        """Convert text to list of token indices."""
        return [self.char_to_idx[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token indices back to text."""
        return ''.join([self.idx_to_char[idx] for idx in tokens])


class LongRangeCopyDataset(Dataset):
    """Dataset for long-range copy task."""

    def __init__(self, file_path: str, block_size: int = 128):
        self.file_path = Path(file_path)
        self.block_size = block_size

        # Vocabulary for long-range copy
        vocab = " :ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz"
        self.tokenizer = CharacterTokenizer(vocab)

        # Load sequences
        with open(self.file_path, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]

        # Tokenize
        tokens = self.tokenizer.encode(sequence)

        # Pad or truncate to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]

        # Create input (tokens[:-1]) and target (tokens[1:])
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Find recall positions (last 5 chars of sequence)
        # Format: "key: XXXXX | distractors | recall: XXXXX"
        # We want to mark the last 5 character positions
        recall_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        if len(tokens) >= 5:
            recall_positions[-5:] = True

        return {
            'input_ids': input_ids,
            'labels': labels,
            'recall_positions': recall_positions
        }


class InductionDataset(Dataset):
    """Dataset for induction task."""

    def __init__(self, file_path: str, pattern_file: Optional[str] = None, block_size: int = 128):
        self.file_path = Path(file_path)
        self.block_size = block_size

        # Vocabulary for induction (lowercase + space)
        vocab = " abcdefghijklmnopqrstuvwxyz"
        self.tokenizer = CharacterTokenizer(vocab)

        # Load sequences
        with open(self.file_path, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]

        # Load patterns if provided
        self.patterns = None
        if pattern_file:
            with open(pattern_file, 'r') as f:
                self.patterns = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]

        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        # Capture true sequence length BEFORE padding/truncating
        seq_len = len(tokens)

        # Pad or truncate to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
            seq_len = min(seq_len, self.block_size)

        # Create input and target
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mark pattern completion positions (last 4 chars + the predicted 5th)
        # The last 4 chars are the prompt, position -1 is where we predict the 5th char
        pattern_positions = torch.zeros(len(input_ids), dtype=torch.bool)
        # Mark pattern positions relative to real seq_len, not padding end
        if seq_len >= 6:
            start = seq_len - 6
            end = seq_len - 1  # exclusive
            pattern_positions[start:end] = True

        # induction_target_pos: the single position where model sees P[3]
        # and must predict P[4] — used for pat_acc5 (5th-char-only accuracy)
        induction_target_pos = seq_len - 2  # input_ids[seq_len-2] = P[3], label = P[4]
        induction_target_pos = min(induction_target_pos, len(input_ids) - 1)

        result = {
            'input_ids': input_ids,
            'labels': labels,
            'pattern_positions': pattern_positions,
            'induction_target_pos': induction_target_pos,
            'seq_len': seq_len,
        }

        # Add pattern if available
        if self.patterns:
            result['pattern'] = self.patterns[idx]

        return result


def get_longrange_dataloader(
    length: str,  # 'short', 'medium', or 'long'
    split: str = 'train',  # 'train' or 'val'
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, CharacterTokenizer]:
    """
    Create a DataLoader for long-range copy task.

    Args:
        length: One of 'short' (128), 'medium' (528), or 'long' (2048)
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader and tokenizer
    """
    block_sizes = {'short': 128, 'medium': 528, 'long': 2048}
    block_size = block_sizes[length]

    file_path = f"data/longrange_copy/{split}_{length}.txt"
    dataset = LongRangeCopyDataset(file_path, block_size=block_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader, dataset.tokenizer


def get_induction_dataloader(
    length: str,  # 'short', 'medium', or 'long'
    split: str = 'train',  # 'train' or 'val'
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, CharacterTokenizer]:
    """
    Create a DataLoader for induction task.

    Args:
        length: One of 'short' (128), 'medium' (512), or 'long' (2048)
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader and tokenizer
    """
    block_sizes = {'short': 128, 'medium': 512, 'long': 2048}
    block_size = block_sizes[length]

    file_path = f"data/induction/{split}_{length}.txt"
    pattern_file = f"data/induction/{split}_{length}_patterns.txt"

    dataset = InductionDataset(file_path, pattern_file=pattern_file, block_size=block_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader, dataset.tokenizer


if __name__ == "__main__":
    print("Testing data loaders...\n")

    # Test long-range copy
    print("=" * 80)
    print("LONG-RANGE COPY DATALOADER TEST")
    print("=" * 80)

    dataloader, tokenizer = get_longrange_dataloader('short', 'val', batch_size=2)
    batch = next(iter(dataloader))

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Recall positions shape: {batch['recall_positions'].shape}")

    # Decode first example
    print("\nFirst example (decoded):")
    decoded = tokenizer.decode(batch['input_ids'][0].tolist())
    print(f"  Input: {decoded[:100]}...")
    print(f"  Recall positions marked: {batch['recall_positions'][0].sum().item()} positions")

    # Test induction
    print("\n" + "=" * 80)
    print("INDUCTION DATALOADER TEST")
    print("=" * 80)

    dataloader, tokenizer = get_induction_dataloader('short', 'val', batch_size=2)
    batch = next(iter(dataloader))

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Pattern positions shape: {batch['pattern_positions'].shape}")

    # Decode first example
    print("\nFirst example (decoded):")
    decoded = tokenizer.decode(batch['input_ids'][0].tolist())
    print(f"  Input: {decoded[:100]}...")
    print(f"  Pattern: {batch['pattern'][0]}")
    print(f"  Pattern positions marked: {batch['pattern_positions'][0].sum().item()} positions")

    print("\n" + "=" * 80)
    print("✓ Data loaders working correctly!")
    print("=" * 80)
