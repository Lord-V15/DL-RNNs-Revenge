"""
Inline data generation for synthetic tasks (copy, induction)
Generates data on-the-fly without requiring pre-generated files
"""

import torch
from torch.utils.data import Dataset
import random
import string


class LongRangeCopyDataset(Dataset):
    """
    Generates long-range copy task on-the-fly.
    Format: key: <5 chars> | <N distractors> | recall: <5 chars>
    """
    def __init__(self, n_samples=10000, length='short'):
        self.n_samples = n_samples
        self.length = length

        # Distractor lengths per spec
        distractor_lens = {'short': 100, 'medium': 500, 'long': 2000}
        self.distractor_len = distractor_lens[length]

        # Block sizes (from corrected dataset_utils)
        block_sizes = {'short': 144, 'medium': 544, 'long': 2048}
        self.block_size = block_sizes[length]

        # Vocabulary: 26 lowercase + space + colon + pipe = 29 chars
        self.vocab = " :|" + string.ascii_lowercase
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate key (5 random lowercase letters)
        key = ''.join(random.choices(string.ascii_lowercase, k=5))

        # Generate distractors (ensure no 5-char substring matches key)
        while True:
            distractors = ''.join(random.choices(string.ascii_lowercase, k=self.distractor_len))
            # Check if key appears as substring
            if key not in distractors:
                # Also check all 5-char substrings
                has_match = False
                for i in range(len(distractors) - 4):
                    if distractors[i:i+5] == key:
                        has_match = True
                        break
                if not has_match:
                    break

        # Build sequence
        sequence = f"key: {key} | {distractors} | recall: {key}"

        # Tokenize
        tokens = [self.char_to_idx[ch] for ch in sequence]
        seq_len = len(tokens)

        # Pad to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))  # Pad with space (idx 0)
        else:
            tokens = tokens[:self.block_size]
            seq_len = min(seq_len, self.block_size)

        # Create input/target pairs (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mark recall positions (last 5 characters of the key in recall section)
        # Positions: seq_len-6 to seq_len-1 (5 positions) in input_ids
        recall_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        if seq_len >= 6:
            recall_mask[seq_len-6:seq_len-1] = True

        return input_ids, labels, recall_mask


class InductionDataset(Dataset):
    """
    Generates induction task on-the-fly.
    Format: <M random> <5-char pattern> <M random> <first 4 of pattern>
    Model must predict the 5th character.
    """
    def __init__(self, n_samples=10000, length='short'):
        self.n_samples = n_samples
        self.length = length

        # Distractor lengths per spec (M chars on each side)
        distractor_lens = {'short': 50, 'medium': 200, 'long': 1000}
        self.distractor_len = distractor_lens[length]

        # Block sizes
        block_sizes = {'short': 128, 'medium': 512, 'long': 2048}
        self.block_size = block_sizes[length]

        # Vocabulary: 26 lowercase + space = 27 chars
        self.vocab = " " + string.ascii_lowercase
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate pattern (5 random lowercase letters)
        pattern = ''.join(random.choices(string.ascii_lowercase, k=5))

        # Generate prefix and suffix
        prefix = ''.join(random.choices(self.vocab, k=self.distractor_len))
        suffix = ''.join(random.choices(self.vocab, k=self.distractor_len))

        # Ensure pattern doesn't appear in prefix or suffix
        while pattern in prefix or pattern in suffix:
            prefix = ''.join(random.choices(self.vocab, k=self.distractor_len))
            suffix = ''.join(random.choices(self.vocab, k=self.distractor_len))

        # Build sequence: <prefix> <pattern> <suffix> <full pattern>
        sequence = prefix + pattern + suffix + pattern

        # Tokenize
        tokens = [self.char_to_idx[ch] for ch in sequence]
        seq_len = len(tokens)

        # Pad to block_size
        if len(tokens) < self.block_size:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
            seq_len = min(seq_len, self.block_size)

        # Create input/target pairs
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Mark pattern completion positions (last 5 positions before padding)
        # These are the 4 prompt chars + the predicted 5th char
        pattern_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        if seq_len >= 6:
            pattern_mask[seq_len-6:seq_len-1] = True

        return input_ids, labels, pattern_mask


if __name__ == "__main__":
    print("Testing Long-range Copy dataset...")
    ds = LongRangeCopyDataset(n_samples=5, length='short')
    print(f"Vocab size: {ds.vocab_size}")
    print(f"Block size: {ds.block_size}")

    for i in range(2):
        input_ids, labels, mask = ds[i]
        print(f"\nSample {i}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Recall positions: {mask.nonzero().flatten().tolist()}")

        # Decode a bit
        decoded = ''.join([ds.idx_to_char[idx.item()] for idx in input_ids[:50]])
        print(f"  First 50 chars: {decoded!r}")

    print("\n" + "="*80)
    print("Testing Induction dataset...")
    ds = InductionDataset(n_samples=5, length='short')
    print(f"Vocab size: {ds.vocab_size}")
    print(f"Block size: {ds.block_size}")

    for i in range(2):
        input_ids, labels, mask = ds[i]
        print(f"\nSample {i}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Pattern positions: {mask.nonzero().flatten().tolist()}")

        # Decode last 20 chars (where pattern should be)
        decoded = ''.join([ds.idx_to_char[idx.item()] for idx in input_ids[-20:]])
        print(f"  Last 20 chars: {decoded!r}")
