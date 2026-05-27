"""
Induction task generator (Task 4.3)
Format: <M random chars> <5-char pattern P> <M random chars> <first 4 chars of P>
The model must predict the 5th character of P at the final position.
"""

import random
import string
from pathlib import Path

def generate_random_chars(length, vocab):
    """Generate random characters from vocabulary."""
    return ''.join(random.choices(vocab, k=length))

def pattern_appears_in_text(pattern, text):
    """Check if pattern appears as substring in text."""
    return pattern in text

def generate_sequence(distractor_length):
    """
    Generate a single induction sequence.
    Format: <M random chars> <5-char pattern P> <M random chars> <first 4 chars of P>

    The model needs to predict the 5th character of P.
    """
    # Vocabulary: 26 lowercase + space (27 total)
    vocab = string.ascii_lowercase + ' '

    # Generate 5-character pattern
    max_attempts = 100
    for _ in range(max_attempts):
        pattern = ''.join(random.choices(string.ascii_lowercase, k=5))

        # Generate prefix random chars
        prefix = generate_random_chars(distractor_length, vocab)

        # Generate suffix random chars
        suffix = generate_random_chars(distractor_length, vocab)

        # Check that pattern doesn't appear in prefix or suffix
        if not pattern_appears_in_text(pattern, prefix) and not pattern_appears_in_text(pattern, suffix):
            break
    else:
        # Fallback: accept what we have
        pass

    # Construct sequence: prefix + pattern + suffix + first 4 chars of pattern
    # The model should predict the 5th char of pattern
    sequence = prefix + pattern + suffix + pattern[:4]

    return sequence, pattern

def generate_dataset(num_samples, distractor_length, output_file):
    """Generate a dataset of induction sequences."""
    sequences = []
    patterns = []

    for _ in range(num_samples):
        sequence, pattern = generate_sequence(distractor_length)
        sequences.append(sequence)
        patterns.append(pattern)

    # Write sequences to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(sequences))

    # Also save patterns for evaluation
    pattern_file = output_file.parent / (output_file.stem + "_patterns.txt")
    with open(pattern_file, 'w') as f:
        f.write('\n'.join(patterns))

    print(f"Generated {num_samples} sequences with distractor length {distractor_length} per side")
    print(f"Average sequence length: {len(sequences[0])} characters")
    print(f"Saved sequences to: {output_file}")
    print(f"Saved patterns to: {pattern_file}")

def main():
    # Create output directory
    output_dir = Path("data/induction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets for three distractor lengths (M = chars on each side)
    configs = [
        (50, "short"),    # ~110 chars total
        (200, "medium"),  # ~410 chars total
        (1000, "long"),   # ~2010 chars total
    ]

    for distractor_length, length_name in configs:
        print(f"\n=== Generating {length_name} sequences (M={distractor_length} per side) ===")

        # Generate training set
        train_file = output_dir / f"train_{length_name}.txt"
        generate_dataset(10000, distractor_length, train_file)

        # Generate validation set
        val_file = output_dir / f"val_{length_name}.txt"
        generate_dataset(1000, distractor_length, val_file)

    print("\n✓ All induction datasets generated!")
    print(f"Output directory: {output_dir.absolute()}")

    # Show vocabulary info
    vocab = string.ascii_lowercase + ' '
    print(f"\nVocabulary size: {len(vocab)} characters")
    print(f"Characters: lowercase a-z + space")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
