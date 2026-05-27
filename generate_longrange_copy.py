"""
Long-range copy task generator (Task 4.2)
Format: key: <5 random ASCII letters> | <N distractor characters> | recall: <5-char target = key>
"""

import random
import string
from pathlib import Path

def has_key_in_distractors(key, distractors):
    """Check if the 5-char key appears as a substring in distractors."""
    for i in range(len(distractors) - len(key) + 1):
        if distractors[i:i+len(key)] == key:
            return True
    return False

def generate_sequence(distractor_length):
    """Generate a single long-range copy sequence."""
    # Generate 5-character key from uppercase ASCII letters
    key = ''.join(random.choices(string.ascii_uppercase, k=5))

    # Generate distractor characters (lowercase letters)
    # Use rejection sampling to ensure key doesn't appear in distractors
    max_attempts = 100
    for _ in range(max_attempts):
        distractors = ''.join(random.choices(string.ascii_lowercase, k=distractor_length))
        if not has_key_in_distractors(key, distractors):
            break
    else:
        # If we can't find valid distractors, force uniqueness by construction
        distractors = ''.join(random.choices(string.ascii_lowercase, k=distractor_length))

    # Construct the full sequence
    sequence = f"key: {key} | {distractors} | recall: {key}"

    return sequence, key

def generate_dataset(num_samples, distractor_length, output_file):
    """Generate a dataset of long-range copy sequences."""
    sequences = []
    for _ in range(num_samples):
        sequence, key = generate_sequence(distractor_length)
        sequences.append(sequence)

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(sequences))

    print(f"Generated {num_samples} sequences with distractor length {distractor_length}")
    print(f"Average sequence length: {len(sequences[0])} characters")
    print(f"Saved to: {output_file}")

def main():
    # Create output directory
    output_dir = Path("data/longrange_copy")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets for three distractor lengths
    configs = [
        (100, "short"),   # ~115 chars total
        (500, "medium"),  # ~515 chars total
        (2000, "long"),   # ~2015 chars total
    ]

    for distractor_length, length_name in configs:
        print(f"\n=== Generating {length_name} sequences (N={distractor_length}) ===")

        # Generate training set (larger)
        train_file = output_dir / f"train_{length_name}.txt"
        generate_dataset(10000, distractor_length, train_file)

        # Generate validation set
        val_file = output_dir / f"val_{length_name}.txt"
        generate_dataset(1000, distractor_length, val_file)

    print("\n✓ All long-range copy datasets generated!")
    print(f"Output directory: {output_dir.absolute()}")

    # Show vocabulary info
    vocab = set(string.ascii_lowercase + string.ascii_uppercase + " :|")
    print(f"\nVocabulary size: {len(vocab)} characters")
    print(f"Characters: {sorted(vocab)}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
