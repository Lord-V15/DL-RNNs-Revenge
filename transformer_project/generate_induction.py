"""
Induction task generator (Task 4.3) — CORRECTED.
Format: <M random chars> <5-char pattern P> <M random chars> <full 5-char pattern P>
The second occurrence of P is appended in FULL. In autoregressive training the
model reads the first 4 chars of the second P (P[0:4]) and must predict the 5th
char P[4] at the final position — the genuine induction step (recall P[4] from
the FIRST occurrence of P earlier in the sequence).

NOTE ON THE BUG THIS FIXES:
The previous version appended only pattern[:4]. That meant the sequence ended at
P[3], and there was no P[4] token to predict against — the discriminating metric
ended up scoring the model on predicting the PROMPT characters (P[0:4]), which
are not the induction target. With the full pattern appended, the final token IS
P[4], so the model is correctly scored on the induction step.
"""
import random
import string
from pathlib import Path


def generate_random_chars(length, vocab):
    return ''.join(random.choices(vocab, k=length))


def pattern_appears_in_text(pattern, text):
    return pattern in text


def generate_sequence(distractor_length):
    """
    Format: <M random> <P (5)> <M random> <P (5)>
    The model reads the first 4 chars of the SECOND P and must predict P[4].
    """
    vocab = string.ascii_lowercase + ' '
    max_attempts = 100
    for _ in range(max_attempts):
        pattern = ''.join(random.choices(string.ascii_lowercase, k=5))
        prefix = generate_random_chars(distractor_length, vocab)
        suffix = generate_random_chars(distractor_length, vocab)
        if not pattern_appears_in_text(pattern, prefix) and \
           not pattern_appears_in_text(pattern, suffix):
            break
    else:
        pass

    # Append the FULL pattern (corrected): the final char is P[4], the target.
    sequence = prefix + pattern + suffix + pattern
    return sequence, pattern


def generate_dataset(num_samples, distractor_length, output_file):
    sequences = []
    patterns = []
    for _ in range(num_samples):
        sequence, pattern = generate_sequence(distractor_length)
        sequences.append(sequence)
        patterns.append(pattern)

    with open(output_file, 'w') as f:
        f.write('\n'.join(sequences))

    pattern_file = output_file.parent / (output_file.stem + "_patterns.txt")
    with open(pattern_file, 'w') as f:
        f.write('\n'.join(patterns))

    print(f"Generated {num_samples} sequences with distractor length "
          f"{distractor_length} per side")
    print(f"Average sequence length: {len(sequences[0])} characters")


def main():
    output_dir = Path("data/induction")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [(50, "short"), (200, "medium"), (1000, "long")]
    for distractor_length, length_name in configs:
        print(f"\n=== Generating {length_name} (M={distractor_length} per side) ===")
        generate_dataset(10000, distractor_length, output_dir / f"train_{length_name}.txt")
        generate_dataset(1000, distractor_length, output_dir / f"val_{length_name}.txt")
    print("\nAll induction datasets generated (corrected: full pattern appended).")


if __name__ == "__main__":
    random.seed(42)
    main()
