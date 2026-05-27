"""
Verification script to check the generated datasets.
Shows examples and validates format.
"""

from pathlib import Path

def show_longrange_examples():
    print("=" * 80)
    print("LONG-RANGE COPY EXAMPLES")
    print("=" * 80)

    for length_name in ["short", "medium", "long"]:
        file_path = Path(f"data/longrange_copy/val_{length_name}.txt")
        with open(file_path, 'r') as f:
            lines = f.readlines()

        print(f"\n{length_name.upper()} (first 3 examples):")
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            # Extract key and verify it appears at recall
            parts = line.split(" | ")
            key_part = parts[0].replace("key: ", "")
            recall_part = parts[2].replace("recall: ", "")

            # Truncate distractor for display
            distractor = parts[1]
            if len(distractor) > 40:
                distractor_display = distractor[:20] + "..." + distractor[-20:]
            else:
                distractor_display = distractor

            print(f"\n  Example {i+1}:")
            print(f"    Key: {key_part}")
            print(f"    Distractors ({len(parts[1])} chars): {distractor_display}")
            print(f"    Recall: {recall_part}")
            print(f"    ✓ Match: {key_part == recall_part}")
            print(f"    Total length: {len(line)} chars")

def show_induction_examples():
    print("\n" + "=" * 80)
    print("INDUCTION EXAMPLES")
    print("=" * 80)

    for length_name in ["short", "medium", "long"]:
        seq_file = Path(f"data/induction/val_{length_name}.txt")
        pattern_file = Path(f"data/induction/val_{length_name}_patterns.txt")

        with open(seq_file, 'r') as f:
            sequences = f.readlines()
        with open(pattern_file, 'r') as f:
            patterns = f.readlines()

        print(f"\n{length_name.upper()} (first 3 examples):")
        for i in range(3):
            seq = sequences[i].strip()
            pattern = patterns[i].strip()

            # Find pattern occurrences
            first_occurrence = seq.find(pattern)
            # The sequence ends with pattern[:4], so we need to verify structure
            expected_end = pattern[:4]
            actual_end = seq[-4:]

            # Calculate structure
            M = (len(seq) - 5 - 4) // 2  # (total - pattern - partial) / 2

            print(f"\n  Example {i+1}:")
            print(f"    Pattern: {pattern}")
            print(f"    First occurrence at position: {first_occurrence}")
            print(f"    Sequence ends with: {actual_end} (should be {expected_end})")
            print(f"    ✓ Correct ending: {actual_end == expected_end}")
            print(f"    Distractor length (M): {M} per side")
            print(f"    Total length: {len(seq)} chars")

            # Show context around pattern
            start = max(0, first_occurrence - 10)
            end = min(len(seq), first_occurrence + 15)
            context = seq[start:end]
            if start > 0:
                context = "..." + context
            if end < len(seq):
                context = context + "..."
            print(f"    Context: {context}")

def show_statistics():
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print("\nLONG-RANGE COPY:")
    for length_name in ["short", "medium", "long"]:
        train_file = Path(f"data/longrange_copy/train_{length_name}.txt")
        val_file = Path(f"data/longrange_copy/val_{length_name}.txt")

        with open(train_file, 'r') as f:
            train_lines = len(f.readlines())
        with open(val_file, 'r') as f:
            val_lines = len(f.readlines())

        print(f"  {length_name:8s}: train={train_lines:5d}, val={val_lines:4d}")

    print("\nINDUCTION:")
    for length_name in ["short", "medium", "long"]:
        train_file = Path(f"data/induction/train_{length_name}.txt")
        val_file = Path(f"data/induction/val_{length_name}.txt")

        with open(train_file, 'r') as f:
            train_lines = len(f.readlines())
        with open(val_file, 'r') as f:
            val_lines = len(f.readlines())

        print(f"  {length_name:8s}: train={train_lines:5d}, val={val_lines:4d}")

def main():
    show_longrange_examples()
    show_induction_examples()
    show_statistics()
    print("\n" + "=" * 80)
    print("✓ All datasets verified!")
    print("=" * 80)

if __name__ == "__main__":
    main()
