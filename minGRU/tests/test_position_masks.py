"""Verify the long-range copy and induction position masks land on the
right characters.

The position masks drive Table 2 of the report (the discriminating-signal
metrics) and feed Plot 2 (per-position breakdown). A silent off-by-one here
would make all the synthetic-task metrics misleading without any error
message — exactly the kind of bug that's hard to catch downstream.

These tests use very small, hand-crafted fixtures where we know exactly
what the mask SHOULD be, then assert character-by-character.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.data.induction import InductionDataset, _locate_second_occurrence  # noqa: E402
from src.data.longrange_copy import LongRangeCopyDataset, _parse_line  # noqa: E402


# ---------------------------------------------------------------------------
# Long-range copy mask helpers (unit-test the parse function directly)
# ---------------------------------------------------------------------------

def test_longrange_parse_locates_recall_start():
    """`_parse_line` should find the start of the 5-char recall target."""
    line = "key: abcde | xxxxxx | recall: abcde"
    parsed, start = _parse_line(line)
    assert parsed == line
    # "recall: " is 8 chars; it starts at index len(line) - 13 (last 5 are target).
    expected = len(line) - 5
    assert start == expected, f"recall start at {start}, expected {expected}"
    # The 5 chars from `start` onward are the target.
    assert line[start:start + 5] == "abcde"


def test_longrange_parse_rejects_malformed():
    """Missing 'recall: ' should raise ValueError, not return silently."""
    try:
        _parse_line("key: abcde | xxxx | abcde")  # missing 'recall:' marker
    except ValueError:
        return
    raise AssertionError("malformed line should raise ValueError")


# ---------------------------------------------------------------------------
# Long-range copy mask end-to-end
# ---------------------------------------------------------------------------

def _write_handcrafted_copy(data_dir: Path) -> None:
    """Two lines with known recall positions for direct verification."""
    data_dir.mkdir(parents=True, exist_ok=True)
    # Each line: "key: ABCDE | xxxxxxxxxx | recall: ABCDE"
    # Lengths: 5 + 5 + 3 + 10 + 3 + 8 + 5 = 39 chars.
    lines = [
        "key: hello | xxxxxxxxxx | recall: hello",
        "key: world | yyyyyyyyyy | recall: world",
    ]
    (data_dir / "train_short.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / "val_short.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_longrange_mask_recall_positions_are_correct():
    """The mask should be True at exactly the 5 recall positions and nowhere else."""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "longrange_copy"
        _write_handcrafted_copy(data_dir)
        ds = LongRangeCopyDataset(data_dir, split="train", length="short")

        # Inspect the original (un-shifted) mask. The 5 recall positions are
        # the last 5 chars of each 39-char line, i.e. indices 34..38.
        mask = ds._recall_mask  # [N, seq_len]
        seq_len = mask.shape[1]
        assert seq_len == 39, f"seq_len should be 39, got {seq_len}"
        for i in range(mask.shape[0]):
            true_idx = mask[i].nonzero().squeeze(-1).tolist()
            assert true_idx == [34, 35, 36, 37, 38], (
                f"row {i}: recall mask at {true_idx}, expected [34, 35, 36, 37, 38]"
            )


def test_longrange_mask_aligns_with_y_after_shift():
    """When get_batch shifts (x = seq[:-1], y = seq[1:]), the returned mask
    is over y. So a mask True at original position k must mean y at position
    k-1 is the recall char. We verify by decoding y[mask] back to chars and
    confirming they spell the keys.
    """
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "longrange_copy"
        _write_handcrafted_copy(data_dir)
        ds = LongRangeCopyDataset(data_dir, split="train", length="short")

        # Use iter_batches for determinism.
        x, y, mask = next(iter(ds.iter_batches(batch_size=2)))
        # y[mask] should yield 10 tokens (2 rows × 5 recall chars).
        recall_tokens = y[mask]
        assert recall_tokens.numel() == 10
        # Decode them back. Since we know row 0 is "hello" and row 1 is
        # "world", the concatenation should be "helloworld".
        decoded = ds.vocab.decode(recall_tokens)
        # The order may differ across rows but each row's 5 chars should
        # appear in order. Easiest: check the set of chars is right and
        # each row's chunk spells correctly.
        per_row = y[mask].view(2, 5)
        assert ds.vocab.decode(per_row[0]) == "hello"
        assert ds.vocab.decode(per_row[1]) == "world"


# ---------------------------------------------------------------------------
# Induction mask
# ---------------------------------------------------------------------------

def test_induction_locate_second_occurrence_basic():
    """_locate_second_occurrence should find the SECOND occurrence of P."""
    seq = "xxxxxhellxxxx"  # 'hell' appears only once — should raise.
    try:
        _locate_second_occurrence(seq, "hell")
    except ValueError:
        pass
    else:
        raise AssertionError("single occurrence should raise ValueError")

    seq = "xxhelloxxhelloxx"  # 'hello' appears twice at positions 2 and 9
    pos = _locate_second_occurrence(seq, "hello")
    assert pos == 9, f"expected second occurrence at 9, got {pos}"


def test_induction_locate_rejects_three_occurrences():
    """Generator bug check: more than two occurrences should raise."""
    seq = "abcabcabc"
    try:
        _locate_second_occurrence(seq, "abc")
    except ValueError:
        return
    raise AssertionError("three-occurrence sequence should raise ValueError")


def _write_handcrafted_induction(data_dir: Path) -> None:
    """Two lines with known pattern positions for direct verification."""
    data_dir.mkdir(parents=True, exist_ok=True)
    # Format: <8 a's> <P=jklmn> <8 b's> <P=jklmn>
    # Length: 8 + 5 + 8 + 5 = 26.
    # Second-P starts at index 8 + 5 + 8 = 21.
    line1 = "aaaaaaaa" + "jklmn" + "bbbbbbbb" + "jklmn"
    line2 = "cccccccc" + "opqrs" + "dddddddd" + "opqrs"
    (data_dir / "train_short.txt").write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")
    (data_dir / "train_short_patterns.txt").write_text("jklmn\nopqrs\n", encoding="utf-8")
    (data_dir / "val_short.txt").write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")
    (data_dir / "val_short_patterns.txt").write_text("jklmn\nopqrs\n", encoding="utf-8")


def test_induction_mask_marks_5_positions_starting_at_second_P():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "induction"
        _write_handcrafted_induction(data_dir)
        ds = InductionDataset(data_dir, split="train", length="short")
        mask = ds._pattern_mask  # [2, 26]
        assert mask.shape == (2, 26)
        for i in range(2):
            true_idx = mask[i].nonzero().squeeze(-1).tolist()
            assert true_idx == [21, 22, 23, 24, 25], (
                f"row {i}: pattern mask at {true_idx}, expected [21, 22, 23, 24, 25]"
            )


def test_induction_mask_decodes_to_pattern_chars():
    """y[mask] should decode to the pattern characters for each row."""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "induction"
        _write_handcrafted_induction(data_dir)
        ds = InductionDataset(data_dir, split="train", length="short")
        # Walk iter_batches deterministically.
        x, y, mask = next(iter(ds.iter_batches(batch_size=2)))
        per_row = y[mask].view(2, 5)
        # The mask in y-coordinates is the original mask shifted left by 1,
        # so y[mask] should give us the LAST FIVE CHARS of each line — which
        # ARE the pattern chars (since the pattern is right at the end).
        assert ds.vocab.decode(per_row[0]) == "jklmn"
        assert ds.vocab.decode(per_row[1]) == "opqrs"


def test_induction_per_row_count_is_5():
    """Sanity: every row in the mask should have exactly 5 True entries."""
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "induction"
        _write_handcrafted_induction(data_dir)
        ds = InductionDataset(data_dir, split="train", length="short")
        per_row = ds._pattern_mask.sum(dim=1)
        assert torch.all(per_row == 5), f"per-row counts: {per_row.tolist()}"


def main() -> int:
    tests = [
        test_longrange_parse_locates_recall_start,
        test_longrange_parse_rejects_malformed,
        test_longrange_mask_recall_positions_are_correct,
        test_longrange_mask_aligns_with_y_after_shift,
        test_induction_locate_second_occurrence_basic,
        test_induction_locate_rejects_three_occurrences,
        test_induction_mask_marks_5_positions_starting_at_second_P,
        test_induction_mask_decodes_to_pattern_chars,
        test_induction_per_row_count_is_5,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except (AssertionError, Exception) as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
