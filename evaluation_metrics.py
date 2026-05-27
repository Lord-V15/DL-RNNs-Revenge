"""
Evaluation metrics for long-range copy and induction tasks.
Shows how to compute the discriminating signals mentioned in the proposal.
"""

import torch
import torch.nn.functional as F
from typing import Dict
from dataset_utils import get_longrange_dataloader, get_induction_dataloader


def compute_longrange_metrics(logits: torch.Tensor, labels: torch.Tensor,
                               recall_positions: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for long-range copy task.

    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        recall_positions: Boolean mask for recall positions (batch_size, seq_len)

    Returns:
        Dictionary with 'overall_ppl' and 'recall_ppl'
    """
    # Overall perplexity
    overall_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='mean'
    )
    overall_ppl = torch.exp(overall_loss).item()

    # Recall-position perplexity (the discriminating signal)
    if recall_positions.sum() > 0:
        recall_logits = logits[recall_positions]
        recall_labels = labels[recall_positions]
        recall_loss = F.cross_entropy(recall_logits, recall_labels, reduction='mean')
        recall_ppl = torch.exp(recall_loss).item()
    else:
        recall_ppl = float('inf')

    return {
        'overall_ppl': overall_ppl,
        'recall_ppl': recall_ppl,  # This is the key metric!
    }


def compute_induction_metrics(logits: torch.Tensor, labels: torch.Tensor,
                               pattern_positions: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for induction task.

    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        pattern_positions: Boolean mask for pattern completion positions (batch_size, seq_len)

    Returns:
        Dictionary with 'overall_ppl', 'pattern_ppl', and 'pattern_accuracy'
    """
    # Overall perplexity
    overall_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='mean'
    )
    overall_ppl = torch.exp(overall_loss).item()

    # Pattern completion metrics (the discriminating signals)
    if pattern_positions.sum() > 0:
        pattern_logits = logits[pattern_positions]
        pattern_labels = labels[pattern_positions]

        # Perplexity at pattern positions
        pattern_loss = F.cross_entropy(pattern_logits, pattern_labels, reduction='mean')
        pattern_ppl = torch.exp(pattern_loss).item()

        # Exact-match accuracy at pattern positions
        pattern_preds = pattern_logits.argmax(dim=-1)
        pattern_accuracy = (pattern_preds == pattern_labels).float().mean().item()
    else:
        pattern_ppl = float('inf')
        pattern_accuracy = 0.0

    return {
        'overall_ppl': overall_ppl,
        'pattern_ppl': pattern_ppl,        # Key metric 1
        'pattern_accuracy': pattern_accuracy,  # Key metric 2
    }


def example_evaluation():
    """
    Example showing how to use these metrics in a training/evaluation loop.
    """
    print("=" * 80)
    print("EVALUATION METRICS DEMO")
    print("=" * 80)

    # Example 1: Long-range copy
    print("\n1. LONG-RANGE COPY METRICS")
    print("-" * 80)

    dataloader, tokenizer = get_longrange_dataloader('short', 'val', batch_size=4)
    batch = next(iter(dataloader))

    # Simulate model predictions (random logits for demo)
    batch_size, seq_len = batch['input_ids'].shape
    vocab_size = tokenizer.vocab_size
    fake_logits = torch.randn(batch_size, seq_len, vocab_size)

    metrics = compute_longrange_metrics(
        fake_logits,
        batch['labels'],
        batch['recall_positions']
    )

    print(f"\nOverall PPL: {metrics['overall_ppl']:.2f}")
    print(f"Recall PPL:  {metrics['recall_ppl']:.2f}  ← Key discriminating metric!")
    print("\nInterpretation:")
    print("  - Overall PPL measures general language modeling")
    print("  - Recall PPL measures retrieval capability specifically")
    print("  - Lower recall PPL = better long-range memory")
    print("  - We expect: Transformer < Causal gMLP < minGRU (at long distances)")

    # Example 2: Induction
    print("\n" + "=" * 80)
    print("2. INDUCTION METRICS")
    print("-" * 80)

    dataloader, tokenizer = get_induction_dataloader('short', 'val', batch_size=4)
    batch = next(iter(dataloader))

    # Simulate model predictions
    batch_size, seq_len = batch['input_ids'].shape
    vocab_size = tokenizer.vocab_size
    fake_logits = torch.randn(batch_size, seq_len, vocab_size)

    metrics = compute_induction_metrics(
        fake_logits,
        batch['labels'],
        batch['pattern_positions']
    )

    print(f"\nOverall PPL:       {metrics['overall_ppl']:.2f}")
    print(f"Pattern PPL:       {metrics['pattern_ppl']:.2f}  ← Key metric 1!")
    print(f"Pattern Accuracy:  {metrics['pattern_accuracy']:.1%}  ← Key metric 2!")
    print("\nInterpretation:")
    print("  - Overall PPL measures general language modeling")
    print("  - Pattern PPL measures in-context learning specifically")
    print("  - Pattern Accuracy measures exact pattern completion success")
    print("  - We expect: Transformer > Causal gMLP > minGRU")
    print("  - Transformers have 'induction heads' for this exact capability")

    print("\n" + "=" * 80)
    print("INTEGRATION INTO TRAINING LOOP")
    print("=" * 80)
    print("""
# During validation, compute both overall and discriminating metrics:

for batch in val_dataloader:
    with torch.no_grad():
        logits = model(batch['input_ids'])

        # For long-range copy:
        metrics = compute_longrange_metrics(
            logits, batch['labels'], batch['recall_positions']
        )
        # Log both metrics, but focus on recall_ppl for comparisons

        # For induction:
        metrics = compute_induction_metrics(
            logits, batch['labels'], batch['pattern_positions']
        )
        # Log all three, but pattern_ppl and pattern_accuracy are key

# Report in tables:
# Table 1: Overall PPL per (paradigm × task × length)
# Table 2: Discriminating metrics (recall_ppl for copy, pattern_ppl/acc for induction)
    """)


def show_hypothesis_testing():
    """Show how discriminating metrics test the hypotheses."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING WITH DISCRIMINATING METRICS")
    print("=" * 80)

    print("""
H2 — Long-range retrieval favours attention
  Test using: recall_ppl at different distractor lengths
  Expected ranking: Transformer < Causal gMLP < minGRU
  Length effect: Gap should widen as N increases (100 → 500 → 2000)

H3 — In-context pattern continuation favours attention
  Test using: pattern_ppl and pattern_accuracy
  Expected ranking: Transformer > Causal gMLP ≈ minGRU
  Length effect: Transformer should maintain high accuracy, others degrade

Why discriminating metrics matter:
  - Overall PPL mixes all capabilities together
  - Discriminating metrics isolate the specific capability we're testing
  - Example: A model might have good overall PPL but fail at retrieval
  - The recall/pattern positions are where paradigm differences emerge

Smoke-test gate (Day 3-4):
  - If all three paradigms have recall_ppl within 5% → task doesn't discriminate
  - If all three have >95% pattern_accuracy → task is too easy
  - Redesign task parameters before committing compute to full runs
    """)


if __name__ == "__main__":
    example_evaluation()
    show_hypothesis_testing()

    print("\n" + "=" * 80)
    print("✓ Evaluation metrics ready for use!")
    print("=" * 80)
