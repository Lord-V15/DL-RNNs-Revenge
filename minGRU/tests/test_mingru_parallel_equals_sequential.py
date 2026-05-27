"""Verify the log-space parallel scan equals the naive sequential recurrence.

This is THE most important test in the repo. If the parallel scan is wrong,
every training run is invalid — minGRU is forward-passed via the scan, so a
bug here corrupts every gradient.

We compare two ways of computing the same minGRU forward pass:
  1. The parallel scan in `MinGRU.forward` (log-space, batched over time).
  2. A naive Python loop applying the recurrence h_t = (1-z_t)*h_{t-1} + z_t*h_tilde_t
     one step at a time.

They must agree to within fp32 tolerance. We test:
  * Short sequences (T=8) — basic correctness.
  * Long sequences (T=2048) — the regime where naive log-space matters.
  * Multi-batch (B=4) — confirms no cross-sample leakage.
  * Gradients — the scan must be autograd-compatible end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.models.mingru import MinGRU, _g  # noqa: E402


def _sequential_forward(gru: MinGRU, x: torch.Tensor) -> torch.Tensor:
    """Apply minGRU's recurrence step-by-step in Python.

    Reimplements the math of MinGRU.forward without using the parallel scan:
        h_t = (1 - sigmoid(z_logits_t)) * h_{t-1} + sigmoid(z_logits_t) * g(h_tilde_pre_t)
        y_t = out_proj(h_t)
    with h_0 = 0.

    Used as the ground-truth reference in correctness tests. Runs in fp32.
    """
    B, T, D = x.shape
    z_logits, h_tilde_pre = gru.proj(x.float()).chunk(2, dim=-1)
    z = torch.sigmoid(z_logits)
    h_tilde = _g(h_tilde_pre)
    h = torch.zeros(B, gru.d_inner, dtype=torch.float32, device=x.device)
    outs = []
    for t in range(T):
        h = (1.0 - z[:, t]) * h + z[:, t] * h_tilde[:, t]
        outs.append(h)
    stacked = torch.stack(outs, dim=1)
    return gru.out_proj(stacked.to(x.dtype))


def _make_gru(d_model: int = 16, seed: int = 0) -> MinGRU:
    """Build a small minGRU with deterministic weights."""
    torch.manual_seed(seed)
    gru = MinGRU(d_model=d_model, expansion_factor=1.0)
    gru.eval()
    return gru


def test_short_sequence_matches_sequential():
    """T=8: simplest case, mostly catches setup bugs."""
    gru = _make_gru()
    x = torch.randn(2, 8, 16)
    y_parallel = gru(x)
    y_sequential = _sequential_forward(gru, x)
    max_diff = (y_parallel - y_sequential).abs().max().item()
    assert max_diff < 1e-4, f"short-seq parallel vs sequential differ by {max_diff:.2e}"


def test_long_sequence_matches_sequential():
    """T=2048: the regime where naive (non-log-space) scan would underflow.

    Tolerance loosened slightly vs the short-seq test because logcumsumexp
    over 2048 entries accumulates a bit more error than over 8. 1e-3 is
    still tight enough to catch any actual bug (real bugs typically produce
    differences in the 0.1+ range).
    """
    gru = _make_gru()
    x = torch.randn(2, 2048, 16)
    y_parallel = gru(x)
    y_sequential = _sequential_forward(gru, x)
    max_diff = (y_parallel - y_sequential).abs().max().item()
    assert max_diff < 1e-3, f"long-seq parallel vs sequential differ by {max_diff:.2e}"


def test_batched_no_crosstalk():
    """B>1: confirm the scan doesn't mix information across batch elements.

    Run two batches: (1) [a, b] together, (2) [a] alone and [b] alone, then
    compare. They must match exactly except for floating point.
    """
    gru = _make_gru()
    torch.manual_seed(1)
    a = torch.randn(1, 64, 16)
    b = torch.randn(1, 64, 16)
    together = gru(torch.cat([a, b], dim=0))
    alone_a = gru(a)
    alone_b = gru(b)
    diff_a = (together[0:1] - alone_a).abs().max().item()
    diff_b = (together[1:2] - alone_b).abs().max().item()
    assert diff_a < 1e-5, f"batch crosstalk on sample a: {diff_a:.2e}"
    assert diff_b < 1e-5, f"batch crosstalk on sample b: {diff_b:.2e}"


def test_gradient_flows_through_scan():
    """The scan must be autograd-compatible.

    A common bug in hand-rolled scans is detaching at some point, which
    silently zeros gradients to upstream parameters. We backprop a simple
    sum and assert every Parameter has a finite, non-zero gradient.
    """
    gru = _make_gru()
    gru.train()
    x = torch.randn(2, 32, 16, requires_grad=False)
    y = gru(x)
    loss = y.sum()
    loss.backward()
    for name, p in gru.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
        # The proj and out_proj layers must see non-zero grads; if one of
        # them is exactly zero we've probably wired the scan wrong.
        assert p.grad.abs().max().item() > 0, f"zero gradient for {name}"


def test_step_matches_forward_at_each_position():
    """MinGRU.step (sequential, used for generation) must match MinGRU.forward
    position-by-position.

    The scan and step paths share the recurrence math but use different
    code paths. A bug in only one would diverge here.
    """
    gru = _make_gru(d_model=8)
    torch.manual_seed(2)
    x = torch.randn(1, 16, 8)
    y_forward = gru(x)

    h = None
    y_steps = []
    for t in range(x.shape[1]):
        y_t, h = gru.step(x[:, t], h)
        y_steps.append(y_t)
    y_stepwise = torch.stack(y_steps, dim=1)

    max_diff = (y_forward - y_stepwise).abs().max().item()
    assert max_diff < 1e-4, f"forward vs step diverge: {max_diff:.2e}"


def main() -> int:
    """Run all tests sequentially and print a compact report.

    Doesn't depend on pytest — each test function is a top-level def, and we
    just call them in order. Any AssertionError propagates and shows the
    failed assertion's message. pytest can also discover and run them.
    """
    tests = [
        test_short_sequence_matches_sequential,
        test_long_sequence_matches_sequential,
        test_batched_no_crosstalk,
        test_gradient_flows_through_scan,
        test_step_matches_forward_at_each_position,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
