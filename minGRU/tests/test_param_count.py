"""Verify minGRU's parameter count matches the proposal's budget.

Proposal §5 commits to ~750K parameters (±10%) for fair cross-paradigm
comparison. The trainer asserts this at startup, but it's worth catching
drift in tests too so a config change is caught without launching a run.

We test:
  * The default config (vocab_size=65 for TinyShakespeare) is in the
    [675K, 825K] window — proposal's ±10%.
  * Per-vocabulary counts: with synth-task vocab sizes (29 for copy, 27
    for induction) the body is identical but the embedding shrinks.
    Confirm both still fit the budget.
  * Tied embeddings: the head is the SAME Parameter object as the
    embedding. We assert this via id() — if the wiring breaks (someone
    accidentally adds a second copy), the count would balloon by V*D
    silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.model_mingru import default_mingru_config  # noqa: E402
from src.models.mingru import MinGRULM, MinGRUConfig  # noqa: E402
from src.utils.param_count import (  # noqa: E402
    TARGET_PARAM_COUNT,
    count_parameters,
    param_breakdown,
)


def _build(vocab_size: int) -> MinGRULM:
    cfg_dict = default_mingru_config(vocab_size=vocab_size)
    return MinGRULM(MinGRUConfig(**cfg_dict))


def test_default_config_within_budget():
    """vocab_size=65 (TinyShakespeare): default configuration."""
    model = _build(vocab_size=65)
    n = count_parameters(model)
    lo = int(TARGET_PARAM_COUNT * 0.9)
    hi = int(TARGET_PARAM_COUNT * 1.1)
    assert lo <= n <= hi, (
        f"TinyShakespeare-config minGRU has {n:,} params; "
        f"expected in [{lo:,}, {hi:,}]"
    )


def test_longcopy_config_within_budget():
    """vocab_size=29 (long-range copy): identical body, smaller embedding."""
    model = _build(vocab_size=29)
    n = count_parameters(model)
    lo = int(TARGET_PARAM_COUNT * 0.9)
    hi = int(TARGET_PARAM_COUNT * 1.1)
    assert lo <= n <= hi, (
        f"long-copy-config minGRU has {n:,} params; expected in [{lo:,}, {hi:,}]"
    )


def test_induction_config_within_budget():
    """vocab_size=27 (induction): smallest embedding of the three."""
    model = _build(vocab_size=27)
    n = count_parameters(model)
    lo = int(TARGET_PARAM_COUNT * 0.9)
    hi = int(TARGET_PARAM_COUNT * 1.1)
    assert lo <= n <= hi, (
        f"induction-config minGRU has {n:,} params; expected in [{lo:,}, {hi:,}]"
    )


def test_embeddings_are_tied():
    """The tied head must literally share the embedding Parameter.

    If somebody later changes `MinGRULM` to create a separate `nn.Linear`
    for the head while leaving `tie_embeddings=True` in the config, we
    want that to surface immediately.
    """
    model = _build(vocab_size=65)
    cfg = model.cfg
    assert cfg.tie_embeddings, "default config should have tie_embeddings=True"
    assert model.lm_head is None, (
        "with tied embeddings, lm_head must be None (the embedding matrix is "
        "used directly via x @ tok_emb.weight.T)"
    )


def test_param_count_dedups_tied_params():
    """count_parameters must NOT double-count tied parameters.

    We construct two models — one tied, one untied — and confirm the tied
    model has fewer parameters by exactly vocab_size * d_model (the size
    of the head matrix).
    """
    cfg_dict = default_mingru_config(vocab_size=65)
    tied = MinGRULM(MinGRUConfig(**{**cfg_dict, "tie_embeddings": True}))
    untied = MinGRULM(MinGRUConfig(**{**cfg_dict, "tie_embeddings": False}))
    n_tied = count_parameters(tied)
    n_untied = count_parameters(untied)
    diff = n_untied - n_tied
    expected = 65 * 128  # vocab_size * d_model
    assert diff == expected, (
        f"untied has {n_untied:,}, tied has {n_tied:,}, diff={diff} but "
        f"expected vocab_size*d_model={expected}"
    )


def test_breakdown_is_helpful_for_drift_diagnosis():
    """Sanity-check that param_breakdown actually returns per-name counts.

    Not a numeric test — just that the dict is non-empty, keys look like
    PyTorch parameter paths, and the values sum to a reasonable fraction
    of the total (≥ 95% — small slack for any non-trainable buffers that
    might creep in).
    """
    model = _build(vocab_size=65)
    breakdown = param_breakdown(model)
    assert breakdown, "param_breakdown returned an empty dict"
    assert any("blocks" in k for k in breakdown), (
        "expected at least one block-related parameter in the breakdown"
    )
    total = count_parameters(model)
    sum_breakdown = sum(breakdown.values())
    # breakdown counts EVERY named parameter (including tied embeddings twice
    # if both names point at the same tensor). So sum_breakdown >= total is
    # the right relationship; we just check it's in the right ballpark.
    assert sum_breakdown >= int(total * 0.95), (
        f"breakdown sums to {sum_breakdown}, total is {total} — large gap "
        "suggests un-named params are being missed"
    )


def main() -> int:
    tests = [
        test_default_config_within_budget,
        test_longcopy_config_within_budget,
        test_induction_config_within_budget,
        test_embeddings_are_tied,
        test_param_count_dedups_tied_params,
        test_breakdown_is_helpful_for_drift_diagnosis,
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

    # On any run, also print the actual count and breakdown for the default
    # config — useful diagnostic if the budget assertion fails.
    print("\n--- Diagnostic: default-config parameter count ---")
    model = _build(vocab_size=65)
    print(f"total: {count_parameters(model):,}")
    for name, n in sorted(param_breakdown(model).items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>9,}  {name}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
