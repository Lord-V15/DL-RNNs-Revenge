"""minGRU language model.

Implements the minimal GRU from Feng et al. (2024), "Were RNNs All We Needed?".

Key design points:
  * The update gate `z_t` and candidate `h_tilde_t` are computed from `x_t` only,
    NOT from `h_{t-1}`. This removes the sequential dependency on the previous
    hidden state, so the recurrence can be evaluated in parallel via a prefix scan.
  * The recurrence is `h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t`, a first-order
    linear recurrence in `h_{t-1}` with coefficients `(1 - z_t)` and forcing term
    `z_t * h_tilde_t`. Any first-order linear recurrence admits an associative
    parallel scan.
  * The scan is performed in log-space for numerical stability: at sequence
    length 2048, naive cumulative products of values in (0, 1) underflow fp32.
    Feng et al. provide a log-space "parallel scan log" formulation; this file
    implements it.
  * The block structure follows Feng et al. Appendix C.2 language modelling recipe:
    Conv1d (kernel 4, causal) → minGRU → MLP, each with residual + RMSNorm pre-norm.

Param budget target: ~750K with d_model=128, L=4, expansion=1.0, mlp_mult=4.5.

CORRECTNESS: `parallel_scan_log` MUST produce outputs equivalent to the naive
sequential recurrence within fp32 tolerance. See
tests/test_mingru_parallel_equals_sequential.py — that test must pass before
trusting any training result from this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MinGRUConfig:
    """Hyperparameters for the minGRU language model.

    Defaults target ~737K parameters with vocab_size=65 (TinyShakespeare).
    Param counts on other vocabularies will differ slightly via the embedding
    + tied head; the body is identical.
    """
    vocab_size: int = 65
    d_model: int = 128
    n_blocks: int = 4
    expansion_factor: float = 1.0   # multiplier on d_model for the minGRU inner width
    mlp_mult: float = 4.5           # MLP hidden width as a multiple of d_model
    conv_kernel: int = 4            # causal conv kernel size in each block
    dropout: float = 0.2
    tie_embeddings: bool = True
    rms_norm_eps: float = 1e-5


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root mean square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class CausalConv1d(nn.Module):
    """Depthwise causal 1D convolution along the time axis.

    Uses left-padding so position t can only see positions <= t. Operates on
    tensors of shape [B, T, D] internally by transposing to [B, D, T] for
    nn.Conv1d.
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Depthwise: groups = d_model. Each channel mixes only with itself across time.
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        # Left-pad with (kernel_size - 1) zeros so output length == T and the
        # conv at position t reads positions [t - k + 1, t].
        x = x.transpose(1, 2)  # [B, D, T]
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)       # [B, D, T]
        return x.transpose(1, 2)  # [B, T, D]


# ---------------------------------------------------------------------------
# Parallel prefix scan (log-space)
# ---------------------------------------------------------------------------

def _g(x: torch.Tensor) -> torch.Tensor:
    """Activation `g(.)` from Feng et al. ensuring h_tilde is positive.

    Used so that the log-space scan can take logs of `h_tilde` without
    worrying about sign. Feng et al. define:
        g(x) = where(x >= 0, x + 0.5, sigmoid(x))
    which is continuous at 0 (both branches equal 0.5) and strictly positive.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def _log_g(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(g(x)).

    For x >= 0: log(x + 0.5).
    For x <  0: log(sigmoid(x)) = -softplus(-x).
    """
    return torch.where(
        x >= 0,
        torch.log(F.relu(x) + 0.5),   # ReLU guards against negative leakage from bf16
        -F.softplus(-x),
    )


def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    """Parallel scan for the first-order linear recurrence
        h_t = a_t * h_{t-1} + b_t,  with h_0 = 0,
    evaluated entirely in log-space for numerical stability.

    The closed form is
        h_t = sum_{k=1..t} (prod_{j=k+1..t} a_j) * b_k
    which in log-space (assuming b_k > 0) is
        log h_t = logcumsumexp_k [ (sum_{j=k+1..t} log a_j) + log b_k ]
                = logcumsumexp_k [ (S_t - S_k) + log b_k ]
    where S_t = sum_{j=1..t} log a_j is the cumulative sum of log coefficients.

    Letting `a_star = cumsum(log_a)` (i.e. S_t) and
            `log_h0_plus_b_star = logcumsumexp(log b_k - S_k)`,
    we have `log h_t = a_star_t + log_h0_plus_b_star_t`.

    Args:
        log_coeffs: log a_t, shape [B, T, D]. a_t = (1 - z_t), in (0, 1), so
            log_coeffs <= 0.
        log_values: log b_t, shape [B, T, D]. b_t = z_t * h_tilde_t, > 0 by
            construction (h_tilde via g(.) is positive; z_t via sigmoid is
            positive).

    Returns:
        h: shape [B, T, D], the hidden states h_1 ... h_T (with h_0 = 0).

    NOTE: Assumes h_0 = 0. To support a non-zero h_0, prepend log(h_0) to
    `log_values` and 0 to `log_coeffs`, then drop the first output position.
    Not needed for our LM use case which always starts from zero state.
    """
    # S_t = cumulative sum of log a along the time dim
    a_star = log_coeffs.cumsum(dim=1)
    # logcumsumexp of (log b_t - S_t)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


# ---------------------------------------------------------------------------
# minGRU cell
# ---------------------------------------------------------------------------

class MinGRU(nn.Module):
    """Minimal GRU with parallel scan.

    The forward pass operates on the entire sequence at once. Both gates depend
    on x_t only (not on h_{t-1}), which is what enables the parallel scan.

    For inference / generation, a sequential mode is also provided, returning
    the final hidden state alongside the outputs so callers can roll out
    token-by-token with constant memory.
    """

    def __init__(self, d_model: int, expansion_factor: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expansion_factor)
        # Single linear producing [z_logits | h_tilde] of size 2 * d_inner.
        self.proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Output projection back to d_model. When expansion_factor == 1.0 this
        # is square; we keep it explicit so non-unit expansion works without
        # special-casing.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward.

        Args:
            x: [B, T, D] input sequence.

        Returns:
            y: [B, T, D] output sequence.
        """
        z_logits, h_tilde_pre = self.proj(x).chunk(2, dim=-1)
        # Compute log-coefficients and log-values entirely in log-space.
        # log a_t = log(1 - sigmoid(z_logits)) = log(sigmoid(-z_logits))
        #        = -softplus(z_logits)
        log_a = -F.softplus(z_logits)
        # log b_t = log(z_t) + log(g(h_tilde_pre))
        #         = -softplus(-z_logits) + log_g(h_tilde_pre)
        log_b = -F.softplus(-z_logits) + _log_g(h_tilde_pre)

        # Run the scan in fp32 for numerical safety. logcumsumexp on bf16 is
        # noisy at length ~2048. We cast back to the input dtype on exit.
        in_dtype = x.dtype
        h = parallel_scan_log(log_a.float(), log_b.float()).to(in_dtype)
        return self.out_proj(h)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrence for inference.

        Args:
            x_t: [B, D] input at time t.
            h_prev: [B, d_inner] previous hidden state, or None for h_0 = 0.

        Returns:
            y_t: [B, D] output at time t.
            h_t: [B, d_inner] new hidden state.
        """
        z_logits, h_tilde_pre = self.proj(x_t).chunk(2, dim=-1)
        z = torch.sigmoid(z_logits)
        h_tilde = _g(h_tilde_pre)
        if h_prev is None:
            h_prev = torch.zeros_like(h_tilde)
        h_t = (1.0 - z) * h_prev + z * h_tilde
        y_t = self.out_proj(h_t)
        return y_t, h_t


# ---------------------------------------------------------------------------
# Block: Conv4 -> minGRU -> MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP with GELU, following the standard pre-norm transformer recipe."""

    def __init__(self, d_model: int, mult: float, dropout: float):
        super().__init__()
        hidden = int(round(d_model * mult))
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MinGRUBlock(nn.Module):
    """One Conv4 -> minGRU -> MLP block with pre-norm and residuals.

    Per Feng et al. Appendix C.2:
        x = x + Conv4(RMSNorm(x))
        x = x + minGRU(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, cfg: MinGRUConfig):
        super().__init__()
        self.norm_conv = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.conv = CausalConv1d(cfg.d_model, cfg.conv_kernel)
        self.drop_conv = nn.Dropout(cfg.dropout)

        self.norm_gru = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.gru = MinGRU(cfg.d_model, cfg.expansion_factor)
        self.drop_gru = nn.Dropout(cfg.dropout)

        self.norm_mlp = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.mlp_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_conv(self.conv(self.norm_conv(x)))
        x = x + self.drop_gru(self.gru(self.norm_gru(x)))
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------

class MinGRULM(nn.Module):
    """minGRU language model with tied input/output embeddings.

    Forward returns logits of shape [B, T, V]. Loss computation is left to the
    trainer so per-position masks (recall positions, pattern-completion
    positions) can be applied externally.
    """

    def __init__(self, cfg: MinGRUConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MinGRUBlock(cfg) for _ in range(cfg.n_blocks)])
        self.norm_out = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        if cfg.tie_embeddings:
            # Tied head: logits = x @ tok_emb.weight.T (no extra Parameter).
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: [B, T] int64 token ids.

        Returns:
            logits: [B, T, V]
        """
        x = self.emb_drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        if self.lm_head is None:
            logits = x @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(x)
        return logits

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count parameters. With tied embeddings, the embedding matrix is
        counted once (it is the same Parameter object used as the head)."""
        if only_trainable:
            params = (p for p in self.parameters() if p.requires_grad)
        else:
            params = self.parameters()
        seen = set()
        total = 0
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
