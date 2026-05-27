"""
Causal gMLP for Autoregressive Sequence Modeling
Based on Liu et al. (2021) "Pay Attention to MLPs"
Modified with causal masking for autoregressive use.

Target: ~750K parameters at d_model=128, d_ffn=768, L=4 blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit with causal masking for autoregressive generation.

    Uses a fixed-window causal spatial projection: each position mixes from
    the preceding `window_size` positions via a learnable weight matrix.
    Parameters are O(window_size^2) — constant regardless of sequence length.
    At short sequences (seq_len <= window_size), this reduces to the full n×n
    lower-triangular projection from the original gMLP paper.
    """
    def __init__(self, d_ffn, seq_len, window_size=256):
        super().__init__()
        self.d_ffn = d_ffn
        self.seq_len = seq_len
        self.window_size = min(window_size, seq_len)
        self.norm = nn.LayerNorm(d_ffn // 2)

        # Learnable spatial projection: [window_size × window_size], lower-triangular
        self.spatial_weight = nn.Parameter(torch.zeros(self.window_size, self.window_size))
        self.spatial_bias = nn.Parameter(torch.zeros(self.window_size))

        # Initialize near identity
        nn.init.normal_(self.spatial_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.spatial_weight.add_(torch.eye(self.window_size))

        # Causal mask for the window
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(self.window_size, self.window_size))
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_ffn // 2]
        Returns:
            [batch_size, seq_len, d_ffn // 2]
        """
        batch_size, seq_len, d = x.shape
        x_norm = self.norm(x)

        W = self.spatial_weight * self.causal_mask  # [W, W] lower-triangular

        if seq_len <= self.window_size:
            # Short sequence: use full n×n causal mixing (exact gMLP)
            W_cropped = W[:seq_len, :seq_len]
            gated = torch.einsum('ij,bjd->bid', W_cropped, x_norm)
            gated = gated + self.spatial_bias[:seq_len].view(1, seq_len, 1)
        else:
            # Long sequence: sliding window causal mixing
            # Pad input with zeros on the left so we can use unfold
            pad = self.window_size - 1
            x_padded = F.pad(x_norm, (0, 0, pad, 0))  # [B, pad+L, D]

            # Unfold into windows: [B, L, window_size, D]
            x_windows = x_padded.unfold(1, self.window_size, 1)  # [B, L, D, W]
            x_windows = x_windows.permute(0, 1, 3, 2)  # [B, L, W, D]

            # Apply the last row of W to each window (each position uses full window)
            # W[-1, :] gives the weights for a position that sees all W past positions
            w_row = W[-1, :]  # [W]
            gated = torch.einsum('blwd,w->bld', x_windows, w_row)
            gated = gated + self.spatial_bias[-1]

        return gated


class gMLPBlock(nn.Module):
    """
    Single gMLP block with causal masking.

    Architecture: Norm → Linear → GELU → split → SGU_causal → multiply → Linear → residual
    """
    def __init__(self, d_model, d_ffn, seq_len, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.sgu = CausalSpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x

        # Norm
        x = self.norm(x)

        # Channel projection + activation
        x = self.channel_proj1(x)  # [B, L, d_ffn]
        x = self.activation(x)

        # Split into two halves
        u, v = torch.chunk(x, 2, dim=-1)  # Each: [B, L, d_ffn//2]

        # Apply causal spatial gating to v
        v = self.sgu(v)  # [B, L, d_ffn//2]

        # Multiply (gating)
        x = u * v  # [B, L, d_ffn//2]

        # Project back to d_model
        x = self.channel_proj2(x)  # [B, L, d_model]
        x = self.dropout(x)

        # Residual connection
        return residual + x


class CausalGMLP(nn.Module):
    """
    Causal gMLP Language Model

    Configuration for ~750K parameters:
    - d_model: 128
    - d_ffn: 768
    - n_layers: 4
    - seq_len: varies by task
    """
    def __init__(self, vocab_size, d_model=128, d_ffn=512, n_layers=4,
                 seq_len=256, dropout=0.2, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(d_model, d_ffn, seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Final norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - token indices
            targets: [batch_size, seq_len] - target tokens (for training)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar (if targets provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [B, L, d_model]

        # Positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
        pos_emb = self.pos_embedding(positions)  # [1, L, d_model]

        # Combine
        x = token_emb + pos_emb  # [B, L, d_model]

        # Apply gMLP blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.ln_f(x)  # [B, L, d_model]

        # Language model head
        logits = self.lm_head(x)  # [B, L, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1  # Ignore padding
            )

        return logits, loss

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.

        Args:
            idx: [batch_size, seq_len] - conditioning tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k most likely tokens
        Returns:
            [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx if idx.size(1) <= self.seq_len else idx[:, -self.seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


def create_causal_gmlp(vocab_size, seq_len=256, config='base', dropout=None):
    """
    Factory function to create Causal gMLP models with different configs.

    Args:
        vocab_size: size of vocabulary
        seq_len: maximum sequence length
        config: 'base' for ~750K params
        dropout: override dropout rate (if None, uses config default)
    """
    configs = {
        'base': {
            'd_model': 128,
            'd_ffn': 512,  # Reduced from 768 to hit ~750K target
            'n_layers': 4,
            'dropout': 0.2,
            'tie_weights': True
        }
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    model_config = configs[config]
    if dropout is not None:
        model_config['dropout'] = dropout
    model = CausalGMLP(
        vocab_size=vocab_size,
        seq_len=seq_len,
        **model_config
    )

    print(f"Created Causal gMLP model:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  d_ffn: {model_config['d_ffn']}")
    print(f"  Layers: {model_config['n_layers']}")
    print(f"  Parameters: {model.count_parameters():,}")

    return model


if __name__ == "__main__":
    # Test parameter counts at different sequence lengths
    print("Testing Causal gMLP parameter counts:\n")

    vocab_size = 65  # TinyShakespeare

    for seq_len in [128, 256, 512, 2048]:
        model = create_causal_gmlp(vocab_size, seq_len=seq_len)
        print()

    # Test forward pass
    print("\nTesting forward pass:")
    model = create_causal_gmlp(vocab_size, seq_len=256)
    batch_size = 4
    seq_len = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    print("\nTesting generation:")
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"Generated shape: {generated.shape}")
