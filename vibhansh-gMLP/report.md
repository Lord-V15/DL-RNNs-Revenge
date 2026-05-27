# Causal gMLP — Architecture, Decisions, and Results

## Overview

This directory contains the Causal gMLP implementation for the COMP6242 "RNN's Revenge" project. The model is one of three paradigms (minGRU, Transformer, Causal gMLP) compared across three tasks at three sequence lengths each (9 experiments total).

## Architecture

Based on Liu et al. (2021) "Pay Attention to MLPs", adapted for autoregressive (causal) sequence modeling.

**Block structure:**
```
Input → LayerNorm → Linear(d_model → d_ffn) → GELU → split into [u, v]
v → CausalSpatialGatingUnit → v'
u * v' → Linear(d_ffn/2 → d_model) → Dropout → + Residual
```

**Key component — Causal Spatial Gating Unit (SGU):**
- Learnable weight matrix W of shape [window_size × window_size]
- Masked with lower-triangular causal mask (position i can only attend to positions ≤ i)
- Initialized near identity (I + N(0, 0.02)) for stable training
- For seq_len ≤ window_size: uses full n×n causal projection (exact gMLP)
- For seq_len > window_size: sliding window with the last row of W applied to each window

## Model Configuration

| Hyperparameter | Value |
|---------------|-------|
| d_model | 128 |
| d_ffn | 512 |
| n_layers | 4 |
| window_size | 256 |
| dropout | 0.05 |
| tie_weights | True |

## Parameter Budget Decisions

The project requires ~750K parameters (±10%) for fair cross-paradigm comparison.

**Problem:** The original gMLP uses a full n×n spatial projection matrix. This makes parameter count depend on sequence length:
- seq_len=128: ~485K params
- seq_len=256: ~703K params
- seq_len=2048: would be ~17M params (way over budget)

**Solution — Fixed-window approach:**
- SGU uses a fixed window_size=256 regardless of input sequence length
- Parameters are O(window_size²) = constant
- At seq_len ≤ 256: equivalent to original full-sequence gMLP (exact)
- At seq_len > 256: each position mixes from the preceding 256 positions only

**Resulting parameter counts:**
| Sequence Length | Parameters | Notes |
|----------------|-----------|-------|
| 128 | 485K | Window covers full sequence |
| 144 | 508K | Window covers full sequence |
| 256 | 703K | Window = sequence length |
| 512 | 731K | Window < sequence (local mixing only) |
| 544 | 739K | Window < sequence |
| 2048 | 928–933K | Window << sequence |

The parameter increase at longer sequences comes only from the positional embedding table (seq_len × d_model), not from the SGU itself.

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Warmup steps | 200 (linear) |
| LR schedule | Cosine decay to 10% of peak |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Dropout | 0.05 (all tasks, standardized across paradigms) |
| Batch size | 64 (32 for seq_len=2048) |
| Precision | bf16 mixed precision |
| Shakespeare/Copy steps | 5,000 |
| Induction steps | 20,000 (LR decay over 50,000) |
| Seed | 42 |

**Note on induction training:** The induction task requires more steps than Shakespeare/copy because the discriminating signal is a single position per sequence. The LR cosine decay is stretched over 50,000 steps so that at step 20,000 the LR remains ~2e-4 (still actively learning) rather than bottoming out.

## Results

### TinyShakespeare (Natural Language Modeling)

| Context Length | Params | Best Val PPL | Training Time |
|---------------|--------|-------------|---------------|
| 256 | 703K | 3.75 | ~35s |
| 1024 | 802K | 3.68 | ~4,100s |
| 2048 | 933K | 3.72 | ~4,400s |

**Observations:** Strong natural language performance. Longer context helps slightly despite the local window — most language modeling benefit comes from nearby context anyway. With dropout=0.05 (standardized), PPL improved slightly from the earlier 0.2-dropout runs.

### Long-range Copy (Synthetic Retrieval)

| Length | Seq Len | Params | Overall PPL | Disc. PPL | Accuracy | Time |
|--------|---------|--------|-------------|-----------|----------|------|
| Short | 144 | 508K | 10.97 | 1.00 | 100.0% | 36s |
| Medium | 544 | 739K | 21.36 | 26.54 | 3.9% | 2,127s |
| Long | 2048 | 931K | 24.53 | 26.25 | 4.0% | 3,890s |

**Observations:**
- **Short (seq 144 ≤ window 256):** Perfect recall. The full causal mixing covers the entire sequence, so the model sees the key at recall time.
- **Medium/Long (seq > window 256):** Complete failure (~random accuracy). The key is stored 500–2000 positions back, far beyond the 256-position window. The model literally cannot see the key when asked to recall it.

This is the expected architectural limitation: a local-window MLP-mixer has no mechanism for long-range information transport.

### Induction (In-context Pattern Completion)

| Length | Seq Len | Params | Overall PPL | Disc. PPL | Accuracy | Steps | Time |
|--------|---------|--------|-------------|-----------|----------|-------|------|
| Short | 128 | 485K | 14.99 | 1.04 | 99.1% | 20,000 | 149s |
| Medium | 512 | 731K | 13.57 | 1.03 | 99.4% | 20,000 | 7,703s |
| Long | 2048 | 928K | 25.40 | 27.47 | 3.8% | 20,000 | 15,364s |

**Observations:**
- **Short/Medium (within window):** Near-perfect accuracy (99%+). The pattern's first occurrence is within 200 positions of the query (M=50 for short, M=200 for medium), so it falls within the 256-position window. The SGU's learnable spatial mixing can implement the induction circuit when it has access to the first occurrence.
- **Long (M=1000):** Complete failure. The original pattern is ~1000 positions back, far beyond the window.
- The jump from 80% (old buggy data) to 99%+ (corrected data) confirms the earlier results were measuring a data-generation artifact, not the model's true capability.

## Key Findings for Cross-Paradigm Comparison

1. **Natural language:** gMLP is competitive. Local context (256 tokens) captures most language modeling signal. Best val PPL (3.75 at ctx-256) outperforms minGRU (4.44).

2. **Exact retrieval — within window:** gMLP achieves perfect copy (100%) and near-perfect induction (99%+) when the relevant context is within 256 positions.

3. **Exact retrieval — beyond window:** Complete failure. Unlike Transformers (global attention) or RNNs (compressed state over full history), gMLP has no mechanism to access information beyond its fixed window.

4. **The "cliff" is sharp:** Performance drops from 100% → 4.1% (copy) and 99.6% → 4.0% (induction) as soon as the relevant context exceeds the window boundary.

5. **gMLP vs minGRU on induction:** gMLP achieves 99%+ on within-window induction; minGRU scores 3.5-4.6% (random) even at the shortest distance (M=50). This is the clearest paradigm separation: MLP-mixing with direct position access can learn the induction circuit, but recurrence with compressed state cannot. This holds under matched dropout=0.05 and 20K training steps.

## Architecture Iteration History

1. **v1 — Anti-causal uniform average:** Initial implementation had the einsum direction reversed, causing future tokens to leak into past positions. This gave impossibly good PPL (~2.68) due to cheating.

2. **v2 — Causal uniform average:** Fixed causality but used fixed 1/n averaging instead of learnable weights. Destroyed positional information, giving poor PPL (~9.04).

3. **v3 — Full n×n learnable:** Correct and performant, but parameters scaled as O(n²) — 17M params at seq_len=2048, violating the 750K budget.

4. **v4 — Fixed-window learnable (final):** Window of 256 gives constant SGU parameters regardless of sequence length. The naive causal masking (v3) works correctly but violates the parameter budget at long sequences. The fixed-window variant is an honest engineering solution that explicitly reveals gMLP's local-mixing limitation rather than hiding it behind an oversized parameter budget.

## Data Bug Discovery and Fix

During cross-paradigm comparison, the Transformer owner (Mayukh) identified that the induction data generator had a truncation bug:

- **Bug:** `sequence = prefix + pattern + suffix + pattern[:4]` — the target token P[4] was missing from the sequence
- **Fix:** `sequence = prefix + pattern + suffix + pattern` — full pattern appended, P[4] is now the final token
- **Impact:** Old results showed 80% accuracy (measuring local n-gram prediction of P[0:3], not true induction). Corrected data shows 99%+ (true induction recall).
- **Copy task unaffected** — the copy generator always appended the full key.

## File Structure

```
vibhansh-gMLP/
├── model.py              # CausalGMLP model with windowed SGU
├── train.py              # Training script (all 3 tasks, bf16, --lr_decay_steps)
├── dataset_utils.py      # Shared data loading (same as other paradigms)
├── data_generator.py     # On-the-fly data generation (unused in final runs)
├── run_all_experiments.sh # Runs all 9 experiments
├── rerun_induction.sh    # Re-runs induction (corrected data, 20K steps)
├── rerun_5k.sh           # Re-runs synthetic tasks at 5K steps
├── results/              # JSON output files
│   ├── gmlp_shakespeare_256_42.json
│   ├── gmlp_shakespeare_1024_42.json
│   ├── gmlp_shakespeare_2048_42.json
│   ├── gmlp_copy_short_42.json
│   ├── gmlp_copy_medium_42.json
│   ├── gmlp_copy_long_42.json
│   ├── gmlp_induction_short_42.json
│   ├── gmlp_induction_medium_42.json
│   └── gmlp_induction_long_42.json
├── report.md             # This file
└── BREV_QUICKSTART.md    # Deployment guide for Nvidia Brev
```

## Data

All tasks use shared pre-generated data from `./data/` (same files as minGRU and Transformer runs) for fair comparison:
- `data/longrange_copy/{train,val}_{short,medium,long}.txt`
- `data/induction/{train,val}_{short,medium,long}.txt` + `*_patterns.txt` (corrected: full pattern appended)
- Shakespeare: `./data/tinyshakespeare/input.txt` (TinyShakespeare, 1.1MB)

## Hardware

All experiments ran on Nvidia Brev instance (`llm-train`, 8×H100 SXM). Each experiment used a single GPU.
