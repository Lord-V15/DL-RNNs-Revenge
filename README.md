# Transformer's Revenge: Stress-Testing minGRU's Claims Against Attention on Algorithmic Reasoning Tasks

**COMP6242 Deep Learning · Semester 1, 2026**

Mayukh Das (u7965027) · Vibhansh Gupta (u7861976) · Daniel Vaz (u7990536) · Adam Clark (u7437561)

---

## Abstract

Feng et al. (2024) claim that minGRU, a minimal recurrent unit trainable via parallel log-space scans, matches Transformer performance on language modelling with O(T) complexity. We stress-test this claim by moving beyond statistical language modelling into tasks requiring explicit algorithmic reasoning: verbatim sequence copying and induction head pattern completion.

In a controlled 27-experiment grid (3 models × 3 tasks × 3 sequence lengths, ~750-800K parameter target), we show that the claim holds for language modelling but **categorically fails** for algorithmic reasoning. minGRU's input-only gating cannot perform content-based retrieval at any sequence length. A causal gMLP, adapted as a third paradigm, succeeds within its fixed 256-token window but fails beyond it. Only the Transformer's content-dependent attention enables unbounded algorithmic reasoning.

---

## Motivation

Language modelling benchmarks test statistical pattern learning from local context. But many real-world capabilities (retrieval, reasoning, in-context learning) require **content-based information routing**: using the current input as a query to selectively access past information. We designed tasks that specifically isolate this capability to determine where minGRU's architectural limitations become fatal.

The key architectural insight: minGRU's gate g_t = σ(W·x_t) depends only on the current input, not on the hidden state or any future query. This means the model must decide **at write-time** what to store, without knowledge of what will be needed later. For tasks requiring **read-time** decisions (like induction heads), this is a structural impossibility.

---

## Models

All models share 4 layers, hidden dimension 128, and approximately matched parameters:

| | Transformer | gMLP | minGRU |
|---|---|---|---|
| **Parameters** | ~795K | ~508-933K* | ~737K |
| **Attention/Window** | 4 heads | w=256 | -- |
| **FFN** | GELU, 4×d | GELU, d_ffn=512 | GELU, 4.5×d |
| **Normalisation** | RMSNorm | LayerNorm | RMSNorm |
| **Position encoding** | RoPE | Learned | Implicit (recurrence) |
| **Additional** | bias=False | -- | Conv1d (k=4) |

*gMLP parameter count varies with sequence length due to T×T spatial weight matrix.

**Transformer**: nanoGPT enhanced with LLaMA-recipe modifications (RoPE, RMSNorm, no bias). GELU activation (SwiGLU introduced instability at this scale in prior coursework).

**minGRU**: Direct reproduction of Feng et al. Appendix C.2 language modelling recipe. Each block: causal Conv1d (k=4) → minGRU cell → MLP, with pre-norm RMSNorm and residual connections. Parallel training via log-space cumulative sums.

**Causal gMLP**: Our adaptation of the originally bidirectional gMLP (Liu et al. 2021) for autoregressive generation. Uses a causally-masked, windowed Spatial Gating Unit (SGU) with fixed spatial weights W_s ∈ R^(T×T), banded to window size w=256.

---

## Tasks

### Task 1: TinyShakespeare (Language Modelling Baseline)
Character-level language modelling on 1.1M-character Shakespeare corpus (65-char vocab). 90/10 train/val split. This directly mirrors Feng et al.'s benchmark. **Metric**: validation perplexity.

### Task 2: Long-Range Copy (Verbatim Retrieval)
Format: `[content][separator][content]` — model must reproduce the first half verbatim after the separator. Content drawn from 26-character alphabet.
- Short: T=128 (64 tokens to memorise)
- Medium: T=528 (256 tokens)
- Long: T=2048 (1024 tokens)

No statistical shortcut exists. **Metric**: recall perplexity (computed only on recalled portion; 1.0 = perfect, ~26 = random).

### Task 3: Induction Heads (Content-Based Retrieval)
Sequences containing repeated bigram patterns: after observing [A][B] earlier, model must predict B when A reappears. 5 induction patterns per sequence.
- Short: T=128
- Medium: T=256
- Long: T=2048

Tests content-based retrieval: model must use current token A as a **query** to find what followed A previously. **Metric**: masked accuracy on induction positions only (~0.04 = random, 1.0 = perfect).

---

## Results

### Summary Heatmap

All 27 experiments at a glance:

| | Shak 256 | Shak 1024 | Shak 2048 | Copy Short | Copy Med | Copy Long | Ind Short | Ind Med | Ind Long |
|---|---|---|---|---|---|---|---|---|---|
| **Transformer** | 4.41 | 4.24 | 4.24 | 1.00 | 1.03 | 26.12 | 0.97 | 1.00 | 1.00 |
| **gMLP** | 3.79 | 3.72 | 3.71 | 1.00 | 26.38 | 26.05 | 1.00 | 1.00 | 0.04 |
| **minGRU** | 4.41 | 4.34 | 4.32 | 26.22 | 26.20 | 26.09 | 0.04 | 0.05 | 0.03 |

Green = pass, Red = fail. Shakespeare: perplexity (lower better). Copy: recall PPL (1.0 perfect). Induction: accuracy (1.0 perfect).

### Key Findings

1. **minGRU fails all algorithmic tasks** at every sequence length (PPL≈26, accuracy≈0.04). This confirms the architectural limitation is fundamental, not a capacity or training issue.

2. **gMLP succeeds within its window, fails beyond it.** Copy-short (T=128 < w=256): perfect. Copy-medium (T=528 > w=256): random. The boundary is sharp, not gradual.

3. **Transformer succeeds on all tasks** except copy-long (T=2048), where it remains at random after 5K training steps. However...

4. **Phase transition observed.** On induction-long, the Transformer stays at random chance (0.04) for 7,000 steps, then abruptly jumps to perfect accuracy (1.0) within ~2,000 steps. This circuit formation pattern (consistent with Olsson et al. 2022, Power et al. 2022) implies the copy-long failure is a training budget issue, not an architectural one.

5. **Shakespeare parity confirmed.** All models achieve comparable perplexity (3.7-4.4), replicating Feng et al.'s finding that minGRU matches Transformers on language modelling.

### The Write-Time vs Read-Time Distinction

Feng et al.'s own "selective copying" task (where minGRU achieves 99.5%) tests **write-time gating**: certain tokens are marked as important at encoding time, and the gate learns to store them. Our tasks test **read-time gating**: retrieval conditioned on a future query. This is the precise boundary that input-only gating cannot cross (Merrill et al. 2024).

---

## Training Protocol

Identical across all 27 experiments:

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Learning rate | Cosine decay: 3e-4 → 3e-5 |
| Warmup | 200 steps (linear) |
| Dropout | 0.05 |
| Gradient clipping | Max norm 1.0 |
| Batch size | 64 (32 for gMLP at T=2048) |
| Steps | 5,000 (Shakespeare, Copy); 20,000 (Induction) |
| Hardware | NVIDIA A100 (Google Colab) |
| Seed | 42 |

---

## Repository Structure

```
DL-RNNs-Revenge/
├── report/
│   ├── main.tex                    # Full LaTeX report
│   ├── references.bib              # Bibliography (13 references)
│   └── figures/                    # All publication figures (PNG)
│       ├── 06_phase_transition.png # Induction-long phase transition
│       ├── 07_summary_heatmap.png  # 27-experiment summary
│       ├── fig1_copy_ppl_line.png  # Copy task line plot
│       ├── fig2_induction_accuracy_line.png
│       ├── 03_shakespeare_ppl.png
│       └── 04_wall_clock.png
├── transformer_project/
│   ├── model.py                    # Transformer (RoPE + RMSNorm + no bias)
│   ├── train_synthetic.py          # Training script
│   ├── dataset_utils.py            # Data loading + evaluation
│   ├── config/                     # Per-experiment configs
│   └── out/                        # 9 run summaries (summary.json with eval_history)
├── minGRU/
│   ├── src/models/mingru.py        # minGRU with parallel log-space scan
│   ├── src/training/               # Training loop
│   ├── configs/                    # Experiment configurations
│   └── runs/                       # 9 run logs (metrics.csv, config, console)
├── vibhansh-gMLP/
│   ├── model.py                    # Causal gMLP with windowed SGU
│   ├── dataset_utils.py            # Data loading
│   └── results/                    # 9 run summaries (.json)
├── plots.ipynb                     # Figure generation notebook
├── run_transformer.ipynb           # Colab: all 9 transformer experiments
├── run_mingru.ipynb                # Colab: all 9 minGRU experiments
└── run_gmlp.ipynb                  # Colab: all 9 gMLP experiments
```

---

## Reproducing Results

1. **Generate data**: Each `run_*.ipynb` notebook includes data generation cells. Synthetic tasks use deterministic random seeds.

2. **Train**: Open the relevant `run_*.ipynb` in Google Colab with A100 runtime. Each notebook runs all 9 experiments for its model sequentially.

3. **Plot**: Run `plots.ipynb` after all experiments complete (reads from `out/`, `runs/`, and `results/` directories).

4. **Compile report**: Upload `report/` folder to Overleaf or compile locally with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.

---

## References

- Feng, L. et al. (2024). "Were RNNs All We Needed?" arXiv:2410.01201
- Liu, H. et al. (2021). "Pay Attention to MLPs." NeurIPS 34.
- Olsson, C. et al. (2022). "In-context Learning and Induction Heads." Transformer Circuits Thread.
- Su, J. et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing.
- Touvron, H. et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971
- Merrill, W. & Sabharwal, A. (2024). "The Illusion of State in State-Space Models." arXiv:2404.08819
- Power, A. et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." ICLR Workshop.

---

## License

This repository is submitted as coursework for COMP6242 Deep Learning at the Australian National University. All code and analysis is original work by the team members listed above.
