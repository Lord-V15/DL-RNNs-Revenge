# RNN's Revenge — Transformer

Transformer side of the COMP6242 paradigm comparison project.
Owner: Mayukh.

Decoder-only Transformer with RoPE + RMSNorm, ~796K parameters,
matched to minGRU and Causal gMLP per proposal v4.0 §5.

## Setup

```bash
pip install -r requirements.txt
python data/tinyshakespeare/prepare.py   # one-time download + tokenise
```

## Run

```bash
# Primary runs (proposal §6, TinyShakespeare rows)
python train.py config/tshake_256.py
python train.py config/tshake_1024.py
python train.py config/tshake_2048.py
```

Each writes `out/<run_name>/summary.json` with all metrics required
by proposal §7: best val PPL, peak GPU memory, throughput, wall-clock,
full hyperparameters, git SHA, timestamp.

## CLI overrides

Any config var can be overridden:

```bash
python train.py config/tshake_256.py --seed=1337 --max_iters=2000
```

Useful flags:
- `--seed=N` — change the RNG seed
- `--max_iters=N` — train for fewer steps (smoke test)
- `--batch_size=N` — reduce if OOM at length 2048
- `--compile_model=True` — turn on `torch.compile` (slow first iter, faster after)

## Files

| File | Purpose |
|---|---|
| `model.py` | RoPE + RMSNorm decoder-only Transformer (Karpathy lineage) |
| `train.py` | Single-GPU trainer, writes summary.json per run |
| `configurator.py` | Karpathy-style config file + CLI override system |
| `config/tshake_*.py` | One config per length (256, 1024, 2048) |
| `data/tinyshakespeare/prepare.py` | Download + char-tokenise + 90/10 split |

## Architecture notes

Defined in `model.py`. ~796K params at the configured size (d_model=128,
n_head=4, L=4, vocab=65, no bias).

- **Positional**: RoPE applied to q/k inside attention; no learned `wpe`
- **Norm**: RMSNorm (pre-norm in each block, plus a final RMSNorm before lm_head)
- **MLP**: 4× expansion, GELU, no bias
- **Embedding**: tied input/output (lm_head shares wte's weight)
- **Attention**: causal via `is_causal=True` in scaled_dot_product_attention
  (Flash attention picked automatically by PyTorch on H100 / L4)