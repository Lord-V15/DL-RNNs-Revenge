# RNN's Revenge — Transformer

Decoder-only Transformer with RoPE + RMSNorm for the COMP6242 paradigm comparison project.

## Architecture

- 4 layers, d_model=128, 4 heads, no bias
- RoPE positional encoding, RMSNorm (pre-norm), GELU FFN (4×d)
- Tied input/output embeddings
- Flash attention via PyTorch scaled_dot_product_attention

## Parameter Counts (Actual)

| Task | Seq Length | Parameters |
|------|-----------|------------|
| Shakespeare | 256 | 795,904 |
| Shakespeare | 1024 | 795,904 |
| Shakespeare | 2048 | 795,904 |
| Copy | short (128) | 794,624 |
| Copy | medium (528) | 794,624 |
| Copy | long (2048) | 794,624 |
| Induction | short (128) | 791,040 |
| Induction | medium (256) | 791,040 |
| Induction | long (2048) | 791,040 |

## Results

| Task | Seq Length | Metric | Value |
|------|-----------|--------|-------|
| Shakespeare | 256 | Val PPL | 4.41 |
| Shakespeare | 1024 | Val PPL | 4.24 |
| Shakespeare | 2048 | Val PPL | 4.24 |
| Copy | short | Recall PPL | 1.00 |
| Copy | medium | Recall PPL | 1.03 |
| Copy | long | Recall PPL | 26.12 (fail) |
| Induction | short | Accuracy | 0.97 |
| Induction | medium | Accuracy | 1.00 |
| Induction | long | Accuracy | 1.00 |

## Training Times (NVIDIA A100, Google Colab)

| Task | Seq Length | Steps | Wall Clock |
|------|-----------|-------|------------|
| Shakespeare | 256 | 5,000 | 154s |
| Shakespeare | 1024 | 5,000 | 180s |
| Shakespeare | 2048 | 5,000 | 285s |
| Copy | short | 5,000 | 65s |
| Copy | medium | 5,000 | 101s |
| Copy | long | 5,000 | 237s |
| Induction | short | 20,000 | 246s |
| Induction | medium | 20,000 | 317s |
| Induction | long | 20,000 | 800s |

## Setup

```bash
pip install -r requirements.txt
python data/tinyshakespeare/prepare.py   # one-time download + tokenise
python generate_longrange_copy.py        # generate copy task data
python generate_induction.py             # generate induction task data
```

## Run

```bash
# Shakespeare
python train.py config/tshake_256.py
python train.py config/tshake_1024.py
python train.py config/tshake_2048.py

# Copy
python train_synthetic.py config/copy_short.py
python train_synthetic.py config/copy_medium.py
python train_synthetic.py config/copy_long.py

# Induction
python train_synthetic.py config/induction_short.py
python train_synthetic.py config/induction_medium.py
python train_synthetic.py config/induction_long.py
```

Each run writes `out/<run_name>/summary.json` with best val PPL, recall PPL or induction accuracy, wall-clock, throughput, peak GPU memory, full hyperparameters, and eval history.

## Files

| File | Purpose |
|------|---------|
| `model.py` | RoPE + RMSNorm decoder-only Transformer |
| `train.py` | Shakespeare trainer |
| `train_synthetic.py` | Copy and Induction trainer |
| `dataset_utils.py` | Data loading and evaluation metrics |
| `configurator.py` | Config file + CLI override system |
| `config/tshake_*.py` | Shakespeare configs |
| `config/copy_*.py` | Copy task configs |
| `config/induction_*.py` | Induction task configs |
