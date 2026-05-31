# RNN's Revenge — Causal gMLP

Causal gMLP implementation for the COMP6242 paradigm comparison project. Our adaptation of the originally bidirectional gMLP (Liu et al. 2021) for autoregressive generation.

## Architecture

- 4 layers, d_model=128, d_ffn=512
- Spatial Gating Unit (SGU) with causal lower-triangular masking, window w=256
- LayerNorm (pre-norm), GELU activation
- Learned positional embeddings, tied input/output embeddings
- Parameter count varies with sequence length due to T×T spatial weight matrix

## Parameter Counts (Actual)

| Task | Seq Length | Parameters |
|------|-----------|------------|
| Shakespeare | 256 | 703,360 |
| Shakespeare | 1024 | 801,664 |
| Shakespeare | 2048 | 932,736 |
| Copy | short (128) | 508,096 |
| Copy | medium (528) | 738,944 |
| Copy | long (2048) | 931,456 |
| Induction | short (128) | 484,992 |
| Induction | medium (256) | 731,264 |
| Induction | long (2048) | 927,872 |

## Results

| Task | Seq Length | Metric | Value |
|------|-----------|--------|-------|
| Shakespeare | 256 | Val PPL | 3.79 |
| Shakespeare | 1024 | Val PPL | 3.72 |
| Shakespeare | 2048 | Val PPL | 3.71 |
| Copy | short | Recall PPL | 1.00 |
| Copy | medium | Recall PPL | 26.38 (fail) |
| Copy | long | Recall PPL | 26.05 (fail) |
| Induction | short | Accuracy | 1.00 |
| Induction | medium | Accuracy | 1.00 |
| Induction | long | Accuracy | 0.04 (fail) |

gMLP succeeds within its 256-token window, fails beyond it. The boundary is sharp.

## Training Times (NVIDIA A100, Google Colab)

| Task | Seq Length | Steps | Wall Clock |
|------|-----------|-------|------------|
| Shakespeare | 256 | 5,000 | 92s |
| Shakespeare | 1024 | 5,000 | 2363s |
| Shakespeare | 2048 | 5,000 | 2628s |
| Copy | short | 5,000 | 102s |
| Copy | medium | 5,000 | 831s |
| Copy | long | 5,000 | 2067s |
| Induction | short | 20,000 | 258s |
| Induction | medium | 20,000 | 2925s |
| Induction | long | 20,000 | 7793s |

gMLP is slowest at long sequences due to the unoptimised T×T spatial matrix. Batch size reduced to 32 at T=2048.

## Setup

```bash
pip install torch numpy
```

## Run

```bash
# Shakespeare
python3 train.py --task shakespeare --block_size 256 --seed 42 --max_steps 5000 --batch_size 64
python3 train.py --task shakespeare --block_size 1024 --seed 42 --max_steps 5000 --batch_size 64
python3 train.py --task shakespeare --block_size 2048 --seed 42 --max_steps 5000 --batch_size 32

# Copy
python3 train.py --task copy --length short --seed 42
python3 train.py --task copy --length medium --seed 42
python3 train.py --task copy --length long --seed 42 --batch_size 32

# Induction
python3 train.py --task induction --length short --seed 42 --max_steps 20000
python3 train.py --task induction --length medium --seed 42 --max_steps 20000
python3 train.py --task induction --length long --seed 42 --max_steps 20000 --batch_size 32
```

Or use `run_gmlp.ipynb` on Google Colab (A100 runtime).

## Files

| File | Purpose |
|------|---------|
| `model.py` | Causal gMLP with windowed SGU |
| `train.py` | Training script |
| `dataset_utils.py` | Data loading |
| `data_generator.py` | Synthetic data generation |
| `run_all_experiments.sh` | Shell script to run all 9 experiments |
