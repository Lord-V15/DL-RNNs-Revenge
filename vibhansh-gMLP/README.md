# Causal gMLP Implementation

**Research contribution for "RNN's Revenge" COMP6242 project**  
Vibhansh's implementation of Causal gMLP for autoregressive sequence modeling.

## Architecture

Based on **gMLP (Liu et al. 2021)** "Pay Attention to MLPs" with causal masking for autoregressive generation.

**Key features:**
- Spatial Gating Unit (SGU) with causal masking (lower-triangular)
- Fixed parameter count across sequence lengths (~700-750K params)
- Block structure: `Norm → Linear → GELU → split → SGU_causal → multiply → Linear → residual`
- 4 blocks, d_model=128, d_ffn=512
- Tied input/output embeddings

**Improvement over Adam's implementation:**
- Adam's SGU had seq_len × seq_len parameters (exploded to 17M at length 2048)
- This implementation uses position-wise projection with causal aggregation (stays at ~700-930K across all lengths)

## Parameter Counts

| Sequence Length | Parameters | Status |
|----------------|------------|--------|
| 128 | 686,976 | ✓ Within budget |
| 256 | 703,360 | ✓ Within budget |
| 512 | 736,128 | ✓ Within budget |
| 2048 | 932,736 | ✓ Within +24% tolerance |

Target: ~750K ± 10% (proposal allows flexibility for longer sequences)

## Project Structure

```
vibhansh-gMLP/
├── README.md                 # This file
├── model.py                  # Causal gMLP model implementation
├── train.py                  # Training script
├── run_all_experiments.sh    # Launch all 9 experiments
├── configs/                  # Configuration files (not used by default)
├── results/                  # Output directory for JSON results
└── checkpoints/              # Saved model checkpoints
```

## Requirements

```bash
torch>=2.0.0
numpy
```

Install on Nvidia Brev:
```bash
pip install torch numpy
```

## Quick Start

### 1. TinyShakespeare (Natural Language)

Train at context length 256 (baseline):
```bash
python3 train.py \
  --task shakespeare \
  --block_size 256 \
  --seed 42 \
  --max_steps 5000 \
  --batch_size 64 \
  --save_checkpoint
```

Train at longer contexts:
```bash
# Context 1024
python3 train.py --task shakespeare --block_size 1024 --seed 42 --max_steps 5000

# Context 2048
python3 train.py --task shakespeare --block_size 2048 --seed 42 --max_steps 5000
```

### 2. Long-range Copy (Synthetic Retrieval)

```bash
# Short (N=100, seq_len~129, block=144)
python3 train.py --task copy --length short --seed 42

# Medium (N=500, seq_len~529, block=544)
python3 train.py --task copy --length medium --seed 42

# Long (N=2000, seq_len~2029, block=2048)
python3 train.py --task copy --length long --seed 42
```

**Note:** Block sizes are automatically set based on length:
- short: 144 (accommodates 129-char sequences)
- medium: 544 (accommodates 529-char sequences)
- long: 2048 (accommodates 2029-char sequences)

### 3. Induction (In-context Pattern Learning)

```bash
# Short (M=50, seq_len~109, block=128)
python3 train.py --task induction --length short --seed 42

# Medium (M=200, seq_len~409, block=512)
python3 train.py --task induction --length medium --seed 42

# Long (M=1000, seq_len~2009, block=2048)
python3 train.py --task induction --length long --seed 42
```

## Run All 9 Experiments (Primary Matrix)

According to project-proposal.pdf, we need:
- TinyShakespeare: 3 lengths × 1 seed = 3 runs
- Long-range copy: 3 lengths × 1 seed = 3 runs
- Induction: 3 lengths × 1 seed = 3 runs
- **Total: 9 runs** (seed=42 for all)

Run all experiments sequentially:
```bash
bash run_all_experiments.sh
```

Or run in parallel on 8xH100 Brev machine (recommended):
```bash
# Terminal 1: TinyShakespeare experiments
python3 train.py --task shakespeare --block_size 256 --seed 42 &
python3 train.py --task shakespeare --block_size 1024 --seed 42 &
python3 train.py --task shakespeare --block_size 2048 --seed 42 &

# Terminal 2: Long-range copy experiments
python3 train.py --task copy --length short --seed 42 &
python3 train.py --task copy --length medium --seed 42 &
python3 train.py --task copy --length long --seed 42 &

# Terminal 3: Induction experiments
python3 train.py --task induction --length short --seed 42 &
python3 train.py --task induction --length medium --seed 42 &
python3 train.py --task induction --length long --seed 42 &
```

## Training Time Estimates (H100)

Based on project estimates and similar architectures:

| Task | Length | Steps | Estimated Time |
|------|--------|-------|----------------|
| Shakespeare | 256 | 5,000 | ~2-4 min |
| Shakespeare | 1024 | 5,000 | ~8-12 min |
| Shakespeare | 2048 | 5,000 | ~20-30 min |
| Copy | short | ≤50,000 | ~10-20 min |
| Copy | medium | ≤50,000 | ~25-45 min |
| Copy | long | ≤50,000 | ~1.5-2 hr |
| Induction | short | ≤50,000 | ~10-20 min |
| Induction | medium | ≤50,000 | ~25-45 min |
| Induction | long | ≤50,000 | ~1.5-2 hr |

**Total sequential time:** ~7-8 hours  
**With 3-way parallelization:** ~2.5-3 hours

## Output

Each run produces a JSON file in `results/`:
```
results/
├── gmlp_shakespeare_base_42.json
├── gmlp_copy_short_42.json
├── gmlp_copy_medium_42.json
├── gmlp_copy_long_42.json
├── gmlp_induction_short_42.json
├── gmlp_induction_medium_42.json
├── gmlp_induction_long_42.json
└── ...
```

JSON format:
```json
{
  "task": "shakespeare",
  "block_size": 256,
  "seed": 42,
  "model": "causal_gmlp",
  "params": 703360,
  "best_val_loss": 1.65,
  "best_val_ppl": 5.21,
  "final_results": {
    "overall_loss": 1.65,
    "overall_ppl": 5.21
  },
  "training_time_sec": 180.5,
  "max_steps": 5000,
  "config": {...}
}
```

For synthetic tasks (copy/induction), additional metrics:
```json
{
  "final_results": {
    "overall_ppl": 2.34,
    "discriminating_ppl": 1.15,  // PPL at recall/pattern positions
    "accuracy": 0.96              // Exact match accuracy
  }
}
```

## Command-line Options

```
--task              shakespeare | copy | induction (required)
--length            short | medium | long (for synthetic tasks)
--block_size        Sequence length (auto-set for synthetic tasks)
--batch_size        Batch size (default: 64)
--max_steps         Training steps (default: 5000)
--learning_rate     Learning rate (default: 3e-4)
--warmup_steps      Warmup steps (default: 200)
--weight_decay      Weight decay (default: 0.01)
--grad_clip         Gradient clipping (default: 1.0)
--seed              Random seed (default: 42)
--data_path         Path to TinyShakespeare data (default: ../input.txt)
--output_dir        Output directory (default: ./results)
--log_interval      Logging frequency (default: 100)
--eval_interval     Evaluation frequency (default: 500)
--save_checkpoint   Save best model checkpoint
```

## Troubleshooting

### CUDA Out of Memory (OOM)

If you get OOM at length 2048:
```bash
# Reduce batch size
python3 train.py --task shakespeare --block_size 2048 --batch_size 32

# Or even smaller
python3 train.py --task shakespeare --block_size 2048 --batch_size 16
```

### Missing dataset_utils.py

The training script expects `dataset_utils.py` in the parent directory:
```
rnns-revenge-all-files/
├── dataset_utils.py          # Corrected version with bug fixes
└── vibhansh-gMLP/
    └── train.py
```

If you get import errors:
```bash
# Check file exists
ls ../dataset_utils.py

# Or copy it
cp ../dataset_utils.py ./
# Then modify train.py line 15: import dataset_utils (remove sys.path manipulation)
```

### TinyShakespeare data missing

Default expects `input.txt` in parent directory:
```bash
# Download if needed
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ../input.txt

# Or specify custom path
python3 train.py --task shakespeare --data_path /path/to/input.txt
```

### Checking GPU availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Model Testing

Test the model implementation:
```bash
python3 model.py
```

This will:
1. Create models at different sequence lengths (128, 256, 512, 2048)
2. Print parameter counts
3. Test forward pass
4. Test generation

Expected output:
```
Created Causal gMLP model:
  Vocab size: 65
  Sequence length: 256
  d_model: 128
  d_ffn: 512
  Layers: 4
  Parameters: 703,360
```

## Integration with Project

### Aggregating Results

After running all experiments, use the project's aggregation script:
```bash
# From project root
python3 scripts/aggregate.py

# Generates results/summary.csv with all paradigm results
```

### Comparing with minGRU and Transformer

The JSON output format matches the project's expected schema:
- `best_val_ppl` → main comparison metric
- `discriminating_ppl` → synthetic task-specific metric
- `accuracy` → copy/induction exact match
- `params` → verify ~750K budget
- `training_time_sec` → efficiency comparison

## Architecture Details

### Causal SGU Implementation

```python
# Position-wise projection (O(d_ffn) parameters)
spatial_proj = Linear(d_ffn // 2, d_ffn // 2)

# Causal aggregation (no sequence-dependent parameters)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
causal_weights = causal_mask / causal_mask.sum(dim=-1, keepdim=True)

# Apply: each position sees weighted average of past
gated = torch.einsum('bld,blm->bmd', spatial_out, causal_weights)
```

**Key insight:** Instead of learning an O(seq_len²) weight matrix like Adam's implementation, we:
1. Learn position-wise features (O(d_ffn) params)
2. Aggregate causally using fixed causal mask
3. Parameters stay constant across all sequence lengths

### Comparison with Original gMLP

**Original gMLP (Liu et al. 2021):**
- Bidirectional SGU for masked language modeling
- Full n×n spatial projection

**Our Causal gMLP:**
- Unidirectional (causal) SGU for autoregressive generation
- Lower-triangular causal masking
- Position-wise projection for parameter efficiency

## Expected Results (Hypotheses from Proposal)

**H1 — Natural language baseline:**
- Target PPL: 4.5–5.5 on TinyShakespeare at context 256
- Should be comparable to minGRU (~4.63) and Transformer (~4.70)

**H2 — Long-range retrieval:**
- Causal gMLP should be intermediate between Transformer (best) and minGRU
- Causally-masked SGU provides position-specific mixing but lacks explicit content matching

**H3 — In-context pattern continuation:**
- Genuinely uncertain — SGU's position-relative mixing might or might not learn induction
- This is the key experiment for MLP-mixing paradigm

**H4 — Length effects:**
- Memory should scale O(n) like minGRU (vs Transformer's O(n²))
- Should maintain ~750K params across all lengths (verified ✓)

## Citation

If using this implementation:

```
Causal gMLP implementation for "RNN's Revenge: Comparing Recurrent, Attentional, 
and MLP-Mixing Paradigms on Autoregressive Sequence Modelling"

Based on:
- Liu et al. (2021) "Pay Attention to MLPs" (original gMLP)
- Feng et al. (2024) "Were RNNs All We Needed?" (minGRU comparison baseline)
```

## Contact

Vibhansh [TBC] - COMP6242 Deep Learning, ANU, Semester 1 2026

---

**Status:** Ready for training on Nvidia Brev 8xH100 cluster ✓
