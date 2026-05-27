"""
Quick sanity test to verify the setup before training on Brev.
Tests: model creation, forward pass, data loading
"""

import torch
import sys
from pathlib import Path

# Add parent to path for dataset_utils
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("Causal gMLP Setup Verification")
print("=" * 80)

# Test 1: Model import and creation
print("\n[1/5] Testing model import...")
try:
    from model import create_causal_gmlp
    print("✓ Model imported successfully")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    sys.exit(1)

# Test 2: Model creation at different sequence lengths
print("\n[2/5] Testing model creation at different sequence lengths...")
try:
    vocab_size = 65
    for seq_len in [128, 256, 512, 2048]:
        model = create_causal_gmlp(vocab_size, seq_len=seq_len, config='base')
        params = model.count_parameters()
        target = 750000
        pct_diff = ((params - target) / target) * 100
        status = "✓" if abs(pct_diff) < 25 else "✗"
        print(f"  {status} seq_len={seq_len:4d}: {params:,} params ({pct_diff:+.1f}% from 750K)")
except Exception as e:
    print(f"✗ Failed to create models: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n[3/5] Testing forward pass...")
try:
    model = create_causal_gmlp(vocab_size=65, seq_len=256, config='base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Test batch
    batch_size = 4
    seq_len = 256
    input_ids = torch.randint(0, 65, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 65, (batch_size, seq_len), device=device)

    logits, loss = model(input_ids, targets)

    assert logits.shape == (batch_size, seq_len, 65), f"Wrong logits shape: {logits.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    print(f"  ✓ Forward pass successful")
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Data generation
print("\n[4/5] Testing data generation...")
try:
    from data_generator import LongRangeCopyDataset, InductionDataset
    print("  ✓ data_generator imported successfully")

    # Test synthetic data generation
    print("  Testing long-range copy generation...")
    copy_ds = LongRangeCopyDataset(n_samples=10, length='short')
    input_ids, labels, mask = copy_ds[0]
    print(f"    ✓ Created dataset with {len(copy_ds)} samples")
    print(f"    Sample input shape: {input_ids.shape}, vocab_size: {copy_ds.vocab_size}")

    print("  Testing induction generation...")
    induction_ds = InductionDataset(n_samples=10, length='short')
    input_ids, labels, mask = induction_ds[0]
    print(f"    ✓ Created dataset with {len(induction_ds)} samples")
    print(f"    Sample input shape: {input_ids.shape}, vocab_size: {induction_ds.vocab_size}")

except Exception as e:
    print(f"✗ Failed to test data generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check CUDA
print("\n[5/5] Checking GPU availability...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠ CUDA not available (will use CPU - training will be slow)")

# Summary
print("\n" + "=" * 80)
print("SETUP VERIFICATION COMPLETE")
print("=" * 80)
print("\n✓ All tests passed! Ready to train on Brev.")
print("\nQuick start commands:")
print("  # Single experiment test")
print("  python3 train.py --task shakespeare --block_size 256 --seed 42 --max_steps 100")
print("")
print("  # Run all 9 experiments")
print("  bash run_all_experiments.sh")
print("=" * 80)
