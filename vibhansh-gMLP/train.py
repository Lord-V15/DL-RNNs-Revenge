"""
Training script for Causal gMLP
Supports: TinyShakespeare, Long-range Copy, Induction tasks
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import time
import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model import create_causal_gmlp
from dataset_utils import LongRangeCopyDataset, InductionDataset


class CharDataset(Dataset):
    """Character-level dataset for TinyShakespeare"""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_batch(loader, device):
    """Get next batch from dataloader"""
    try:
        batch = next(loader)
    except StopIteration:
        loader = iter(loader)
        batch = next(loader)
    x, y = batch
    return x.to(device), y.to(device), loader


def train_step(model, optimizer, scheduler, x, y, grad_clip=1.0):
    """Single training step"""
    model.train()
    optimizer.zero_grad()

    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        if isinstance(batch, dict):
            x, y = batch['input_ids'].to(device), batch['labels'].to(device)
        elif len(batch) == 3:
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x, y = batch[0].to(device), batch[1].to(device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Count non-padding tokens
        mask = (y != -1)
        n_tokens = mask.sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return avg_loss, perplexity


@torch.no_grad()
def evaluate_with_metrics(model, val_loader, device, task_type='copy'):
    """
    Evaluate with task-specific metrics (for copy/induction tasks)

    Returns:
        dict with overall_ppl, discriminating_ppl, accuracy (for copy/induction)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    discriminating_loss = 0.0
    discriminating_tokens = 0
    correct_predictions = 0

    for batch in val_loader:
        if isinstance(batch, dict):
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            mask_key = 'recall_positions' if task_type == 'copy' else 'pattern_positions'
            mask = batch.get(mask_key)
            if mask is not None:
                mask = mask.to(device)
        elif len(batch) == 3:
            x, y, mask = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            mask = None

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Overall metrics
        valid_mask = (y != -1)
        n_tokens = valid_mask.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Discriminating metrics (if mask provided)
        if mask is not None:
            discriminating_mask = mask.bool() & valid_mask
            if discriminating_mask.any():
                disc_logits = logits[discriminating_mask]
                disc_targets = y[discriminating_mask]

                disc_loss = F.cross_entropy(disc_logits, disc_targets)
                discriminating_loss += disc_loss.item() * discriminating_mask.sum().item()
                discriminating_tokens += discriminating_mask.sum().item()

                # Accuracy
                preds = disc_logits.argmax(dim=-1)
                correct_predictions += (preds == disc_targets).sum().item()

    results = {
        'overall_loss': total_loss / total_tokens if total_tokens > 0 else 0.0,
        'overall_ppl': math.exp(total_loss / total_tokens) if total_tokens > 0 and total_loss / total_tokens < 20 else float('inf')
    }

    if discriminating_tokens > 0:
        disc_avg_loss = discriminating_loss / discriminating_tokens
        results['discriminating_ppl'] = math.exp(disc_avg_loss) if disc_avg_loss < 20 else float('inf')
        results['accuracy'] = correct_predictions / discriminating_tokens

    return results


def train(config):
    """Main training loop"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Load data based on task
    task = config['task']
    block_size = config['block_size']

    if task == 'shakespeare':
        # Load TinyShakespeare
        data_path = config.get('data_path', '../input.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create character mapping
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {vocab_size}")

        # Encode data
        data = [stoi[c] for c in text]

        # Split
        split_idx = int(0.9 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # Create datasets
        train_dataset = CharDataset(train_data, block_size)
        val_dataset = CharDataset(val_data, block_size)

    elif task == 'copy':
        length_key = config.get('length', 'short')
        data_dir = config.get('data_dir', './data')
        print(f"Loading long-range copy task ({length_key}) from {data_dir}...")

        block_sizes = {'short': 144, 'medium': 544, 'long': 2048}
        block_size = block_sizes[length_key]
        config['block_size'] = block_size

        train_dataset = LongRangeCopyDataset(
            f"{data_dir}/longrange_copy/train_{length_key}.txt", block_size=block_size)
        val_dataset = LongRangeCopyDataset(
            f"{data_dir}/longrange_copy/val_{length_key}.txt", block_size=block_size)
        vocab_size = train_dataset.tokenizer.vocab_size

    elif task == 'induction':
        length_key = config.get('length', 'short')
        data_dir = config.get('data_dir', './data')
        print(f"Loading induction task ({length_key}) from {data_dir}...")

        block_sizes = {'short': 128, 'medium': 512, 'long': 2048}
        block_size = block_sizes[length_key]
        config['block_size'] = block_size

        train_dataset = InductionDataset(
            f"{data_dir}/induction/train_{length_key}.txt",
            pattern_file=f"{data_dir}/induction/train_{length_key}_patterns.txt",
            block_size=block_size)
        val_dataset = InductionDataset(
            f"{data_dir}/induction/val_{length_key}.txt",
            pattern_file=f"{data_dir}/induction/val_{length_key}_patterns.txt",
            block_size=block_size)
        vocab_size = train_dataset.tokenizer.vocab_size

    else:
        raise ValueError(f"Unknown task: {task}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = create_causal_gmlp(
        vocab_size=vocab_size,
        seq_len=block_size,
        config='base',
        dropout=config.get('dropout', 0.2)
    )
    model = model.to(device)

    print(f"\nModel has {model.count_parameters():,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    warmup_steps = config['warmup_steps']
    max_steps = config['max_steps']
    lr_decay_steps = config.get('lr_decay_steps', None) or max_steps

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay over lr_decay_steps (may be longer than max_steps)
            progress = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
            progress = min(progress, 1.0)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\nStarting training for {max_steps} steps...")
    print(f"Task: {task}, Block size: {block_size}, Batch size: {config['batch_size']}")

    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    start_time = time.time()

    for step in range(max_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if isinstance(batch, dict):
            x, y = batch['input_ids'].to(device), batch['labels'].to(device)
        else:
            x, y = batch[0].to(device), batch[1].to(device)

        # Train step
        loss = train_step(model, optimizer, scheduler, x, y, config['grad_clip'])

        # Logging
        if step % config['log_interval'] == 0 or step == max_steps - 1:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d}/{max_steps} | Loss: {loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Evaluation
        if step % config['eval_interval'] == 0 or step == max_steps - 1:
            val_loss, val_ppl = evaluate(model, val_loader, device, max_batches=50)
            print(f"  Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if config.get('save_checkpoint'):
                    checkpoint_path = os.path.join(config['output_dir'], 'best_model.pt')
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': config
                    }, checkpoint_path)
                    print(f"  Saved checkpoint to {checkpoint_path}")

            model.train()

    # Final evaluation
    print("\nFinal evaluation on full validation set...")
    if task in ['copy', 'induction']:
        final_results = evaluate_with_metrics(model, val_loader, device, task_type=task)
        print(f"  Overall PPL: {final_results['overall_ppl']:.4f}")
        if 'discriminating_ppl' in final_results:
            print(f"  Discriminating PPL: {final_results['discriminating_ppl']:.4f}")
            print(f"  Accuracy: {final_results['accuracy']:.4f}")
    else:
        final_loss, final_ppl = evaluate(model, val_loader, device)
        final_results = {'overall_loss': final_loss, 'overall_ppl': final_ppl}
        print(f"  Loss: {final_loss:.4f} | PPL: {final_ppl:.4f}")

    # Save results
    total_time = time.time() - start_time
    results = {
        'task': task,
        'block_size': block_size,
        'seed': config['seed'],
        'model': 'causal_gmlp',
        'params': model.count_parameters(),
        'best_val_loss': best_val_loss,
        'best_val_ppl': math.exp(best_val_loss) if best_val_loss < 20 else float('inf'),
        'final_results': final_results,
        'training_time_sec': total_time,
        'max_steps': max_steps,
        'config': config
    }

    # Save JSON
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if task == 'shakespeare':
        length_label = str(config['block_size'])
    else:
        length_label = config.get('length', 'base')
    result_file = os.path.join(
        output_dir,
        f"gmlp_{task}_{length_label}_{config['seed']}.json"
    )

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Causal gMLP')
    parser.add_argument('--task', type=str, required=True,
                        choices=['shakespeare', 'copy', 'induction'],
                        help='Task to train on')
    parser.add_argument('--length', type=str, default='short',
                        choices=['short', 'medium', 'long'],
                        help='Sequence length for synthetic tasks')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Block size (sequence length)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=200,
                        help='Warmup steps')
    parser.add_argument('--lr_decay_steps', type=int, default=None,
                        help='Total steps for LR cosine decay (defaults to max_steps)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_path', type=str, default='./data/tinyshakespeare/input.txt',
                        help='Path to data file (for shakespeare)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to shared data directory (for copy/induction)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluation interval')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save model checkpoints')

    args = parser.parse_args()

    # Set block size based on task and length
    if args.task == 'copy':
        length_map = {'short': 144, 'medium': 544, 'long': 2048}
        args.block_size = length_map[args.length]
    elif args.task == 'induction':
        length_map = {'short': 128, 'medium': 512, 'long': 2048}
        args.block_size = length_map[args.length]

    # Convert args to config dict
    config = vars(args)

    print("=" * 80)
    print("Causal gMLP Training")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)

    # Train
    train(config)
