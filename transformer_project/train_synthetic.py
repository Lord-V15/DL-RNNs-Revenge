"""
Single-GPU trainer for the synthetic tasks (long-range copy, induction).

Uses Daniel's existing pre-generated text files in data/longrange_copy/
and data/induction/, loaded via dataset_utils.py. Eval reports both
overall PPL and the task-specific discriminating metric (recall_ppl for
long-range copy, pattern_ppl + pattern_accuracy for induction), as
required by proposal v4.0 §7.

Usage:
    python train_synthetic.py config/lrcopy_medium.py
    python train_synthetic.py config/induction_medium.py --max_iters=500   # pre-flight

Writes <out_dir>/summary.json with all §7 metadata + discriminating metrics.
"""

import os
import json
import math
import time
import random
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GPT, GPTConfig
from dataset_utils import LongRangeCopyDataset, InductionDataset
from evaluation_metrics import compute_longrange_metrics, compute_induction_metrics


# ----- defaults (overridden by config file + CLI) -----------------------------
# I/O
out_dir = "out"
eval_interval = 250
log_interval = 50
eval_iters = 50           # number of val batches per eval pass
always_save_checkpoint = True
run_name = "run"

# task (REQUIRED in config)
# task_type: "longrange_copy" or "induction"
# length:    "short" | "medium" | "long"
task_type = "longrange_copy"
length = "medium"

# data
batch_size = 64
# block_size, vocab_size are derived from task_type+length below — don't set here

# model (same as TinyShakespeare runs)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2
bias = False

# optimisation (proposal §5)
learning_rate = 3e-4
min_lr = 3e-5
max_iters = 5000
warmup_iters = 200
# Decay schedule length, decoupled from max_iters. Defaults to max_iters (so
# behaviour is unchanged unless overridden). Set this to follow a fixed cosine
# curve regardless of how many steps you actually train, e.g. train 20K steps
# but decay as if over 50K:  --max_iters=20000 --lr_decay_iters=50000
lr_decay_iters = -1   # -1 -> falls back to max_iters after config is applied
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# system
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
compile_model = False

# task tag for metadata (auto-set, but configurator may override)
task = "synthetic"

# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
# Resolve LR decay length: -1 sentinel means "use max_iters" (default behaviour).
if lr_decay_iters is None or lr_decay_iters < 0:
    lr_decay_iters = max_iters
config = {k: globals()[k] for k in config_keys}
config["lr_decay_iters"] = lr_decay_iters  # ensure resolved value is logged

# -----------------------------------------------------------------------------
# Derive block_size and vocab_size from task_type + length
#
# Proposal §4.2 block sizes for long-range copy: short=128, medium=528, long=2048
# Proposal §4.3 block sizes for induction:       short=128, medium=512, long=2048
# Vocabularies: copy=55 chars (lowercase+uppercase+space+colon+pipe),
#               induction=27 chars (lowercase+space)

if task_type == "longrange_copy":
    # Bumped from proposal {128, 528, 2048} to {144, 544, 2048}: actual sequences
    # are 129/529/2029 chars, so 128/528 truncate the final char of the recall key.
    block_sizes = {"short": 144, "medium": 544, "long": 2048}
    vocab_size = 55
    data_dir = "data/longrange_copy"
elif task_type == "induction":
    block_sizes = {"short": 128, "medium": 512, "long": 2048}
    vocab_size = 27
    data_dir = "data/induction"
else:
    raise ValueError(f"task_type must be 'longrange_copy' or 'induction', got {task_type!r}")

if length not in block_sizes:
    raise ValueError(f"length must be 'short'/'medium'/'long', got {length!r}")

block_size = block_sizes[length]
print(f"task_type={task_type}, length={length} -> block_size={block_size}, vocab_size={vocab_size}")

# -----------------------------------------------------------------------------
# seed everything
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data: load Daniel's pre-generated text files via his Dataset classes
train_file = os.path.join(data_dir, f"train_{length}.txt")
val_file = os.path.join(data_dir, f"val_{length}.txt")

if task_type == "longrange_copy":
    train_ds = LongRangeCopyDataset(train_file, block_size=block_size)
    val_ds = LongRangeCopyDataset(val_file, block_size=block_size)
    position_key = "recall_positions"
else:  # induction
    train_pat = os.path.join(data_dir, f"train_{length}_patterns.txt")
    val_pat = os.path.join(data_dir, f"val_{length}_patterns.txt")
    train_ds = InductionDataset(train_file, pattern_file=train_pat, block_size=block_size)
    val_ds = InductionDataset(val_file, pattern_file=val_pat, block_size=block_size)
    position_key = "pattern_positions"

# Sanity check tokenizer vocab size matches our config
if train_ds.tokenizer.vocab_size != vocab_size:
    raise RuntimeError(
        f"vocab mismatch: tokenizer has {train_ds.tokenizer.vocab_size} chars "
        f"but task_type={task_type} expects {vocab_size}"
    )

print(f"train sequences: {len(train_ds):,} | val sequences: {len(val_ds):,}")
print(f"tokenizer vocab_size: {train_ds.tokenizer.vocab_size}")

# Infinite training loader (we step by iteration, not epoch)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=0, drop_last=True, pin_memory=(device_type == "cuda")
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=0, drop_last=False, pin_memory=(device_type == "cuda")
)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


train_iter = cycle(train_loader)

# -----------------------------------------------------------------------------
# model
# Note: model.block_size needs to be >= the dataset's input length (block_size - 1)
gptconf = GPTConfig(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=vocab_size,
    dropout=dropout,
)
model = GPT(gptconf)
model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

compile_model = True
if compile_model:
    print("compiling model...")
    model = torch.compile(model)

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))


# -----------------------------------------------------------------------------
def to_device(batch):
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate():
    """Run full validation pass; return overall PPL, train_loss, and discriminating metric(s)."""
    model.eval()

    # Validation pass (whole val set, or up to eval_iters batches if huge)
    val_losses, recall_losses, pattern_losses, pattern_correct, pattern_total = [], [], [], 0, 0
    # 5th-char-only induction signal: accuracy at the single P[4] position
    induction5_correct, induction5_total = 0, 0

    n_batches = 0
    for batch in val_loader:
        batch = to_device(batch)
        with ctx:
            logits, loss = model(batch["input_ids"], batch["labels"])
        val_losses.append(loss.item())

        # discriminating metric (per-position)
        positions = batch[position_key]
        if positions.sum() > 0:
            sel_logits = logits[positions]
            sel_labels = batch["labels"][positions]
            sel_loss = torch.nn.functional.cross_entropy(sel_logits, sel_labels, reduction="mean")
            if task_type == "longrange_copy":
                recall_losses.append(sel_loss.item())
            else:
                pattern_losses.append(sel_loss.item())
                pattern_correct += (sel_logits.argmax(dim=-1) == sel_labels).sum().item()
                pattern_total += sel_labels.numel()
                # 5th-char-only: gather logits at induction_target_pos per row
                tgt = batch["induction_target_pos"]  # (B,) long tensor
                valid = tgt >= 0
                if valid.any():
                    rows = torch.arange(logits.size(0), device=logits.device)[valid]
                    cols = tgt[valid]
                    tgt_logits = logits[rows, cols]                 # (n_valid, vocab)
                    tgt_labels = batch["labels"][rows, cols]        # (n_valid,)
                    induction5_correct += (tgt_logits.argmax(dim=-1) == tgt_labels).sum().item()
                    induction5_total += tgt_labels.numel()

        n_batches += 1
        if n_batches >= eval_iters:
            break

    # Quick training-loss estimate (a few train batches, like the TinyShakespeare trainer does)
    train_losses = []
    train_loader_finite = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    for i, batch in enumerate(train_loader_finite):
        if i >= eval_iters:
            break
        batch = to_device(batch)
        with ctx:
            _, loss = model(batch["input_ids"], batch["labels"])
        train_losses.append(loss.item())

    model.train()

    out = {
        "train_loss": float(np.mean(train_losses)),
        "val_loss": float(np.mean(val_losses)),
        "val_ppl": float(np.exp(np.mean(val_losses))),
    }
    if task_type == "longrange_copy":
        out["recall_ppl"] = float(np.exp(np.mean(recall_losses))) if recall_losses else float("inf")
    else:
        out["pattern_ppl"] = float(np.exp(np.mean(pattern_losses))) if pattern_losses else float("inf")
        out["pattern_accuracy"] = pattern_correct / pattern_total if pattern_total else 0.0
        out["induction5_accuracy"] = induction5_correct / induction5_total if induction5_total else 0.0
    return out


def get_lr(it):
    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) past the end of the decay schedule -> floor at min_lr
    if it > lr_decay_iters:
        return min_lr
    # 3) cosine decay over [warmup_iters, lr_decay_iters], clamped to [0, 1]
    #    so the curve is well-defined even when max_iters < lr_decay_iters
    #    (train fewer steps but follow the longer schedule's shape).
    denom = max(1, lr_decay_iters - warmup_iters)
    decay_ratio = (it - warmup_iters) / denom
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# training loop
print(f"\n=== run: {run_name} | task: {task_type}-{length} | block_size: {block_size} | seed: {seed} ===")
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()

best_val_loss = float("inf")
best_val_iter = 0
best_discriminating = float("inf")           # recall_ppl or pattern_ppl
best_pattern_accuracy = 0.0                  # all-5 positions (induction only)
best_induction5_accuracy = 0.0               # 5th-char-only (induction only)
eval_history = []                            # per-eval record for plotting curves
t_start = time.time()
total_tokens = 0

for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g["lr"] = lr

    # eval
    if iter_num % eval_interval == 0:
        m = evaluate()
        # record this eval for the curve (step + all metrics)
        rec = {
            "step": iter_num,
            "lr": lr,
            "train_loss": m["train_loss"],
            "val_loss": m["val_loss"],
            "val_ppl": m["val_ppl"],
            "elapsed_sec": time.time() - t_start,
        }
        if task_type == "longrange_copy":
            rec["recall_ppl"] = m["recall_ppl"]
        else:
            rec["pattern_ppl"] = m["pattern_ppl"]
            rec["pat_acc5"] = m["induction5_accuracy"]
            rec["pat_acc_all"] = m["pattern_accuracy"]
        eval_history.append(rec)

        if task_type == "longrange_copy":
            disc_name, disc_val = "recall_ppl", m["recall_ppl"]
            print(f"step {iter_num:5d} | lr {lr:.2e} | train {m['train_loss']:.4f} | "
                  f"val {m['val_loss']:.4f} | val_ppl {m['val_ppl']:.3f} | "
                  f"recall_ppl {disc_val:.3f} | {time.time()-t_start:.1f}s")
        else:
            disc_name, disc_val = "pattern_ppl", m["pattern_ppl"]
            print(f"step {iter_num:5d} | lr {lr:.2e} | train {m['train_loss']:.4f} | "
                  f"val {m['val_loss']:.4f} | val_ppl {m['val_ppl']:.3f} | "
                  f"pattern_ppl {disc_val:.3f} | pat_acc5 {m['induction5_accuracy']:.3f} | "
                  f"pat_acc_all {m['pattern_accuracy']:.3f} | "
                  f"{time.time()-t_start:.1f}s")

        if m["val_loss"] < best_val_loss:
            best_val_loss = m["val_loss"]
            best_val_iter = iter_num
            best_discriminating = disc_val
            if task_type == "induction":
                best_pattern_accuracy = m["pattern_accuracy"]
                best_induction5_accuracy = m["induction5_accuracy"]
            if always_save_checkpoint and iter_num > 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == max_iters:
        break

    # forward / backward
    batch = to_device(next(train_iter))
    with ctx:
        _, loss = model(batch["input_ids"], batch["labels"])
    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # accumulate token count (input length = block_size - 1)
    total_tokens += batch_size * (block_size - 1)

    if iter_num % log_interval == 0:
        print(f"  iter {iter_num:5d} | loss {loss.item():.4f}")

# -----------------------------------------------------------------------------
# summary
wall_clock = time.time() - t_start
throughput = total_tokens / wall_clock if wall_clock > 0 else 0.0
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device_type == "cuda" else 0.0

summary = {
    "run_name": run_name,
    "task": task_type,
    "length": length,
    "block_size": block_size,
    "vocab_size": vocab_size,
    "seed": seed,
    "n_params": n_params,
    "best_val_loss": best_val_loss,
    "best_val_ppl": math.exp(best_val_loss),
    "best_val_iter": best_val_iter,
    "wall_clock_sec": wall_clock,
    "throughput_tok_per_sec": throughput,
    "peak_gpu_mem_mb": peak_mem_mb,
    "device": device,
    "dtype": dtype,
    "git_sha": git_sha(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hyperparams": config,
    "eval_history": eval_history,
}

if task_type == "longrange_copy":
    summary["best_recall_ppl"] = best_discriminating
else:
    summary["best_pattern_ppl"] = best_discriminating
    summary["best_pattern_accuracy"] = best_pattern_accuracy
    summary["best_induction5_accuracy"] = best_induction5_accuracy

summary_path = os.path.join(out_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== done ===")
print(f"best val ppl: {summary['best_val_ppl']:.3f} @ step {best_val_iter}")
if task_type == "longrange_copy":
    print(f"best recall ppl: {summary['best_recall_ppl']:.3f}  <-- discriminating signal")
else:
    print(f"best pattern ppl: {summary['best_pattern_ppl']:.3f}  <-- discriminating signal")
    print(f"best induction acc (5th char only): {summary['best_induction5_accuracy']:.3f}  <-- the real induction signal")
    print(f"best pattern acc (all 5 positions): {summary['best_pattern_accuracy']:.3f}")
print(f"wall clock:   {wall_clock:.1f}s")
print(f"throughput:   {throughput:,.0f} tok/s")
print(f"peak mem:     {peak_mem_mb:.1f} MB")
print(f"summary -> {summary_path}")
