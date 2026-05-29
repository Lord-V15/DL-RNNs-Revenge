"""
Single-GPU trainer for the COMP6242 paradigm comparison project.

Logs everything required by proposal §7:
  - Best validation PPL
  - Peak GPU memory
  - Throughput (tokens / sec)
  - Wall-clock time
  - Full hyperparameters
  - Git SHA, timestamp, seed

One JSON file per run is written to <out_dir>/summary.json.

Usage:
    python train.py config/tshake_256.py
    python train.py config/tshake_256.py --seed=1337 --max_iters=2000
"""

import os
import sys
import json
import math
import time
import pickle
import random
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch

from model import GPT, GPTConfig


# ----- defaults (overridden by config file + CLI) -----------------------------
# I/O
out_dir = "out"
eval_interval = 250
log_interval = 50
eval_iters = 100
always_save_checkpoint = False  # only save when val improves
run_name = "run"
# data
dataset = "tinyshakespeare"
batch_size = 64
block_size = 256
# model
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
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
# system
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
compile_model = False  # torch.compile - keep off by default for short runs
# task tag (for metadata)
task = "tinyshakespeare"

# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # apply config file overrides + CLI
config = {k: globals()[k] for k in config_keys}

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
# data
data_dir = os.path.join("data", dataset)


def get_batch(split):
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# read vocab_size from meta.pkl
meta_path = os.path.join(data_dir, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
print(f"loaded {dataset}: vocab_size={vocab_size}")

# -----------------------------------------------------------------------------
# model
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=vocab_size,
    dropout=dropout,
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

compile_model = True
if compile_model:
    print("compiling model...")
    model = torch.compile(model)

# bf16 doesn't need GradScaler; fp16 does
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))


# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
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
print(f"\n=== run: {run_name} | task: {task} | block_size: {block_size} | seed: {seed} ===")
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()

X, Y = get_batch("train")
best_val_loss = float("inf")
best_val_iter = 0
t_start = time.time()
total_tokens = 0

for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g["lr"] = lr

    # eval
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        train_loss, val_loss = losses["train"], losses["val"]
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - t_start
        print(f"step {iter_num:5d} | lr {lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f} | val_ppl {val_ppl:.3f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iter = iter_num
            if always_save_checkpoint and iter_num > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == max_iters:
        break

    # forward / backward
    with ctx:
        _, loss = model(X, Y)
    X, Y = get_batch("train")  # async prefetch
    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    total_tokens += batch_size * block_size

    if iter_num % log_interval == 0:
        print(f"  iter {iter_num:5d} | loss {loss.item():.4f}")

# -----------------------------------------------------------------------------
# summary
wall_clock = time.time() - t_start
throughput = total_tokens / wall_clock if wall_clock > 0 else 0.0
peak_mem_mb = (
    torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device_type == "cuda" else 0.0
)

summary = {
    "run_name": run_name,
    "task": task,
    "block_size": block_size,
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
}

summary_path = os.path.join(out_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== done ===")
print(f"best val ppl: {summary['best_val_ppl']:.3f} @ step {best_val_iter}")
print(f"wall clock:   {wall_clock:.1f}s")
print(f"throughput:   {throughput:,.0f} tok/s")
print(f"peak mem:     {peak_mem_mb:.1f} MB")
print(f"summary -> {summary_path}")
