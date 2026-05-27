"""Training loop.

Orchestrates:
  * dataset construction (TinyShakespeare / LongRangeCopy / Induction)
  * model construction (minGRU from configs.model_mingru)
  * optimizer (AdamW) + scheduler (warmup + cosine)
  * training loop with bf16 autocast, gradient clipping, periodic eval
  * checkpointing (best.pt + last.pt) and CSV/console/meta logging
  * memory + throughput tracking once per eval

The single public entry point is `train(cfg, run_dir, resume=False)`. The
caller (scripts/train.py) is responsible for resolving the run directory
name and the experiment lookup; this module is task-agnostic except for the
small dispatch on `cfg.task`.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs.base import TrainConfig
from src.data.tinyshakespeare import TinyShakespeareDataset
from src.data.longrange_copy import LongRangeCopyDataset
from src.data.induction import InductionDataset
from src.eval.metrics import evaluate, EvalResult
from src.eval.memory_throughput import ThroughputTracker
from src.models.mingru import MinGRULM, MinGRUConfig
from src.training.checkpointing import save_checkpoint, load_checkpoint
from src.training.logging_utils import RunLogger
from src.training.schedules import WarmupCosineSchedule
from src.utils.param_count import assert_param_budget
from src.utils.seed import set_seed, get_rng_states, set_rng_states


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_dataset(cfg: TrainConfig) -> tuple[Any, Any, int, int, bool, int | None]:
    """Build train + val dataset handles for the given task.

    Returns:
        train_ds, val_ds, vocab_size, block_size, has_mask, mask_positions
            - train_ds, val_ds: dataset objects with .get_batch / .iter_batches
              (or .iter_val_batches for TinyShakespeare).
            - vocab_size: int.
            - block_size: int. For synthetic tasks this is derived from the
              data file's sequence length minus one (since y is x shifted left).
            - has_mask: True for synthetic tasks, False for TinyShakespeare.
            - mask_positions: 5 for synthetic tasks, None for TS.

    For TinyShakespeare the train and val "datasets" are the same object
    (TinyShakespeareDataset exposes both splits); we return it twice for
    interface uniformity with the synthetic tasks.
    """
    data_root = Path(cfg.data_root)

    if cfg.task == "tinyshakespeare":
        assert cfg.block_size is not None, "block_size required for TinyShakespeare"
        ds = TinyShakespeareDataset(
            block_size=cfg.block_size, data_dir=data_root / "tinyshakespeare"
        )
        return ds, ds, ds.vocab_size, cfg.block_size, False, None

    if cfg.task == "longrange_copy":
        assert cfg.length_tier is not None, "length_tier required for longrange_copy"
        train_ds = LongRangeCopyDataset(
            data_root / "longrange_copy", split="train", length=cfg.length_tier
        )
        val_ds = LongRangeCopyDataset(
            data_root / "longrange_copy", split="val", length=cfg.length_tier
        )
        # y has length seq_len - 1 (we drop the first input position for targets).
        block_size = train_ds.seq_len - 1
        return train_ds, val_ds, train_ds.vocab_size, block_size, True, 5

    if cfg.task == "induction":
        assert cfg.length_tier is not None, "length_tier required for induction"
        train_ds = InductionDataset(
            data_root / "induction", split="train", length=cfg.length_tier
        )
        val_ds = InductionDataset(
            data_root / "induction", split="val", length=cfg.length_tier
        )
        block_size = train_ds.seq_len - 1
        return train_ds, val_ds, train_ds.vocab_size, block_size, True, 1

    raise ValueError(f"unknown task: {cfg.task}")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _get_train_batch(
    train_ds: Any, cfg: TrainConfig, has_mask: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Draw a training batch, returning (x, y, mask_or_None)."""
    if has_mask:
        return train_ds.get_batch(cfg.batch_size, device=device)
    x, y = train_ds.get_batch("train", cfg.batch_size, device=device)
    return x, y, None


def _val_batches(val_ds: Any, cfg: TrainConfig, has_mask: bool, device: torch.device):
    """Iterator over val batches.

    For TinyShakespeare, uses iter_val_batches (non-overlapping windows over
    the val split). For synthetic tasks, iter_batches walks the entire
    pre-generated val set.
    """
    if cfg.task == "tinyshakespeare":
        gen = val_ds.iter_val_batches(cfg.eval_batch_size, device=device)
    else:
        gen = val_ds.iter_batches(cfg.eval_batch_size, device=device)

    if cfg.eval_max_batches is None:
        yield from gen
    else:
        for i, batch in enumerate(gen):
            if i >= cfg.eval_max_batches:
                break
            yield batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, run_dir: Path | str, *, resume: bool = False) -> dict:
    """Run training for one experiment.

    Args:
        cfg: TrainConfig.
        run_dir: directory to write checkpoints, logs, and meta to.
        resume: if True, look for `last.pt` in run_dir/checkpoints and resume
            from it. If no checkpoint exists, falls through to a fresh start.

    Returns:
        dict with final summary: best_val_loss, best_val_ppl, total_steps,
        wall_clock_seconds, n_parameters. Caller can dump this however it
        likes.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir)

    # 1) Seed early. RNG-affected operations (data loading, model init) come
    # after this point.
    set_seed(cfg.seed)

    # 2) Device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.console_print(f"[train] device = {device}")

    # 3) Datasets.
    train_ds, val_ds, vocab_size, block_size, has_mask, mask_positions = _build_dataset(cfg)
    logger.console_print(
        f"[train] task={cfg.task} vocab_size={vocab_size} block_size={block_size} "
        f"has_mask={has_mask}"
    )

    # 4) Model. Patch the vocab size into the config now that we know it.
    model_cfg_dict = dict(cfg.model_config)
    model_cfg_dict["vocab_size"] = vocab_size
    model_cfg_dict["dropout"] = cfg.dropout
    model = MinGRULM(MinGRUConfig(**model_cfg_dict)).to(device)
    n_params = assert_param_budget(
        model,
        target=cfg.param_budget_target,
        tolerance=cfg.param_budget_tolerance,
        name="minGRU",
    )
    logger.console_print(f"[train] minGRU params: {n_params:,}")

    # 5) Optimizer + scheduler.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    schedule = WarmupCosineSchedule(
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.lr_decay_steps or cfg.total_steps,
        min_lr_ratio=cfg.min_lr / cfg.peak_lr,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=schedule)

    # 6) Optional resume.
    start_step = 0
    best_val_loss = math.inf
    if resume:
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            state = load_checkpoint(last_ckpt, model, optimizer, scheduler, map_location=device)
            start_step = state["step"]
            best_val_loss = state["best_val_loss"]
            set_rng_states(state["rng_states"])
            logger.note_resume(from_step=start_step)
            logger.console_print(
                f"[train] resumed from step {start_step} "
                f"(best val loss so far: {best_val_loss:.4f})"
            )
        else:
            logger.console_print(
                f"[train] --resume requested but no checkpoint at {last_ckpt}; "
                "starting fresh"
            )

    # 7) Write metadata once at the start (or after resume note).
    if start_step == 0:
        logger.write_meta(
            config_repr=cfg.pretty_repr(),
            n_params=n_params,
            extra={"device": str(device), "block_size": block_size},
        )

    # 8) Mixed precision context.
    use_bf16 = cfg.mixed_precision == "bf16" and device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )

    # 9) Train loop.
    tracker = ThroughputTracker(device=device)
    tracker.start()

    train_loss_ema = None
    ema_alpha = 0.05  # smoothing for the console-printed running train loss
    wall_start = time.perf_counter()

    for step in range(start_step, cfg.total_steps):
        model.train()
        x, y, mask = _get_train_batch(train_ds, cfg, has_mask, device)
        # mask is unused at train time — we train on all positions per Feng et al.

        with autocast_ctx:
            logits = model(x)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Grad clip applied in fp32; loss.backward already produces fp32 grads
        # under bf16 autocast (autocast scopes the forward, not the grads).
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        tokens_this_step = B * T
        tracker.tick(tokens_this_step)

        # EMA for human-readable console logging.
        loss_item = loss.item()
        train_loss_ema = (
            loss_item if train_loss_ema is None
            else (1 - ema_alpha) * train_loss_ema + ema_alpha * loss_item
        )

        # Lightweight console heartbeat every 50 steps.
        if (step + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.console_print(
                f"[step {step + 1:5d}] loss={loss_item:.4f} ema={train_loss_ema:.4f} "
                f"lr={current_lr:.2e}"
            )

        # Periodic eval + checkpoint.
        if (step + 1) % cfg.eval_interval == 0 or (step + 1) == cfg.total_steps:
            tps = tracker.summarize()
            tracker.start()  # reset for next window

            eval_result = evaluate(
                model,
                _val_batches(val_ds, cfg, has_mask, device),
                device=device,
                has_mask=has_mask,
                mask_positions_per_sequence=mask_positions,
            )

            wall_clock_s = time.perf_counter() - wall_start
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "step": step + 1,
                "wall_clock_s": round(wall_clock_s, 2),
                "lr": current_lr,
                "train_loss_ema": train_loss_ema,
                "tokens_per_s": round(tps.tokens_per_s, 1),
                "peak_mem_mb": round(tps.peak_mem_mb, 1),
                **eval_result.as_csv_row(),
            }
            logger.log_eval(row)
            logger.console_print(
                f"[eval @ step {step + 1}] val_ppl={eval_result.val_ppl:.4f} "
                f"masked_ppl={eval_result.masked_ppl:.4f} "
                f"masked_acc={eval_result.masked_acc:.4f} "
                f"tps={tps.tokens_per_s:.0f} peak_mem={tps.peak_mem_mb:.0f}MB"
            )

            # Always save last.pt; conditionally save best.pt.
            rng = get_rng_states()
            save_checkpoint(
                ckpt_dir / "last.pt",
                model, optimizer, scheduler,
                step=step + 1,
                best_val_loss=best_val_loss,
                rng_states=rng,
                config=cfg.to_dict(),
            )
            if eval_result.val_loss < best_val_loss:
                best_val_loss = eval_result.val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model, optimizer, scheduler,
                    step=step + 1,
                    best_val_loss=best_val_loss,
                    rng_states=rng,
                    config=cfg.to_dict(),
                )
                logger.console_print(
                    f"[eval @ step {step + 1}] new best val loss: {best_val_loss:.4f}"
                )

    # 10) Final summary.
    total_wall = time.perf_counter() - wall_start
    summary = {
        "best_val_loss": best_val_loss,
        "best_val_ppl": float(math.exp(best_val_loss)) if math.isfinite(best_val_loss) else float("nan"),
        "total_steps": cfg.total_steps,
        "wall_clock_seconds": round(total_wall, 2),
        "n_parameters": n_params,
    }
    logger.console_print(f"[train] DONE {summary}")
    logger.close()
    return summary
