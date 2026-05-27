"""Memory and throughput tracking.

Reports peak GPU memory and tokens-per-second for Plot 3 of the report.
Both are measured around a known window of training steps so the numbers
reflect steady-state behaviour, not warmup or first-step compile overhead.

Usage in the trainer:

    tracker = ThroughputTracker()
    tracker.start()
    for step in range(...):
        ...
        tracker.tick(tokens_processed)
    metrics = tracker.summarize()  # returns dict with peak_mem_mb, tokens_per_s
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class ThroughputSnapshot:
    peak_mem_mb: float       # peak GPU memory since last reset, in MB
    tokens_per_s: float      # tokens processed / wall-clock seconds elapsed
    elapsed_s: float
    total_tokens: int


class ThroughputTracker:
    """Track tokens-per-second and peak GPU memory across training steps."""

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self._start_time: float | None = None
        self._total_tokens: int = 0

    def start(self) -> None:
        """Reset counters and peak memory stats. Call after warmup steps so
        compile / kernel-selection overhead doesn't pollute the throughput
        estimate."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self._start_time = time.perf_counter()
        self._total_tokens = 0

    def tick(self, tokens_processed: int) -> None:
        """Record `tokens_processed` more tokens have gone through the model."""
        if self._start_time is None:
            raise RuntimeError("ThroughputTracker.start() must be called first")
        self._total_tokens += tokens_processed

    def summarize(self) -> ThroughputSnapshot:
        """Return a snapshot of throughput and peak memory since `start()`.

        Synchronizes the CUDA stream before reading wall-clock — otherwise
        async kernel launches make tokens/s look optimistic.
        """
        if self._start_time is None:
            raise RuntimeError("ThroughputTracker.start() must be called first")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._start_time

        if self.device.type == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_mem_mb = peak_bytes / (1024 ** 2)
        else:
            peak_mem_mb = float("nan")

        tps = self._total_tokens / max(elapsed, 1e-9)
        return ThroughputSnapshot(
            peak_mem_mb=peak_mem_mb,
            tokens_per_s=tps,
            elapsed_s=elapsed,
            total_tokens=self._total_tokens,
        )
