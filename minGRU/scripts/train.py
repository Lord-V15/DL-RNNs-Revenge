"""Train one experiment.

Usage:
    python scripts/train.py --exp mingru_tinyshakespeare_256
    python scripts/train.py --exp mingru_longcopy_medium --seed 1337
    python scripts/train.py --exp mingru_induction_short --resume

The experiment name must match a key in configs/experiments.py:EXPERIMENTS.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

# Make `configs.*` and `src.*` importable when running this script directly
# (i.e. `python scripts/train.py ...` from the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs import experiments  # noqa: E402
from src.training.trainer import train  # noqa: E402


def _run_dir_for(cfg, runs_root: Path) -> Path:
    """Compute the run directory: runs/<exp_name>__seed<N>__<timestamp>/.

    Timestamp is yyyymmdd-HHMMSS so directories sort chronologically.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{cfg.exp_name}__seed{cfg.seed}__{ts}"
    return runs_root / name


def _existing_run_dir(cfg, runs_root: Path) -> Path | None:
    """When --resume is passed, find the most recent run dir matching this
    exp_name + seed and resume from it. Returns None if no match exists."""
    prefix = f"{cfg.exp_name}__seed{cfg.seed}__"
    candidates = sorted(
        (p for p in runs_root.glob(f"{prefix}*") if p.is_dir()),
        key=lambda p: p.name,
    )
    if not candidates:
        return None
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        required=True,
        help="experiment name (see configs/experiments.py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the seed in the experiment config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from runs/<exp>__seed<N>__<latest>/checkpoints/last.pt",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="override the runs root directory (default: from config)",
    )
    args = parser.parse_args(argv)

    cfg = experiments.get(args.exp)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)

    runs_root = args.runs_root if args.runs_root is not None else Path(cfg.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.resume:
        run_dir = _existing_run_dir(cfg, runs_root)
        if run_dir is None:
            print(
                f"[train] --resume given but no existing run found for "
                f"{cfg.exp_name} seed={cfg.seed}; starting fresh"
            )
            run_dir = _run_dir_for(cfg, runs_root)
            resume = False
        else:
            print(f"[train] resuming run at {run_dir}")
            resume = True
    else:
        run_dir = _run_dir_for(cfg, runs_root)
        resume = False

    summary = train(cfg, run_dir=run_dir, resume=resume)
    print(f"[train] summary: {summary}")
    print(f"[train] run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
