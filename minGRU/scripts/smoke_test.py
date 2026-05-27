"""Smoke test (proposal §4.4 — Day 3-4 gate).

Runs minGRU on each task at the medium length tier for a small number of
steps, so the three of you can quickly compare reports across paradigms and
decide whether the synthetic tasks discriminate.

Gate criterion (per proposal §4.4):
    "If three paradigms reach within 5% relative validation PPL of each
    other on a synthetic task, that task is redesigned or dropped before
    full-length runs."

The 5% comparison is across-paradigm and CANNOT be checked from a single
owner's runs — Mayukh and Adam will run their equivalents, and the three of
you compare the resulting PPLs at the team standup. This script's job is
just to produce minGRU's three numbers.

Outputs:
  * Each smoke run gets its own directory under `runs/smoke/<exp>__seed42__<ts>/`
    so it doesn't pollute the primary `runs/`.
  * A summary table is printed to stdout AND written to
    `runs/smoke/smoke_summary.txt`, ready to paste into Slack.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --steps 1500
    python scripts/smoke_test.py --only mingru_longcopy_medium
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs import experiments  # noqa: E402
from src.training.trainer import train  # noqa: E402


# The three cells the smoke test runs by default, per proposal §4.4: the
# medium length of each task.
_SMOKE_CELLS = [
    "mingru_tinyshakespeare_1024",   # "medium" length for TS sweep
    "mingru_longcopy_medium",
    "mingru_induction_medium",
]


def _smoke_run_dir(exp_name: str, seed: int, runs_root: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return runs_root / f"{exp_name}__seed{seed}__{ts}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="number of training steps per smoke run (proposal: 1-2K)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="eval every N steps; should divide --steps cleanly",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="restrict to specific experiment(s); pass multiple times",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/smoke"),
        help="directory for smoke run outputs (default: runs/smoke)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cells = args.only if args.only else _SMOKE_CELLS
    args.runs_root.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] running {len(cells)} cells at {args.steps} steps each")
    print(f"[smoke] cells: {cells}")
    print(f"[smoke] output: {args.runs_root}")
    print()

    results: list[tuple[str, dict]] = []
    for cell in cells:
        cfg = experiments.get(cell)
        # Shorten the run: smoke-step count and matching eval interval. Keep
        # warmup proportional so the lr schedule still makes sense — the
        # default 200-step warmup over 1500 steps is fine (13%).
        cfg = replace(cfg, seed=args.seed, total_steps=args.steps,
                      eval_interval=args.eval_interval)
        run_dir = _smoke_run_dir(cell, args.seed, args.runs_root)
        print(f"[smoke] --- {cell} ---")
        summary = train(cfg, run_dir=run_dir, resume=False)
        results.append((cell, summary))
        print()

    # Print and save a compact summary table — what you'd paste into Slack.
    lines = [
        "Smoke test summary (minGRU, " + str(args.steps) + " steps):",
        f"  date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"  {'cell':40s}  {'best_val_ppl':>12s}  {'wall_s':>8s}",
    ]
    for cell, summary in results:
        lines.append(
            f"  {cell:40s}  {summary['best_val_ppl']:>12.4f}  "
            f"{summary['wall_clock_seconds']:>8.1f}"
        )
    lines.append("")
    lines.append("Compare these to Mayukh's and Adam's smoke runs. Per proposal §4.4:")
    lines.append("  if all three paradigms reach within 5% relative val PPL on a")
    lines.append("  synthetic task, redesign or drop that task before the full runs.")

    out_text = "\n".join(lines)
    print(out_text)

    summary_path = args.runs_root / "smoke_summary.txt"
    summary_path.write_text(out_text + "\n", encoding="utf-8")
    print(f"\n[smoke] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
