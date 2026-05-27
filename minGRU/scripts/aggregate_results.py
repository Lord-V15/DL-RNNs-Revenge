"""Aggregate completed runs into a single summary.csv.

Walks `runs/`, finds every directory matching `<exp_name>__seed<N>__<timestamp>/`,
reads each run's `metrics.csv` and `meta.txt`, and produces a top-level
`summary.csv` with one row per run.

Selection: for each run, the row reported is the eval with the LOWEST val_loss
(the "best" eval, matching what `best.pt` represents). This is what the report's
Table 1 wants — the best validation PPL per cell.

Multi-seed handling: when more than one run shares the same `exp_name` (i.e.
the same cell of the matrix run at multiple seeds), all seeds appear as
separate rows in summary.csv. A second helper (`aggregate_per_cell`) groups by
exp_name and reports mean ± std for the columns that need it (val_ppl,
masked_ppl, masked_acc) — Table 1 of the proposal expects single point
estimates for most cells, with the TinyShakespeare-256 cell having 3 seeds.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --runs-root path/to/runs
    python scripts/aggregate_results.py --out summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Columns lifted from each run's best eval. Order is preserved in summary.csv.
_PER_RUN_COLUMNS = [
    "run_name",
    "exp_name",
    "seed",
    "task",
    "block_size",
    "length_tier",
    "best_step",
    "best_val_loss",
    "best_val_ppl",
    "best_masked_loss",
    "best_masked_ppl",
    "best_masked_acc",
    "best_masked_loss_pos0",
    "best_masked_loss_pos1",
    "best_masked_loss_pos2",
    "best_masked_loss_pos3",
    "best_masked_loss_pos4",
    "final_tokens_per_s",
    "final_peak_mem_mb",
    "n_parameters",
    "wall_clock_s",
    "n_eval_events",
    "git_sha",
]


def _parse_meta(meta_path: Path) -> dict[str, str]:
    """Parse meta.txt's `key: value` lines into a dict.

    The file has a small header block of key/value pairs, then a `# Config`
    section with the dataclass repr (we don't parse the repr — the typed
    fields we need are duplicated in the per-run config in the checkpoint,
    and the run name encodes seed and exp_name).
    """
    out: dict[str, str] = {}
    if not meta_path.exists():
        return out
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_run_name(name: str) -> tuple[str, int]:
    """Extract (exp_name, seed) from a run directory name.

    Run names look like `mingru_tinyshakespeare_256__seed42__20260520-120000`.
    Returns ('mingru_tinyshakespeare_256', 42). If the format doesn't match,
    raises ValueError — silent fallback would mask real bugs in run naming.
    """
    parts = name.split("__")
    if len(parts) < 2:
        raise ValueError(f"unexpected run name format: {name!r}")
    exp_name = parts[0]
    seed_part = parts[1]
    if not seed_part.startswith("seed"):
        raise ValueError(f"expected 'seedN' in run name; got {seed_part!r} in {name!r}")
    seed = int(seed_part[len("seed"):])
    return exp_name, seed


def _read_metrics_csv(path: Path) -> list[dict[str, str]]:
    """Read a metrics.csv into a list of row dicts. Empty list if missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(s: str) -> float:
    """Parse a float, mapping empty strings and 'nan' to NaN."""
    if s is None or s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _select_best(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the row with the lowest val_loss, or None if no rows."""
    if not rows:
        return None
    best = None
    best_loss = math.inf
    for r in rows:
        try:
            vl = _as_float(r.get("val_loss", ""))
        except ValueError:
            continue
        if math.isfinite(vl) and vl < best_loss:
            best_loss = vl
            best = r
    return best


def _last_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the last row, for fields whose end-of-training value we want
    (e.g. tokens_per_s, peak_mem_mb — Plot 3 wants the steady-state value)."""
    return rows[-1] if rows else None


def _infer_task_and_length(exp_name: str) -> tuple[str, str | None, int | None]:
    """Guess (task, length_tier, block_size) from the exp_name.

    Mirrors configs/experiments.py naming. Falls back to ('unknown', None, None)
    rather than raising — the row still gets written, the inference fields are
    just blank for inspection.
    """
    if "tinyshakespeare" in exp_name:
        # mingru_tinyshakespeare_<block_size>
        try:
            bs = int(exp_name.rsplit("_", 1)[-1])
        except ValueError:
            bs = None
        return "tinyshakespeare", None, bs
    if "longcopy" in exp_name:
        tier = exp_name.rsplit("_", 1)[-1]
        return "longrange_copy", tier, None
    if "induction" in exp_name:
        tier = exp_name.rsplit("_", 1)[-1]
        return "induction", tier, None
    return "unknown", None, None


def aggregate(runs_root: Path) -> list[dict[str, object]]:
    """Build the per-run summary rows.

    Args:
        runs_root: directory holding `<exp>__seed<N>__<timestamp>/` subdirs.

    Returns:
        list of dicts (one per run). Runs without a usable metrics.csv are
        skipped with a printed warning rather than failing the whole sweep.
    """
    rows: list[dict[str, object]] = []
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    for run_dir in run_dirs:
        try:
            exp_name, seed = _parse_run_name(run_dir.name)
        except ValueError as e:
            print(f"[aggregate] skipping {run_dir.name}: {e}")
            continue

        metrics = _read_metrics_csv(run_dir / "metrics.csv")
        if not metrics:
            print(f"[aggregate] skipping {run_dir.name}: empty/missing metrics.csv")
            continue

        best = _select_best(metrics)
        last = _last_row(metrics)
        if best is None or last is None:
            print(f"[aggregate] skipping {run_dir.name}: no usable eval rows")
            continue

        meta = _parse_meta(run_dir / "meta.txt")
        task, tier, block_size = _infer_task_and_length(exp_name)

        row = {
            "run_name": run_dir.name,
            "exp_name": exp_name,
            "seed": seed,
            "task": task,
            "block_size": block_size if block_size is not None else "",
            "length_tier": tier if tier is not None else "",
            "best_step": int(_as_float(best.get("step", "0"))),
            "best_val_loss": _as_float(best.get("val_loss", "")),
            "best_val_ppl": _as_float(best.get("val_ppl", "")),
            "best_masked_loss": _as_float(best.get("masked_loss", "")),
            "best_masked_ppl": _as_float(best.get("masked_ppl", "")),
            "best_masked_acc": _as_float(best.get("masked_acc", "")),
            "best_masked_loss_pos0": _as_float(best.get("masked_loss_pos0", "")),
            "best_masked_loss_pos1": _as_float(best.get("masked_loss_pos1", "")),
            "best_masked_loss_pos2": _as_float(best.get("masked_loss_pos2", "")),
            "best_masked_loss_pos3": _as_float(best.get("masked_loss_pos3", "")),
            "best_masked_loss_pos4": _as_float(best.get("masked_loss_pos4", "")),
            "final_tokens_per_s": _as_float(last.get("tokens_per_s", "")),
            "final_peak_mem_mb": _as_float(last.get("peak_mem_mb", "")),
            "n_parameters": int(_as_float(meta.get("n_parameters", "0"))) if meta.get("n_parameters") else "",
            "wall_clock_s": _as_float(last.get("wall_clock_s", "")),
            "n_eval_events": len(metrics),
            "git_sha": meta.get("git_sha", ""),
        }
        rows.append(row)
    return rows


def aggregate_per_cell(per_run: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group per-run rows by exp_name and report mean ± std.

    For the TinyShakespeare-256 cell with 3 seeds this gives the
    "4.63 ± 0.03" style result. Cells with a single seed get the same value
    in mean and 0.0 (or NaN, see note below) in std.

    We report NaN std (not 0.0) when there's only one seed — a 0.0 std would
    misleadingly suggest measured variance. NaN makes "single seed" visible.
    """
    by_exp: dict[str, list[dict[str, object]]] = {}
    for row in per_run:
        by_exp.setdefault(str(row["exp_name"]), []).append(row)

    out: list[dict[str, object]] = []
    metric_cols = ["best_val_loss", "best_val_ppl", "best_masked_loss",
                   "best_masked_ppl", "best_masked_acc"]
    for exp_name, runs in by_exp.items():
        agg: dict[str, object] = {
            "exp_name": exp_name,
            "task": runs[0]["task"],
            "length_tier": runs[0]["length_tier"],
            "block_size": runs[0]["block_size"],
            "n_seeds": len(runs),
            "seeds": ",".join(str(r["seed"]) for r in runs),
        }
        for col in metric_cols:
            vals = [v for v in (float(r[col]) for r in runs)
                    if math.isfinite(v)]
            if not vals:
                agg[f"{col}_mean"] = float("nan")
                agg[f"{col}_std"] = float("nan")
            else:
                agg[f"{col}_mean"] = sum(vals) / len(vals)
                agg[f"{col}_std"] = (
                    statistics.stdev(vals) if len(vals) > 1 else float("nan")
                )
        out.append(agg)
    return out


def write_csv(rows: list[dict[str, object]], path: Path, columns: list[str] | None = None) -> None:
    """Write `rows` to `path` as CSV. If `columns` is given, restrict and
    order columns to that list; otherwise use the keys of the first row."""
    if not rows:
        print(f"[aggregate] no rows to write to {path}")
        return
    if columns is None:
        columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(rows)} rows to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("summary.csv"),
                        help="output file for the per-run summary")
    parser.add_argument("--out-per-cell", type=Path, default=Path("summary_per_cell.csv"),
                        help="output file for the mean±std per-cell summary")
    args = parser.parse_args(argv)

    if not args.runs_root.exists():
        print(f"[aggregate] runs root not found: {args.runs_root}")
        return 1

    per_run = aggregate(args.runs_root)
    write_csv(per_run, args.out, columns=_PER_RUN_COLUMNS)

    per_cell = aggregate_per_cell(per_run)
    write_csv(per_cell, args.out_per_cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
