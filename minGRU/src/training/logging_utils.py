"""Logging: CSV metrics + console.log mirror + meta.txt.

Three log files per run, all inside the run's directory:

  * metrics.csv — one row per eval event. Columns are determined at first
    write from the keys of the dict passed to `log_eval`. Append mode, so
    --resume doesn't truncate history.

  * console.log — a plain text mirror of stdout. The trainer calls
    `console_print` instead of print(); console_print echoes to stdout AND
    appends a line to console.log. On --resume, new lines are appended.

  * meta.txt — written once at run start, captures everything reproducibility
    needs: git SHA, hostname, start time, full config repr, parameter count.
    On --resume the existing meta.txt is preserved; an APPENDED line records
    the resume event.

A small `RunLogger` class bundles these three concerns. The trainer
instantiates one and calls its methods rather than juggling three files.
"""
from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _git_sha(repo_dir: Path | None = None) -> str:
    """Return the current git commit SHA, or '(no git)' if unavailable.

    Best-effort: runs `git rev-parse HEAD`. If git isn't installed, or this
    isn't a git repo, or the call fails for any reason, returns a sentinel
    string. The trainer should still proceed — it's metadata, not a runtime
    requirement.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(no git)"


def _git_dirty(repo_dir: Path | None = None) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class RunLogger:
    """Bundle of three log streams: metrics.csv, console.log, meta.txt.

    Methods:
        write_meta(config_repr, n_params): called once at run start.
        log_eval(row): one row per evaluation event, written to metrics.csv.
        console_print(*args): print to stdout AND append to console.log.
        note_resume(from_step): append a one-line note to meta.txt.
    """

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.console_path = self.run_dir / "console.log"
        self.meta_path = self.run_dir / "meta.txt"

        # csv.DictWriter is created lazily on first log_eval call, since we
        # don't know the column set until we see the first row. Once written,
        # the column order is fixed for the lifetime of the file.
        self._csv_fieldnames: list[str] | None = None

    # -- console -----------------------------------------------------------

    def console_print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        """Print to stdout and append to console.log.

        Flushes after every call so a crash leaves a complete log on disk.
        """
        msg = sep.join(str(a) for a in args)
        print(msg, end=end, flush=True)
        with open(self.console_path, "a", encoding="utf-8") as f:
            f.write(msg + end)

    # -- metrics -----------------------------------------------------------

    def log_eval(self, row: dict[str, Any]) -> None:
        """Append a row to metrics.csv.

        On the first call, the header is written using `row.keys()` as the
        column order. Subsequent calls MUST supply the same keys; an
        AssertionError is raised on mismatch (silent dropped columns are a
        nasty source of missing data in plots).
        """
        if self._csv_fieldnames is None:
            # First write of this run. If the file already exists (resume),
            # read the existing header instead of overwriting it.
            if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    self._csv_fieldnames = next(reader)
                new_file = False
            else:
                self._csv_fieldnames = list(row.keys())
                new_file = True

            f = open(self.metrics_path, "a", encoding="utf-8", newline="")
            self._csv_file = f
            self._csv_writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            if new_file:
                self._csv_writer.writeheader()

        # Subsequent calls: enforce column consistency.
        missing = set(self._csv_fieldnames) - set(row.keys())
        extra = set(row.keys()) - set(self._csv_fieldnames)
        if missing or extra:
            raise AssertionError(
                f"metrics.csv column mismatch. missing={missing}, extra={extra}. "
                "All eval rows in a run must have the same keys; if you need "
                "new columns mid-run, add them with NaN/sentinel defaults from "
                "the first call onward."
            )
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Flush and close the CSV file. Safe to call more than once."""
        f = getattr(self, "_csv_file", None)
        if f is not None and not f.closed:
            f.flush()
            f.close()

    # -- meta --------------------------------------------------------------

    def write_meta(
        self,
        config_repr: str,
        n_params: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write the once-per-run metadata block.

        Captures: git SHA + dirty flag, hostname, PID, Python version, start
        timestamp (ISO), config repr, parameter count, anything in `extra`.

        Also writes a second file alongside: `config.txt`, a clean copy of
        config_repr without the surrounding metadata, for tools that just
        want the config.
        """
        lines = [
            "# Run metadata",
            f"start_time: {datetime.now().isoformat(timespec='seconds')}",
            f"hostname: {socket.gethostname()}",
            f"pid: {os.getpid()}",
            f"python: {sys.version.split()[0]}",
            f"git_sha: {_git_sha()}",
            f"git_dirty: {_git_dirty()}",
            f"n_parameters: {n_params}",
        ]
        if extra:
            for k, v in extra.items():
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("# Config")
        lines.append(config_repr)

        self.meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.run_dir / "config.txt").write_text(config_repr + "\n", encoding="utf-8")

    def note_resume(self, from_step: int) -> None:
        """Append a one-line note to meta.txt recording a resume event."""
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n# Resume at {datetime.now().isoformat(timespec='seconds')}: "
                f"continuing from step {from_step}\n"
            )
