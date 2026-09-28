"""Enumerate, measure and remove previously trained models.

Training runs accumulate fast. Each checkpoint here is ~134 MB -- 45 MB of
weights and 89 MB of AdamW optimizer state -- and a run keeps ``best.pt``,
``last.pt`` and up to three per-epoch copies, so one run costs ~640 MB. Six runs
had reached 3.6 GB, with another 2.6 GB of dataset snapshots behind them.

Three levels of cleanup, because they trade away different things:

* **prune** -- delete the per-epoch copies, keep ``best.pt`` and ``last.pt``.
  Loses nothing you can still use: the epoch copies exist only so a run that
  ends badly can be rolled back to an earlier peak.
* **archive** -- additionally strip the optimizer state. The model still trains
  nothing further but evaluates, exports and predicts exactly as before, at a
  third of the size. Irreversible for *resuming* that run.
* **delete** -- remove the run entirely, optionally with the dataset snapshot it
  was trained on.

Deletions are confined to the configured outputs and datasets directories; a run
whose path escapes them is refused rather than trusted.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from callisto_trainer.core.logging_utils import get_logger

LOGGER = get_logger(__name__)

# outputs/<task>_<run_id>/  ->  datasets/<kind>/<run_id>/
TASK_TO_SNAPSHOT_KIND = {"unified": "unified", "type": "types", "binary": "binary"}


@dataclass
class TrainedRun:
    """One training run on disk, with everything needed to decide its fate."""

    task: str
    run_id: str
    directory: Path
    size_bytes: int = 0
    checkpoint_count: int = 0
    epoch_copy_count: int = 0
    epoch_copy_bytes: int = 0
    optimizer_state_bytes: int = 0
    has_best: bool = False
    best_score: float | None = None
    monitor: str = ""
    epochs_trained: int = 0
    trained_at: str = ""
    snapshot_dir: Path | None = None
    snapshot_bytes: int = 0
    snapshot_samples: int | None = None

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def total_bytes(self) -> int:
        return self.size_bytes + self.snapshot_bytes

    @property
    def reclaimable_bytes(self) -> int:
        """What pruning and archiving would free, without deleting the model."""
        return self.epoch_copy_bytes + self.optimizer_state_bytes

    def describe_score(self) -> str:
        if self.best_score is None:
            return "-"
        return f"{self.best_score:.4f} {self.monitor or ''}".strip()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read_history(checkpoint_dir: Path) -> tuple[float | None, str, int]:
    """Best validation score, the metric it was chosen by, and epochs trained.

    Read from ``training_history.json`` rather than the checkpoint: the history is
    a few kilobytes, the checkpoint is 134 MB.
    """
    path = checkpoint_dir / "training_history.json"
    if not path.exists():
        return None, "", 0
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "", 0
    if not history:
        return None, "", 0

    monitor = "macro_f1" if "macro_f1" in (history[-1].get("val") or {}) else "pr_auc"
    scores = [
        float(entry.get("val", {}).get(monitor))
        for entry in history
        if isinstance(entry.get("val", {}).get(monitor), (int, float))
    ]
    return (max(scores) if scores else None), monitor, len(history)


def _epoch_copies(checkpoint_dir: Path) -> list[Path]:
    """Per-epoch snapshots, excluding the best.pt/last.pt aliases."""
    return sorted(checkpoint_dir.glob("*_epoch_*.pt"))


def _optimizer_bytes(checkpoint_dir: Path) -> int:
    """Approximate optimizer state held in the surviving aliases.

    Estimated rather than measured: loading two 134 MB checkpoints to weigh a
    dict would make simply listing the runs take seconds. AdamW keeps two moments
    per parameter, so the optimizer state is very close to twice the weights,
    i.e. two thirds of the file.
    """
    total = 0
    for name in ("best.pt", "last.pt"):
        path = checkpoint_dir / name
        if path.exists():
            total += int(path.stat().st_size * 2 / 3)
    return total


def list_runs(outputs_dir: str | Path, datasets_dir: str | Path | None = None) -> list[TrainedRun]:
    """Every training run under ``outputs_dir``, newest first."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return []

    runs: list[TrainedRun] = []
    for directory in sorted(outputs_dir.iterdir(), reverse=True):
        if not directory.is_dir() or "_" not in directory.name:
            continue
        task, _, run_id = directory.name.partition("_")
        if task not in TASK_TO_SNAPSHOT_KIND:
            continue

        checkpoint_dir = directory / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        epoch_copies = _epoch_copies(checkpoint_dir) if checkpoint_dir.exists() else []
        best_score, monitor, epochs = _read_history(checkpoint_dir)

        run = TrainedRun(
            task=task,
            run_id=run_id,
            directory=directory,
            size_bytes=_directory_size(directory),
            checkpoint_count=len(checkpoints),
            epoch_copy_count=len(epoch_copies),
            epoch_copy_bytes=sum(path.stat().st_size for path in epoch_copies),
            optimizer_state_bytes=_optimizer_bytes(checkpoint_dir) if checkpoint_dir.exists() else 0,
            has_best=(checkpoint_dir / "best.pt").exists(),
            best_score=best_score,
            monitor=monitor,
            epochs_trained=epochs,
            trained_at=_trained_at(directory, run_id),
        )

        if datasets_dir is not None:
            kind = TASK_TO_SNAPSHOT_KIND[task]
            snapshot = Path(datasets_dir) / kind / run_id
            if snapshot.exists():
                run.snapshot_dir = snapshot
                run.snapshot_bytes = _directory_size(snapshot)
                run.snapshot_samples = _snapshot_samples(snapshot)
        runs.append(run)
    return runs


def _trained_at(directory: Path, run_id: str) -> str:
    """Human timestamp, from the run id when it parses and mtime otherwise."""
    try:
        return datetime.strptime(run_id, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        try:
            return datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return ""


def _snapshot_samples(snapshot: Path) -> int | None:
    path = snapshot / "snapshot.json"
    if not path.exists():
        return None
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("samples", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _assert_inside(path: Path, *roots: Path | None) -> None:
    """Refuse to touch anything outside the configured directories."""
    resolved = path.resolve()
    for root in roots:
        if root is None:
            continue
        try:
            resolved.relative_to(Path(root).resolve())
            return
        except ValueError:
            continue
    raise ValueError(f"Refusing to delete {path}: outside the project's managed directories.")


def delete_run(
    run: TrainedRun,
    outputs_dir: str | Path,
    datasets_dir: str | Path | None = None,
    include_snapshot: bool = False,
) -> int:
    """Delete a run, optionally with its dataset snapshot. Returns bytes freed."""
    _assert_inside(run.directory, Path(outputs_dir))
    freed = run.size_bytes
    shutil.rmtree(run.directory, ignore_errors=False)

    if include_snapshot and run.snapshot_dir is not None:
        _assert_inside(run.snapshot_dir, Path(datasets_dir) if datasets_dir else None)
        freed += run.snapshot_bytes
        shutil.rmtree(run.snapshot_dir, ignore_errors=False)

    LOGGER.info("Deleted %s (%.0f MB freed)", run.name, freed / 1e6)
    return freed


def prune_epoch_copies(run: TrainedRun, outputs_dir: str | Path) -> int:
    """Delete per-epoch checkpoint copies, keeping best.pt and last.pt.

    Safe: the epoch copies exist only so a run can be rolled back to an earlier
    peak. The selected best model and the resumable last state both survive.
    """
    checkpoint_dir = run.directory / "checkpoints"
    _assert_inside(checkpoint_dir, Path(outputs_dir))

    freed = 0
    for path in _epoch_copies(checkpoint_dir):
        try:
            freed += path.stat().st_size
            path.unlink()
        except OSError as exc:
            LOGGER.warning("Could not remove %s: %s", path, exc)
    LOGGER.info("Pruned %s (%.0f MB freed)", run.name, freed / 1e6)
    return freed


def archive_run(run: TrainedRun, outputs_dir: str | Path) -> int:
    """Strip optimizer state from the surviving checkpoints.

    Cuts each checkpoint to roughly a third. The model still evaluates, exports
    and predicts identically -- only *resuming training* from it is lost, which
    is why this is offered separately from pruning rather than folded into it.
    """
    import torch

    checkpoint_dir = run.directory / "checkpoints"
    _assert_inside(checkpoint_dir, Path(outputs_dir))

    freed = 0
    for name in ("best.pt", "last.pt"):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        try:
            before = path.stat().st_size
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            # An optimizer that never stepped still has a truthy state_dict --
            # param_groups with an empty "state". Rewriting that frees nothing and
            # would grow the file by the flag we add, so check the moments.
            if not (checkpoint.get("optimizer_state") or {}).get("state"):
                continue
            checkpoint["optimizer_state"] = {}
            checkpoint["archived"] = True
            # Write beside the original and swap, so an interrupted save cannot
            # leave a truncated checkpoint where a working one used to be.
            temporary = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
            freed += before - path.stat().st_size
        except Exception as exc:
            LOGGER.warning("Could not archive %s: %s", path, exc)
    LOGGER.info("Archived %s (%.0f MB freed)", run.name, freed / 1e6)
    return freed


def format_bytes(value: int) -> str:
    """Human size, in the units this project actually produces."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit in ("B", "KB") else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} GB"


def total_bytes(runs: Iterable[TrainedRun], include_snapshots: bool = True) -> int:
    return sum(run.total_bytes if include_snapshots else run.size_bytes for run in runs)
