"""Dialog for reviewing and removing previously trained models.

Runs are expensive on disk (~640 MB each) and accumulate quickly, so this offers
three levels rather than only deletion — see
:mod:`callisto_trainer.services.model_registry`. Every destructive action states
exactly what will be removed and how much it frees before doing it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.model_registry import (
    TrainedRun,
    archive_run,
    delete_run,
    format_bytes,
    list_runs,
    prune_epoch_copies,
)
from callisto_trainer.settings import AppSettings

LOGGER = get_logger(__name__)

RunRole = Qt.ItemDataRole.UserRole + 1

COLUMNS = [
    "Model",
    "Trained",
    "Epochs",
    "Best score",
    "Checkpoints",
    "Model size",
    "Dataset snapshot",
]


class ModelManagerDialog(QDialog):
    """List trained runs and free the space they occupy."""

    models_changed = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._runs: list[TrainedRun] = []

        self.setWindowTitle("Trained models")
        self.resize(1080, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "Every training run keeps its checkpoints, reports and figures. A run is "
            "around 640 MB, most of it optimizer state that is only needed to resume "
            "training. The two middle actions free space <b>without</b> losing a usable "
            "model."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_actions)
        layout.addWidget(self.table, 1)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.summary)

        layout.addLayout(self._build_actions())
        self.refresh()

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.prune_button = QPushButton("Free space")
        self.prune_button.setToolTip(
            "Delete the per-epoch checkpoint copies, keeping best.pt and last.pt.\n"
            "Loses nothing usable: those copies exist only to roll a run back to an "
            "earlier peak."
        )
        self.archive_button = QPushButton("Archive")
        self.archive_button.setToolTip(
            "Also strip the optimizer state, cutting each checkpoint to about a third.\n"
            "The model still evaluates, exports and predicts identically — but this run "
            "can no longer be resumed for further training."
        )
        self.delete_button = QPushButton("Delete model")
        self.delete_button.setToolTip("Remove the run entirely: checkpoints, reports and figures.")
        self.delete_all_button = QPushButton("Delete model + dataset")
        self.delete_all_button.setToolTip(
            "Also remove the dataset snapshot this model was trained on.\n"
            "Your annotations are untouched — a snapshot can be re-exported at any time."
        )
        self.refresh_button = QPushButton("Refresh")
        self.close_button = QPushButton("Close")

        for button in (
            self.prune_button, self.archive_button,
            self.delete_button, self.delete_all_button,
        ):
            button.setEnabled(False)
            row.addWidget(button)
        row.addStretch(1)
        row.addWidget(self.refresh_button)
        row.addWidget(self.close_button)

        self.prune_button.clicked.connect(self._prune_selected)
        self.archive_button.clicked.connect(self._archive_selected)
        self.delete_button.clicked.connect(lambda: self._delete_selected(with_snapshot=False))
        self.delete_all_button.clicked.connect(lambda: self._delete_selected(with_snapshot=True))
        self.refresh_button.clicked.connect(self.refresh)
        self.close_button.clicked.connect(self.accept)
        return row

    # -- data --------------------------------------------------------------

    def refresh(self) -> None:
        self._runs = list_runs(self.settings.outputs_dir, self.settings.datasets_dir)
        self.table.setRowCount(len(self._runs))

        for row, run in enumerate(self._runs):
            snapshot = (
                f"{format_bytes(run.snapshot_bytes)}"
                + (f"  ({run.snapshot_samples:,} samples)" if run.snapshot_samples else "")
                if run.snapshot_dir
                else "— already removed"
            )
            values = [
                run.name,
                run.trained_at,
                str(run.epochs_trained or "-"),
                run.describe_score(),
                f"{run.checkpoint_count} ({run.epoch_copy_count} epoch copies)",
                format_bytes(run.size_bytes),
                snapshot,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(RunRole, row)
                if column == 0 and not run.has_best:
                    item.setForeground(QColor("#d4a72c"))
                    item.setToolTip("No best.pt — this run did not finish an epoch.")
                if column == 6 and not run.snapshot_dir:
                    item.setForeground(QColor("#8b949e"))
                self.table.setItem(row, column, item)

        self._update_summary()
        self._update_actions()

    def _update_summary(self) -> None:
        models = sum(run.size_bytes for run in self._runs)
        snapshots = sum(run.snapshot_bytes for run in self._runs)
        reclaimable = sum(run.reclaimable_bytes for run in self._runs)
        self.summary.setText(
            f"{len(self._runs)} run(s) · models {format_bytes(models)} · "
            f"snapshots {format_bytes(snapshots)} · "
            f"up to {format_bytes(reclaimable)} reclaimable without deleting a model"
        )

    def selected_runs(self) -> list[TrainedRun]:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return [self._runs[row] for row in sorted(rows) if 0 <= row < len(self._runs)]

    def _update_actions(self) -> None:
        selected = self.selected_runs()
        self.delete_button.setEnabled(bool(selected))
        self.delete_all_button.setEnabled(any(run.snapshot_dir for run in selected))
        self.prune_button.setEnabled(any(run.epoch_copy_bytes for run in selected))
        self.archive_button.setEnabled(any(run.optimizer_state_bytes for run in selected))

        if selected:
            freed = sum(run.epoch_copy_bytes for run in selected)
            self.prune_button.setText(
                f"Free space ({format_bytes(freed)})" if freed else "Free space"
            )

    # -- actions -----------------------------------------------------------

    def _prune_selected(self) -> None:
        runs = [run for run in self.selected_runs() if run.epoch_copy_bytes]
        if not runs:
            return
        total = sum(run.epoch_copy_bytes for run in runs)
        if not self._confirm(
            "Free space",
            f"Delete the per-epoch checkpoint copies from {len(runs)} run(s)?\n\n"
            f"This frees {format_bytes(total)}. best.pt and last.pt are kept, so the "
            "trained models remain fully usable and resumable.",
        ):
            return

        freed = 0
        for run in runs:
            try:
                freed += prune_epoch_copies(run, self.settings.outputs_dir)
            except Exception as exc:
                self._report_failure(run, exc)
        self._finish(f"Freed {format_bytes(freed)} by removing epoch copies.")

    def _archive_selected(self) -> None:
        runs = [run for run in self.selected_runs() if run.optimizer_state_bytes]
        if not runs:
            return
        total = sum(run.optimizer_state_bytes for run in runs)
        if not self._confirm(
            "Archive models",
            f"Strip the optimizer state from {len(runs)} run(s)?\n\n"
            f"This frees about {format_bytes(total)}. The models still evaluate, export "
            "and predict exactly as before.\n\n"
            "You will no longer be able to RESUME training these runs. This cannot be "
            "undone.",
        ):
            return

        freed = 0
        for run in runs:
            try:
                freed += archive_run(run, self.settings.outputs_dir)
            except Exception as exc:
                self._report_failure(run, exc)
        self._finish(f"Archived {len(runs)} run(s), freeing {format_bytes(freed)}.")

    def _delete_selected(self, with_snapshot: bool) -> None:
        runs = self.selected_runs()
        if not runs:
            return

        total = sum(
            run.size_bytes + (run.snapshot_bytes if with_snapshot else 0) for run in runs
        )
        names = "\n".join(f"  · {run.name}" for run in runs[:12])
        if len(runs) > 12:
            names += f"\n  · ... and {len(runs) - 12} more"

        extra = (
            "\n\nThe dataset snapshots will also be deleted. Your annotations are NOT "
            "affected — a snapshot can be re-exported from the Dataset tab at any time."
            if with_snapshot
            else "\n\nThe dataset snapshots are kept, so these models can be retrained."
        )
        if not self._confirm(
            "Delete trained model" + ("s" if len(runs) > 1 else ""),
            f"Permanently delete {len(runs)} trained model(s)?\n\n{names}\n\n"
            f"This frees {format_bytes(total)} and cannot be undone.{extra}",
            destructive=True,
        ):
            return

        freed = 0
        for run in runs:
            try:
                freed += delete_run(
                    run,
                    self.settings.outputs_dir,
                    self.settings.datasets_dir,
                    include_snapshot=with_snapshot,
                )
            except Exception as exc:
                self._report_failure(run, exc)
        self._finish(f"Deleted {len(runs)} model(s), freeing {format_bytes(freed)}.")

    # -- helpers -----------------------------------------------------------

    def _confirm(self, title: str, message: str, destructive: bool = False) -> bool:
        default = (
            QMessageBox.StandardButton.Cancel if destructive else QMessageBox.StandardButton.Yes
        )
        return (
            QMessageBox.question(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                default,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _report_failure(self, run: TrainedRun, exc: Exception) -> None:
        LOGGER.exception("Could not modify %s", run.name)
        QMessageBox.warning(
            self,
            "Could not complete",
            f"{run.name}:\n{exc}\n\nA file may be open in another program.",
        )

    def _finish(self, message: str) -> None:
        self.refresh()
        self.models_changed.emit()
        self.summary.setText(f"{message}   ·   {self.summary.text()}")
