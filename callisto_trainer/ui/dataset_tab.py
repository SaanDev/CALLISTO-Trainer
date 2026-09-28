"""Dataset tab: current label counts, and export to immutable snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.settings import AppSettings
from callisto_trainer.store.export import (
    DEFAULT_QUIET_VIEW,
    SNAPSHOT_KINDS,
    ExportResult,
    delete_snapshot,
    export_binary_dataset,
    export_type_dataset,
    export_unified_dataset,
    list_snapshots,
    read_snapshot_info,
    snapshot_size_bytes,
)
from callisto_trainer.core.taxonomy import MIN_SUBCLASS_BOXES, PARENT_LABEL, RFI
from callisto_trainer.core.type_priors import OBSERVED_TYPE_SHARES, SHARE_TYPES
from callisto_trainer.store.repository import (
    BURST_TYPES,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
)

LOGGER = get_logger(__name__)

# Each row carries its own snapshot directory, so a delete targets exactly the
# directory shown rather than one re-derived from the row index after a refresh.
SnapshotPathRole = Qt.ItemDataRole.UserRole + 1


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.0f} MB"
    if num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.0f} kB"
    return f"{num_bytes} B"


class _DeleteWorker(QThread):
    """Remove snapshot directories off the UI thread.

    A single unified snapshot is ~5,500 files and over a gigabyte; deleting a few
    of them on the UI thread freezes the window for seconds. One failure does not
    abandon the rest -- each directory is reported and the run continues.
    """

    progress = Signal(int, int, str)
    finished_with = Signal(int, list)  # bytes freed, [(name, message)]

    def __init__(self, datasets_dir, directories: Sequence[Path]) -> None:
        super().__init__()
        self.datasets_dir = datasets_dir
        self.directories = list(directories)

    def run(self) -> None:
        freed = 0
        errors: list[tuple[str, str]] = []
        total = len(self.directories)
        for index, directory in enumerate(self.directories, start=1):
            self.progress.emit(index, total, directory.name)
            try:
                freed += delete_snapshot(self.datasets_dir, directory)
            except Exception as exc:
                LOGGER.exception("Could not delete snapshot %s", directory)
                errors.append((directory.name, str(exc)))
        self.finished_with.emit(freed, errors)


class _ExportWorker(QThread):
    progress = Signal(int, int, str)
    finished_with = Signal(object)
    failed = Signal(str)

    def __init__(
        self, repository, settings: AppSettings, kind: str, options: dict | None = None
    ):
        super().__init__()
        self.repository = repository
        self.settings = settings
        self.kind = kind
        # Extra keyword arguments for the exporter (unified track only).
        self.options = dict(options or {})
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        def report(index: int, total: int, name: str) -> bool:
            self.progress.emit(index, total, name)
            return not self._cancelled

        exporter = {
            "unified": export_unified_dataset,
            "types": export_type_dataset,
            "binary": export_binary_dataset,
        }[self.kind]
        try:
            result = exporter(
                self.repository,
                self.settings.datasets_dir,
                self.settings.pipeline,
                self.settings.outputs_dir,
                progress=report,
                **self.options,
            )
        except Exception as exc:
            LOGGER.exception("Export failed")
            self.failed.emit(repr(exc))
            return
        self.finished_with.emit(result)


class DatasetTab(QWidget):
    """Review what has been labelled, then freeze it into a training snapshot."""

    # Emitted whenever the set of snapshots on disk changes, in either direction:
    # the Train and Evaluate tabs list snapshots and must not keep offering one
    # that has just been deleted.
    snapshots_changed = Signal()

    def __init__(
        self,
        repository: AnnotationRepository,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository = repository
        self.settings = settings
        self._worker: _ExportWorker | None = None
        self._delete_worker: _DeleteWorker | None = None
        # Set by the host window to report snapshots a run currently holds open.
        # Left None in isolation, where nothing else is running.
        self.snapshot_in_use: Callable[[], Sequence[Path | str]] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_counts_group())
        layout.addWidget(self._build_export_group())
        layout.addWidget(self._build_snapshots_group(), 1)

        self.refresh()

    # -- construction ------------------------------------------------------

    def _build_counts_group(self) -> QGroupBox:
        group = QGroupBox("Labelled so far")
        layout = QVBoxLayout(group)
        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.counts_label)
        self.balance_label = QLabel("")
        self.balance_label.setWordWrap(True)
        self.balance_label.setStyleSheet("color: #d4a72c;")
        layout.addWidget(self.balance_label)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export a training snapshot")
        layout = QVBoxLayout(group)

        note = QLabel(
            "A snapshot is a frozen copy of the current labels: the tensors, a manifest with "
            "the train/val/test split, and a ready-to-run config. Snapshots are never "
            "modified afterwards, so a trained model can always be traced back to the exact "
            "data behind it."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b949e;")
        layout.addWidget(note)

        self.export_unified = QPushButton("Export unified dataset  (recommended)")
        self.export_unified.setMinimumHeight(38)
        self.export_unified.setToolTip(
            "One dataset over regions: No_Burst, RFI, Type II, Type III, Type IIIG, "
            "Type IV, Other.\n"
            "Trains a single model that answers burst-or-not, which type, and where.\n"
            "Background samples are mined with the same region finder used at "
            "inference, outside your burst boxes and in no-burst files, so the model "
            "learns to reject the interference it will actually be shown. Those that "
            "measure like interference (carriers, impulses, sweeps, periodic signals, "
            "gain steps) are labelled RFI automatically. Each sample carries the exact "
            "crop, a wide context view and interference features."
        )
        layout.addWidget(self.export_unified)

        options = QHBoxLayout()
        self.hard_negatives = QCheckBox("Mine hard negatives with the latest unified model")
        self.hard_negatives.setToolTip(
            "Rank background candidates by how strongly the newest trained unified "
            "model called them a burst, and take the worst first. Each retraining "
            "then concentrates on the previous model's own false positives. "
            "Available once a unified model has been trained."
        )
        self.synthetic_rfi = QCheckBox("Add synthetic RFI examples")
        self.synthetic_rfi.setChecked(True)
        self.synthetic_rfi.setToolTip(
            "Paint carriers, impulses, sweeps, periodic pulses and gain steps onto a "
            "fifth of the no-burst files and train on the result as RFI, so shapes "
            "that are rare in your labels are still learned as interference."
        )
        self.drop_lines = QCheckBox("Keep carrier-like lines out of burst classes")
        self.drop_lines.setChecked(True)
        self.drop_lines.setToolTip(
            "A thin horizontal region inside a burst box is usually a carrier crossing "
            "the box, and training it as a burst teaches the model to flag carriers. "
            "On: such regions are left out (the drawn box still teaches its burst). "
            "Measured: carrier-shaped false alarms 7 -> 0, at the cost of a few thin "
            "real lanes. Turn off to compare once your burst boxes are drawn tightly "
            "around the bursts themselves."
        )
        self.quiet_view = QCheckBox("Add the quiet-background view (Type IV)")
        self.quiet_view.setChecked(DEFAULT_QUIET_VIEW)
        self.quiet_view.setToolTip(
            "Give every sample a third view: the context strip with each channel's "
            "background taken from its quietest tenth instead of its median. A "
            "continuum lasting most of the file (Type IV) is flattened by the median "
            "and stays bright in this view. Snapshots grow by half."
        )
        options.addWidget(self.hard_negatives)
        options.addWidget(self.synthetic_rfi)
        options.addWidget(self.drop_lines)
        options.addWidget(self.quiet_view)
        options.addStretch(1)
        layout.addLayout(options)
        layout.addLayout(self._build_frequency_row())

        legacy = QLabel("Separate models (kept so earlier checkpoints stay reproducible):")
        legacy.setStyleSheet("color: #8b949e; font-size: 11px; margin-top: 4px;")
        layout.addWidget(legacy)

        row = QHBoxLayout()
        self.export_types = QPushButton("Burst-type only (one per box)")
        self.export_binary = QPushButton("Burst / no-burst only (one per file)")
        row.addWidget(self.export_types)
        row.addWidget(self.export_binary)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.export_unified.clicked.connect(lambda: self._start_export("unified"))
        self.export_types.clicked.connect(lambda: self._start_export("types"))
        self.export_binary.clicked.connect(lambda: self._start_export("binary"))
        return group

    def _build_frequency_row(self) -> QHBoxLayout:
        """How often each burst type really occurs, as a share of all bursts.

        Written into the training config. After training the model's type
        probabilities are shifted from the (deliberately balanced) training mix
        toward these, so an ambiguous burst is called the commoner type; see
        core/type_priors.py.
        """
        row = QHBoxLayout()
        caption = QLabel("How often each type occurs (% of bursts):")
        caption.setToolTip(
            "The share of each burst type among the bursts CALLISTO records. The model "
            "is trained balanced so rare types are still learned, then its type "
            "probabilities are corrected toward these shares, with a strength chosen on "
            "the validation data. Type IIIG counts within Type III. Only which type a "
            "burst is called changes; the burst / no-burst decision does not."
        )
        row.addWidget(caption)
        self.type_frequency: dict[str, QDoubleSpinBox] = {}
        for name in SHARE_TYPES:
            box = QDoubleSpinBox()
            box.setRange(0.0, 100.0)
            box.setDecimals(1)
            box.setSingleStep(0.5)
            box.setSuffix(" %")
            box.setValue(100.0 * OBSERVED_TYPE_SHARES[name])
            row.addWidget(QLabel(name))
            row.addWidget(box)
            self.type_frequency[name] = box
        row.addStretch(1)
        return row

    def _build_snapshots_group(self) -> QGroupBox:
        group = QGroupBox("Existing snapshots")
        layout = QVBoxLayout(group)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Snapshot", "Kind", "Samples", "Size", "Classes", "Splits"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._update_delete_buttons)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.delete_selected_button = QPushButton("Delete selected")
        self.delete_selected_button.setToolTip(
            "Permanently remove the selected snapshot directories from disk.\n"
            "Trained checkpoints live under outputs/ and are not touched, but they "
            "lose the data they can be traced back to."
        )
        self.delete_all_button = QPushButton("Delete all snapshots")
        self.delete_all_button.setToolTip(
            "Permanently remove every snapshot directory. Labels in the database "
            "are not affected -- you can always export again."
        )
        row.addWidget(self.delete_selected_button)
        row.addWidget(self.delete_all_button)
        row.addStretch(1)
        self.disk_label = QLabel("")
        self.disk_label.setStyleSheet("color: #8b949e;")
        row.addWidget(self.disk_label)
        layout.addLayout(row)

        self.delete_selected_button.clicked.connect(self._delete_selected)
        self.delete_all_button.clicked.connect(self._delete_all)
        return group

    # -- data --------------------------------------------------------------

    def refresh(self) -> None:
        verdicts = self.repository.verdict_counts()
        types = self.repository.box_type_counts()
        burst = verdicts.get(VERDICT_BURST, 0)
        no_burst = verdicts.get(VERDICT_NO_BURST, 0)

        type_text = "   ".join(f"{name}: {types.get(name, 0):,}" for name in BURST_TYPES)
        burst_boxes = sum(types.get(name, 0) for name in BURST_TYPES)
        self.counts_label.setText(
            f"Files: {burst:,} burst · {no_burst:,} no burst\n"
            f"Marked bursts: {burst_boxes:,}    ({type_text})\n"
            "Interference (RFI) is found automatically at export, outside the burst "
            "boxes and in no-burst files."
        )

        warnings = []
        # Classes fold into their fallback until they have enough examples;
        # say so up front instead of letting the class silently vanish. RFI is
        # never drawn, so its count is only known at export.
        for child, parent in PARENT_LABEL.items():
            if child == RFI:
                continue
            count = types.get(child, 0)
            if count < MIN_SUBCLASS_BOXES:
                warnings.append(
                    f"{child}: {count} box(es). Until there are {MIN_SUBCLASS_BOXES}, "
                    f"they train as {parent}."
                )
        top_level = [name for name in BURST_TYPES if name not in PARENT_LABEL]
        missing = [name for name in top_level if types.get(name, 0) == 0]
        if missing:
            warnings.append(
                "No examples yet of: " + ", ".join(missing)
                + ". A class with no boxes is left out of the model."
            )
        elif burst_boxes and min(types.get(n, 0) for n in top_level) < 20:
            warnings.append(
                "One or more burst types has under 20 examples. Training will run, but "
                "expect the rare class to score poorly until it has more."
            )
        if burst == 0 or no_burst == 0:
            warnings.append(
                "The burst / no-burst model needs files of both kinds before it can train."
            )
        elif max(burst, no_burst) / min(burst, no_burst) >= 3.0:
            larger, smaller = (
                ("burst", "no-burst") if burst > no_burst else ("no-burst", "burst")
            )
            ratio = max(burst, no_burst) / min(burst, no_burst)
            warnings.append(
                f"Burst / no-burst files are imbalanced {ratio:.1f}:1 in favour of "
                f"{larger}. Metrics will look good even for a model that always answers "
                f"'{larger}', and real {smaller} files will be misclassified. Label more "
                f"{smaller} files."
            )
        self.balance_label.setText("\n".join(warnings))

        self.export_unified.setEnabled(burst_boxes > 0 and no_burst > 0)
        self.export_types.setEnabled(burst_boxes > 0)
        self.export_binary.setEnabled(burst > 0 and no_burst > 0)
        self.hard_negatives.setEnabled(self._latest_unified_checkpoint() is not None)
        if not self.hard_negatives.isEnabled():
            self.hard_negatives.setChecked(False)
        self._refresh_snapshots()

    def _latest_unified_checkpoint(self):
        from callisto_trainer.services.assist import find_latest_checkpoint

        return find_latest_checkpoint(self.settings.outputs_dir, "unified")

    def _unified_options(self) -> dict:
        options: dict = {}
        if self.hard_negatives.isChecked():
            checkpoint = self._latest_unified_checkpoint()
            if checkpoint is not None:
                options["hard_negative_checkpoint"] = checkpoint
        if not self.synthetic_rfi.isChecked():
            options["synthetic_rfi_ratio"] = 0.0
        if not self.drop_lines.isChecked():
            options["drop_line_positives"] = False
        if self.quiet_view.isChecked() != DEFAULT_QUIET_VIEW:
            options["quiet_view"] = self.quiet_view.isChecked()
        options["type_frequencies"] = {
            name: box.value() for name, box in self.type_frequency.items()
        }
        return options

    def _refresh_snapshots(self) -> None:
        rows: list[tuple[Path, dict]] = []
        for kind, _label, _task in SNAPSHOT_KINDS:
            for directory in list_snapshots(self.settings.datasets_dir, kind):
                rows.append((directory, read_snapshot_info(directory)))

        self.table.setRowCount(len(rows))
        total_bytes = 0
        for index, (directory, info) in enumerate(rows):
            classes = "  ".join(
                f"{name}: {count:,}" for name, count in sorted(info.get("class_counts", {}).items())
            )
            splits = "  ".join(
                f"{name}: {count:,}" for name, count in sorted(info.get("split_counts", {}).items())
            )
            size = snapshot_size_bytes(directory)
            total_bytes += size
            for column, text in enumerate(
                [
                    directory.name,
                    info.get("kind", directory.parent.name),
                    f"{info.get('samples', 0):,}",
                    _format_size(size),
                    classes,
                    splits,
                ]
            ):
                item = QTableWidgetItem(text)
                item.setToolTip(str(directory))
                # The path travels with the row so a deletion always targets the
                # directory the operator saw, not a re-derived guess at it.
                item.setData(SnapshotPathRole, str(directory))
                self.table.setItem(index, column, item)

        self.disk_label.setText(
            f"{len(rows)} snapshot(s), {_format_size(total_bytes)} on disk" if rows else ""
        )
        self.delete_all_button.setEnabled(bool(rows))
        self._update_delete_buttons()

    # -- deletion ----------------------------------------------------------

    def _update_delete_buttons(self) -> None:
        busy = self._delete_worker is not None or self._worker is not None
        self.delete_selected_button.setEnabled(bool(self._selected_snapshots()) and not busy)
        self.delete_all_button.setEnabled(self.table.rowCount() > 0 and not busy)

    def _selected_snapshots(self) -> list[Path]:
        paths: list[Path] = []
        for index in self.table.selectionModel().selectedRows() if self.table.selectionModel() else []:
            item = self.table.item(index.row(), 0)
            if item is not None:
                paths.append(Path(item.data(SnapshotPathRole)))
        return paths

    def _all_snapshots(self) -> list[Path]:
        return [
            Path(self.table.item(row, 0).data(SnapshotPathRole))
            for row in range(self.table.rowCount())
            if self.table.item(row, 0) is not None
        ]

    def _delete_selected(self) -> None:
        self._confirm_and_delete(self._selected_snapshots(), "the selected snapshot(s)")

    def _delete_all(self) -> None:
        self._confirm_and_delete(self._all_snapshots(), "every snapshot")

    def _confirm_and_delete(self, directories: list[Path], description: str) -> None:
        if not directories or self._delete_worker is not None or self._worker is not None:
            return

        # A snapshot being trained or evaluated on right now would have its
        # manifest pulled out from under a running process; that surfaces as an
        # unreadable-file crash several minutes in, so refuse up front.
        in_use = self._snapshots_in_use()
        blocked = [d for d in directories if d.resolve() in in_use]
        if blocked:
            QMessageBox.warning(
                self,
                "Snapshot in use",
                "A run is using "
                + ", ".join(d.name for d in blocked)
                + ".\n\nStop it on the Train or Evaluate tab before deleting.",
            )
            return

        total = sum(snapshot_size_bytes(d) for d in directories)
        names = "\n".join(f"  · {d.parent.name}/{d.name}" for d in directories[:12])
        if len(directories) > 12:
            names += f"\n  · ... and {len(directories) - 12} more"

        choice = QMessageBox.question(
            self,
            "Delete snapshots?",
            f"Permanently delete {description}?\n\n{names}\n\n"
            f"This frees {_format_size(total)} and cannot be undone.\n\n"
            "Your labels are not affected -- they live in the database and you can "
            "export again at any time. Trained checkpoints under outputs/ are also "
            "left alone, but they will no longer point at the data they were "
            "trained on.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, len(directories))
        self.progress.setValue(0)
        self.status.setText(f"Deleting {len(directories)} snapshot(s)...")

        self._delete_worker = _DeleteWorker(self.settings.datasets_dir, directories)
        self._delete_worker.progress.connect(self._on_delete_progress)
        self._delete_worker.finished_with.connect(self._on_deleted)
        self._delete_worker.finished.connect(self._release_delete_worker)
        self._delete_worker.start()
        self._update_delete_buttons()

    def _snapshots_in_use(self) -> set[Path]:
        """Snapshot directories a run currently holds open, via the host window."""
        if self.snapshot_in_use is None:
            return set()
        try:
            return {Path(p).resolve() for p in self.snapshot_in_use() if p}
        except Exception:  # a broken hook must never block a deletion outright
            LOGGER.exception("snapshot_in_use hook failed")
            return set()

    def _release_delete_worker(self) -> None:
        worker, self._delete_worker = self._delete_worker, None
        if worker is not None:
            worker.deleteLater()
        self._update_delete_buttons()

    def _on_delete_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"Deleting {done} of {total}: {name}")

    def _on_deleted(self, freed: int, errors: list) -> None:
        self.progress.setVisible(False)
        lines = [f"Deleted {_format_size(freed)} of snapshots."]
        for name, message in errors:
            lines.append(f"Could not delete {name}: {message}")
        self.status.setText("\n".join(lines))
        self.refresh()
        self.snapshots_changed.emit()

    # -- export ------------------------------------------------------------

    def _start_export(self, kind: str) -> None:
        if self._worker is not None:
            return
        for button in (self.export_unified, self.export_types, self.export_binary):
            button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Exporting...")

        self._worker = _ExportWorker(
            self.repository,
            self.settings,
            kind,
            self._unified_options() if kind == "unified" else None,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_with.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._release_worker)
        self._worker.start()


    def _release_worker(self) -> None:
        """Drop the finished worker once its thread has actually stopped.

        Clearing the reference from inside a result handler destroys the QThread
        while ``run()`` is still returning -- Qt reports "Destroyed while thread
        is still running" and the process can crash. ``QThread.finished`` fires
        after ``run()`` has returned, which is the only safe moment.
        """
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_progress(self, index: int, total: int, name: str) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(index)
        self.status.setText(f"Exporting {index:,} of {total:,}: {name}")

    def _on_finished(self, result: ExportResult) -> None:
        self.progress.setVisible(False)

        lines = [f"Wrote {result.summary()} to {result.directory}"]
        if result.folded:
            lines.append(
                "Folded for now (too few examples): "
                + ", ".join(f"{child} → {parent}" for child, parent in result.folded.items())
            )
        if result.automatic_rfi:
            found = sum(result.automatic_rfi.values())
            kinds = ", ".join(
                f"{kind} {count:,}" for kind, count in
                sorted(result.automatic_rfi.items(), key=lambda item: -item[1])
            )
            lines.append(f"{found:,} background region(s) labelled RFI automatically ({kinds}).")
        if result.synthetic_rfi:
            lines.append(f"{result.synthetic_rfi:,} synthetic RFI example(s) added.")
        smaller = result.overlap_labelled.get("smaller box", 0)
        joined = result.overlap_labelled.get("joined boxes", 0)
        if smaller or joined or result.mixed_type_regions_dropped:
            lines.append(
                f"Overlapping boxes: {smaller:,} region(s) took the smaller box's type, "
                f"{joined:,} straddling same-type boxes were kept as that type, and "
                f"{result.mixed_type_regions_dropped:,} split between two types were left out."
            )
        dropped = result.carrier_regions_dropped + result.line_regions_dropped
        if dropped:
            lines.append(
                f"{dropped:,} carrier or line-shaped region(s) inside burst boxes were left "
                "out rather than trained as bursts."
            )
        if result.failed:
            lines.append(f"{result.failed} sample(s) could not be written.")
        problems = result.blocking_problems()
        if problems:
            lines.append("Before training: " + "  ".join(problems))
        for warning in result.balance_warnings():
            lines.append("⚠ " + warning)
        self.status.setText("\n".join(lines))
        self.refresh()
        self.snapshots_changed.emit()

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.status.setText(f"Export failed: {message}")
        self.refresh()
