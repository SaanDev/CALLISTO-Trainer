"""Batch import of FITS files.

Scanning and header-reading happen on a worker thread so a folder with tens of
thousands of files never freezes the window, and the run can be cancelled at any
point without losing what has already been imported.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.importer import ImportResult, find_fits_files, import_files
from callisto_trainer.store.repository import AnnotationRepository

LOGGER = get_logger(__name__)


class _ImportWorker(QThread):
    progress = Signal(int, int, str)
    finished_with = Signal(object)  # ImportResult

    def __init__(self, repository: AnnotationRepository, paths: list[Path], read_headers: bool):
        super().__init__()
        self.repository = repository
        self.paths = paths
        self.read_headers = read_headers
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        def report(index: int, total: int, name: str) -> bool:
            self.progress.emit(index, total, name)
            return not self._cancelled

        result = import_files(
            self.repository,
            self.paths,
            read_headers=self.read_headers,
            progress=report,
        )
        self.finished_with.emit(result)


class ImportTab(QWidget):
    """Choose folders, review what was found, then import."""

    imported = Signal()

    def __init__(self, repository: AnnotationRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self._found: list[Path] = []
        self._worker: _ImportWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Import e-CALLISTO <b>.fit.gz</b> files. Files are referenced where they are "
            "and never copied, so importing from a large archive costs no extra disk. "
            "Re-importing the same folder is safe: duplicates are detected by path and "
            "by file content."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        layout.addWidget(intro)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_found_group(), 1)
        layout.addWidget(self._build_action_group())

        self.refresh_summary()

    # -- construction ------------------------------------------------------

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("1. Choose what to import")
        row = QHBoxLayout(group)

        self.add_folder = QPushButton("Add folder...")
        self.add_files = QPushButton("Add files...")
        self.clear_list = QPushButton("Clear")
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        self.read_headers = QCheckBox("Read FITS headers (recommended)")
        self.read_headers.setChecked(True)
        self.read_headers.setToolTip(
            "Reads station, date and the true frequency axis. Slower, but the queue "
            "and the metadata branch of the binary model both depend on it."
        )

        for widget in (self.add_folder, self.add_files, self.clear_list, self.recursive, self.read_headers):
            row.addWidget(widget)
        row.addStretch(1)

        self.add_folder.clicked.connect(self._choose_folder)
        self.add_files.clicked.connect(self._choose_files)
        self.clear_list.clicked.connect(self._clear)
        return group

    def _build_found_group(self) -> QGroupBox:
        group = QGroupBox("2. Files found")
        layout = QVBoxLayout(group)
        self.found_list = QListWidget()
        self.found_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.found_label = QLabel("Nothing selected yet.")
        self.found_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.found_label)
        layout.addWidget(self.found_list)
        return group

    def _build_action_group(self) -> QGroupBox:
        group = QGroupBox("3. Import")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.import_button = QPushButton("Import into dataset")
        self.import_button.setMinimumHeight(34)
        self.import_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        row.addWidget(self.import_button, 1)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.summary)

        self.import_button.clicked.connect(self._start_import)
        self.cancel_button.clicked.connect(self._cancel_import)
        return group

    # -- selection ---------------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of FITS files")
        if folder:
            self._scan([Path(folder)])

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS files", "", "FITS (*.fit.gz *.fits.gz *.fit *.fits)"
        )
        if files:
            self._scan([Path(f) for f in files])

    def _scan(self, roots: list[Path]) -> None:
        self.status.setText("Scanning...")
        found = find_fits_files(roots, recursive=self.recursive.isChecked())
        known = set(self._found)
        self._found.extend(path for path in found if path not in known)

        self.found_list.clear()
        # Only a preview is listed; a 100k-row widget would be pointless here.
        for path in self._found[:500]:
            self.found_list.addItem(str(path))
        if len(self._found) > 500:
            self.found_list.addItem(f"... and {len(self._found) - 500:,} more")

        self.found_label.setText(f"{len(self._found):,} FITS files selected.")
        self.import_button.setEnabled(bool(self._found))
        self.status.setText("")

    def _clear(self) -> None:
        self._found.clear()
        self.found_list.clear()
        self.found_label.setText("Nothing selected yet.")
        self.import_button.setEnabled(False)

    # -- import ------------------------------------------------------------

    def _start_import(self) -> None:
        if not self._found or self._worker is not None:
            return

        self.import_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self._found))
        self.progress.setValue(0)

        self._worker = _ImportWorker(self.repository, list(self._found), self.read_headers.isChecked())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_with.connect(self._on_finished)
        self._worker.finished.connect(self._release_worker)
        self._worker.start()

    def _cancel_import(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling...")


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
        self.progress.setValue(index)
        self.status.setText(f"Importing {index:,} of {total:,}: {name}")

    def _on_finished(self, result: ImportResult) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.import_button.setEnabled(bool(self._found))
        self.status.setText(f"Done: {result.summary()}")
        if result.errors:
            first = result.errors[0]
            self.status.setText(
                f"{self.status.text()}  First problem: {Path(first[0]).name} - {first[1]}"
            )

        self._clear()
        self.refresh_summary()
        self.imported.emit()

    def refresh_summary(self) -> None:
        total = self.repository.total_files()
        counts = self.repository.status_counts()
        boxes = self.repository.total_boxes()
        self.summary.setText(
            f"Dataset: {total:,} files "
            f"({counts.get('labeled', 0):,} reviewed, {counts.get('pending', 0):,} pending, "
            f"{counts.get('error', 0):,} unreadable) · {boxes:,} marked bursts"
        )
