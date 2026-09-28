"""Main window: Import · Label · Dataset · Train · Evaluate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from callisto_trainer import __version__
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.settings import AppSettings
from callisto_trainer.store.db import Database
from callisto_trainer.store.repository import AnnotationRepository
from callisto_trainer.ui.dataset_tab import DatasetTab
from callisto_trainer.ui.evaluate_tab import EvaluateTab
from callisto_trainer.ui.import_tab import ImportTab
from callisto_trainer.ui.label_tab import LabelTab
from callisto_trainer.ui.predict_tab import PredictTab
from callisto_trainer.ui.train_tab import TrainTab

LOGGER = get_logger(__name__)

STATE_GEOMETRY = "window.geometry"
STATE_TAB = "window.tab"


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        settings.ensure_directories()

        self.database = Database(settings.database_path)
        self.repository = AnnotationRepository(self.database)

        self.setWindowTitle(f"CALLISTO Trainer {__version__}")
        self.resize(1600, 950)

        self.tabs = QTabWidget()
        self.import_tab = ImportTab(self.repository)
        self.label_tab = LabelTab(self.repository, settings)
        self.dataset_tab = DatasetTab(self.repository, settings)
        self.train_tab = TrainTab(settings)
        self.evaluate_tab = EvaluateTab(settings)
        self.predict_tab = PredictTab(settings)

        self.tabs.addTab(self.import_tab, "Import")
        self.tabs.addTab(self.label_tab, "Label")
        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.evaluate_tab, "Evaluate")
        self.tabs.addTab(self.predict_tab, "Predict")
        self.setCentralWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("")
        self.status.addPermanentWidget(self.status_label)

        self.import_tab.imported.connect(self._on_imported)
        self.label_tab.dataset_changed.connect(self._refresh_status)
        self.dataset_tab.snapshots_changed.connect(self.train_tab.refresh_snapshots)
        self.dataset_tab.snapshots_changed.connect(self.evaluate_tab.refresh_snapshots)
        # The Dataset tab owns snapshot deletion but cannot see the runners, so
        # it asks here which directories a run is currently reading.
        self.dataset_tab.snapshot_in_use = self._snapshots_in_use
        self.train_tab.training_finished.connect(self.evaluate_tab.refresh_snapshots)
        self.train_tab.training_finished.connect(self.predict_tab.refresh_models)
        self.evaluate_tab.inspect_file_requested.connect(self._inspect_file)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_menu()
        self._restore_state()
        self._refresh_status()

    # -- menu --------------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        goto_import = QAction("Go to &Import", self)
        goto_import.setShortcut(QKeySequence("Ctrl+1"))
        goto_import.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(goto_import)

        goto_label = QAction("Go to &Label", self)
        goto_label.setShortcut(QKeySequence("Ctrl+2"))
        goto_label.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        file_menu.addAction(goto_label)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        reset_menu = self.menuBar().addMenu("&Reset")

        reset_view = QAction("Reset &view and layout", self)
        reset_view.setToolTip("Restore zoom, contrast, colormap, filters and panel sizes.")
        reset_view.triggered.connect(self._reset_view_and_layout)
        reset_menu.addAction(reset_view)

        clear_file = QAction("Clear &this file's labels", self)
        clear_file.setToolTip("Remove all boxes and the verdict for the file on screen.")
        clear_file.triggered.connect(self._clear_current_file)
        reset_menu.addAction(clear_file)

        reset_menu.addSeparator()

        recheck_axes = QAction("Recheck frequency &axes...", self)
        recheck_axes.setToolTip(
            "Re-read every file's frequency and time axes and re-measure the bursts "
            "whose axis changed. Run this after the reader improves; your labels are "
            "never touched."
        )
        recheck_axes.triggered.connect(self._recheck_axes)
        reset_menu.addAction(recheck_axes)

        manage_models = QAction("Manage trained &models...", self)
        manage_models.setToolTip(
            "Review trained models, free the space their checkpoints occupy, or "
            "delete runs you no longer need."
        )
        manage_models.triggered.connect(self._manage_models)
        reset_menu.addAction(manage_models)

        clear_cache = QAction("Clear display &cache", self)
        clear_cache.setToolTip(
            "Delete cached spectra so every file is re-read from the original FITS."
        )
        clear_cache.triggered.connect(self._clear_display_cache)
        reset_menu.addAction(clear_cache)

        reset_all = QAction("Reset &entire dataset...", self)
        reset_all.setToolTip("Delete every imported file, box and verdict. Irreversible.")
        reset_all.triggered.connect(self._reset_dataset)
        reset_menu.addAction(reset_all)

        assist_menu = self.menuBar().addMenu("&Assist")
        triage = QAction("&Score queue for triage...", self)
        triage.setToolTip(
            "Run the newest trained unified model (or burst / no-burst model) over "
            "unreviewed files and sort the queue by burst probability, so likely "
            "bursts -- and the model's likely false alarms -- come first."
        )
        triage.triggered.connect(self._score_for_triage)
        assist_menu.addAction(triage)

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts = QAction("&Keyboard shortcuts", self)
        shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts)

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard shortcuts",
            "<b>Navigation</b><br>"
            "A / ← &nbsp; previous file<br>"
            "D / → &nbsp; next file<br>"
            "F &nbsp; next unreviewed file<br>"
            "R &nbsp; reset zoom<br><br>"
            "<b>Labelling</b><br>"
            "B &nbsp; mark as Burst<br>"
            "N &nbsp; mark as No burst<br>"
            "U &nbsp; mark as Unsure<br>"
            "1 / 2 / 3 / 4 / 5 &nbsp; set selected box to Type II / Type III / "
            "Type IIIG / Type IV / Other<br>"
            "P &nbsp; suggest regions<br>"
            "V &nbsp; toggle raw / preprocessed view<br>"
            "Del &nbsp; delete selected box<br><br>"
            "<b>Canvas</b><br>"
            "Drag &nbsp; draw a box around a burst (interference is found "
            "automatically; never box it)<br>"
            "Shift-drag or middle-drag &nbsp; pan<br>"
            "Scroll &nbsp; zoom",
        )

    # -- state -------------------------------------------------------------

    def _restore_state(self) -> None:
        geometry = self.repository.get_state(STATE_GEOMETRY)
        if geometry:  # an empty value means "reset to defaults"
            try:
                self.restoreGeometry(QByteArray.fromHex(geometry.encode("ascii")))
            except Exception:
                LOGGER.debug("Could not restore window geometry")

        tab = self.repository.get_state(STATE_TAB)
        if tab and tab.isdigit():
            self.tabs.setCurrentIndex(min(int(tab), self.tabs.count() - 1))

        if self.repository.total_files():
            self.label_tab.restore_session()

    def _snapshots_in_use(self) -> list[Path]:
        """Snapshot directories a training or evaluation run is reading right now.

        Both runners are launched with ``--config <snapshot>/config.yaml``, so the
        config's parent is the snapshot. Deleting one mid-run would pull the
        manifest out from under a child process and surface minutes later as an
        unreadable-file crash.
        """
        directories: list[Path] = []
        for tab in (self.train_tab, self.evaluate_tab):
            runner = getattr(tab, "runner", None)
            if runner is not None and runner.is_running and runner.job is not None:
                directories.append(Path(runner.job.config_path).parent)
        return directories

    def _on_tab_changed(self, index: int) -> None:
        self.repository.set_state(STATE_TAB, str(index))
        widget = self.tabs.widget(index)
        if widget is self.import_tab:
            self.import_tab.refresh_summary()
        elif widget is self.dataset_tab:
            self.dataset_tab.refresh()
        elif widget is self.train_tab:
            self.train_tab.refresh_snapshots()
        elif widget is self.evaluate_tab:
            self.evaluate_tab.refresh_snapshots()
        elif widget is self.predict_tab:
            self.predict_tab.refresh_models()

    # -- reset actions -----------------------------------------------------

    def _reset_view_and_layout(self) -> None:
        self.label_tab.reset_view_and_layout()
        self.resize(1600, 950)
        self.repository.set_state(STATE_GEOMETRY, "")
        self.status.showMessage("View and layout reset.", 4000)

    def _clear_current_file(self) -> None:
        record_id = self.label_tab._current_file_id
        if record_id is None:
            QMessageBox.information(self, "No file open", "Open a file in the Label tab first.")
            return
        record = self.repository.get_file(record_id)
        boxes = self.repository.boxes_for_file(record_id)
        if record is None or (record.verdict is None and not boxes):
            self.status.showMessage("This file has no labels to clear.", 4000)
            return

        confirm = QMessageBox.question(
            self,
            "Clear this file's labels?",
            f"Remove the verdict and {len(boxes)} marked burst(s) from "
            f"{record.file_name}?\n\nThe file stays imported and returns to unreviewed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.label_tab.clear_current_file_labels()
            self._refresh_status()

    def _recheck_axes(self) -> None:
        """Re-derive frequency axes and burst physics from the files themselves.

        Needed whenever the FITS reader improves: a file's frequency range and
        every measured drift rate are derived values, and nothing else notices
        that they have gone stale.
        """
        from PySide6.QtWidgets import QProgressDialog

        from callisto_trainer.services.repair import refresh_axes_and_physics

        total = self.repository.total_files()
        if total == 0:
            self.status.showMessage("No files imported yet.", 4000)
            return

        confirm = QMessageBox.question(
            self,
            "Recheck frequency axes?",
            f"Re-read the axes of {total:,} file(s) and re-measure any bursts whose "
            "frequency axis changed.\n\n"
            "Drift rates are measured in MHz per second, so they are only correct once "
            "the frequency axis is. Your boxes, types and verdicts are not modified.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        dialog = QProgressDialog("Rechecking axes...", "Cancel", 0, total, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)

        def report(index: int, count: int, name: str) -> bool:
            dialog.setValue(index)
            dialog.setLabelText(f"Checking {index:,} of {count:,}\n{name}")
            QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            result = refresh_axes_and_physics(self.repository, progress=report)
        except Exception as exc:
            LOGGER.exception("Axis recheck failed")
            dialog.close()
            QMessageBox.critical(self, "Recheck failed", str(exc))
            return
        dialog.close()

        message = [result.summary()]
        if result.examples:
            message += ["", "Corrected ranges, for example:"]
            message += [f"  {line}" for line in result.examples[:6]]
        if result.boxes_remeasured:
            message += [
                "",
                "Drift rates were recomputed for the affected bursts. Re-export any "
                "dataset snapshot you intend to train on so it picks up the corrected "
                "physics.",
            ]
        QMessageBox.information(self, "Frequency axes rechecked", "\n".join(message))

        self.label_tab.loader.cache.clear()
        self.label_tab.refresh_queue()
        self.dataset_tab.refresh()
        self._refresh_status()

    def _manage_models(self) -> None:
        """Open the trained-model manager and re-sync the tabs afterwards."""
        from callisto_trainer.ui.model_manager import ModelManagerDialog

        dialog = ModelManagerDialog(self.settings, self)
        dialog.models_changed.connect(self._refresh_model_lists)
        dialog.exec()
        self._refresh_model_lists()

    def _refresh_model_lists(self) -> None:
        """Re-read snapshots and checkpoints wherever they are offered.

        Needed after a deletion: the Train, Evaluate and Predict tabs all hold
        combo boxes pointing at paths that may no longer exist.
        """
        self.dataset_tab.refresh()
        self.train_tab.refresh_snapshots()
        self.evaluate_tab.refresh_snapshots()
        self.predict_tab.refresh_models()

    def _clear_display_cache(self) -> None:
        loader = self.label_tab.loader
        loader.cache.clear()
        removed = 0
        if loader.disk_cache is not None:
            removed = len(list(Path(loader.disk_cache.directory).glob("*.npy")))
            loader.disk_cache.clear()
        self.status.showMessage(
            f"Display cache cleared ({removed:,} cached spectra removed). "
            "The next pass will re-read from the original files.",
            6000,
        )

    def _reset_dataset(self) -> None:
        """Delete everything, after an explicit typed confirmation and a backup."""
        from PySide6.QtWidgets import QInputDialog

        total_files = self.repository.total_files()
        total_boxes = self.repository.total_boxes(confirmed_only=False)
        if total_files == 0:
            self.status.showMessage("The dataset is already empty.", 4000)
            return

        text, accepted = QInputDialog.getText(
            self,
            "Reset entire dataset",
            f"This permanently deletes {total_files:,} imported file record(s) and "
            f"{total_boxes:,} marked burst(s).\n\n"
            "Your FITS files on disk are NOT touched, and a backup of the database "
            "will be written first, but every label you have made will be gone from "
            "the app.\n\n"
            'Type  RESET  to confirm:',
        )
        if not accepted or text.strip().upper() != "RESET":
            self.status.showMessage("Reset cancelled.", 4000)
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.settings.database_path.with_name(
            f"{self.settings.database_path.stem}.backup_{stamp}.db"
        )
        try:
            self.repository.reset_dataset(backup_path)
        except Exception as exc:
            LOGGER.exception("Dataset reset failed")
            QMessageBox.critical(self, "Reset failed", str(exc))
            return

        self.label_tab.reload_after_reset()
        self.import_tab.refresh_summary()
        self.dataset_tab.refresh()
        self._refresh_status()
        self.tabs.setCurrentWidget(self.import_tab)
        QMessageBox.information(
            self,
            "Dataset reset",
            f"The dataset is now empty.\n\nA backup of the previous database was "
            f"saved to:\n{backup_path}",
        )

    def _score_for_triage(self) -> None:
        """Rank unreviewed files by a binary model's burst probability.

        This only changes the review *order*; no label is ever written from a
        prediction. The point is to spend attention on the files most likely to
        contain something worth marking.
        """
        from callisto_trainer.services.assist import find_latest_checkpoint

        # The unified model ranks by the same evidence Predict decides on, so it
        # is preferred: reviewing its top-ranked quiet files is the quickest way
        # to find, and then label, its false alarms.
        checkpoint = find_latest_checkpoint(self.settings.outputs_dir, "unified")
        if checkpoint is None:
            checkpoint = find_latest_checkpoint(self.settings.outputs_dir, "binary")
        if checkpoint is None:
            QMessageBox.information(
                self,
                "No trained model yet",
                "Triage ordering needs a trained unified or burst / no-burst "
                "checkpoint.\n\nLabel some files, export a snapshot from the Dataset "
                "tab, and train it first.",
            )
            return

        pending = [
            (record.id, record.path)
            for record in self.repository.files()
            if not record.is_reviewed and record.status != "error"
        ]
        if not pending:
            QMessageBox.information(self, "Nothing to score", "Every file has been reviewed.")
            return

        confirm = QMessageBox.question(
            self,
            "Score for triage?",
            f"Run {Path(checkpoint).name} over {len(pending):,} unreviewed file(s)?\n\n"
            "This reads every file and may take a while. Only the review order "
            "changes; no labels are written.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._run_triage(checkpoint, pending)

    def _run_triage(self, checkpoint: Path, pending: list[tuple[int, str]]) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from callisto_trainer.services.assist import score_files_for_triage

        dialog = QProgressDialog("Scoring files...", "Cancel", 0, len(pending), self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)

        metadata_rows = {
            record.id: {
                "station": record.station,
                "date": record.obs_date,
                "freq_min_mhz": record.freq_min_mhz,
                "freq_max_mhz": record.freq_max_mhz,
            }
            for record in self.repository.files()
        }

        def report(index: int, total: int, name: str) -> bool:
            dialog.setValue(index)
            dialog.setLabelText(f"Scoring {index:,} of {total:,}\n{name}")
            QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            if checkpoint.parent.parent.name.startswith("unified_"):
                from callisto_trainer.services.assist import score_files_with_unified

                scores = score_files_with_unified(
                    pending, checkpoint, self.settings.pipeline, progress=report
                )
            else:
                scores = score_files_for_triage(
                    pending, checkpoint, self.settings.pipeline, metadata_rows, progress=report
                )
        except Exception as exc:
            LOGGER.exception("Triage scoring failed")
            dialog.close()
            QMessageBox.warning(self, "Scoring failed", str(exc))
            return

        for file_id, probability in scores.items():
            self.repository.set_burst_probability(file_id, probability)
        dialog.close()

        self.label_tab.queue.order_by.setCurrentIndex(
            self.label_tab.queue.order_by.findData("probability")
        )
        self.tabs.setCurrentWidget(self.label_tab)
        self.status.showMessage(
            f"Scored {len(scores):,} file(s); queue sorted by burst probability.", 8000
        )

    def _inspect_file(self, path: str) -> None:
        """Jump from a misclassified result straight to that file in the Label tab."""
        file_id = self.repository.file_id_for_path(path)
        if file_id is None:
            self.status.showMessage(
                f"{path} is not in this dataset, so it cannot be opened for review.", 6000
            )
            return
        self.tabs.setCurrentWidget(self.label_tab)
        if not self.label_tab.queue.select_file(file_id):
            # The file exists but the active filter hides it; clear and retry.
            self.label_tab.queue.status_filter.setCurrentIndex(0)
            self.label_tab.queue.search.clear()
            self.label_tab.queue.select_file(file_id)

    def _on_imported(self) -> None:
        self.label_tab.refresh_queue(keep_selection=False)
        if self.label_tab.queue.current_file_id() is None:
            self.label_tab.queue.select_row(0)
        self._refresh_status()
        self.tabs.setCurrentWidget(self.label_tab)

    def _refresh_status(self) -> None:
        counts = self.repository.status_counts()
        total = sum(counts.values())
        reviewed = counts.get("labeled", 0) + counts.get("skipped", 0)
        types = self.repository.box_type_counts()
        breakdown = "  ".join(f"{name}: {count:,}" for name, count in sorted(types.items()))
        self.status_label.setText(
            f"{reviewed:,}/{total:,} reviewed   ·   {self.repository.total_boxes():,} bursts"
            + (f"   ·   {breakdown}" if breakdown else "")
        )

    def closeEvent(self, event) -> None:
        self.repository.set_state(
            STATE_GEOMETRY, bytes(self.saveGeometry().toHex()).decode("ascii")
        )
        self.label_tab.shutdown()
        self.train_tab.shutdown()
        self.evaluate_tab.shutdown()
        self.predict_tab.shutdown()
        event.accept()
