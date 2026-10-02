"""Predict tab: run trained models on new, unlabelled files.

The cascade is region-based (see :mod:`callisto_trainer.core.inference`): the
binary model judges the whole file, then bright regions are located and each crop
is typed. The UI is explicit that region *location* is a heuristic, so a result
is never presented as more than it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.crops import normalize_full_spectrum
from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes
from callisto_trainer.core.inference import (
    DEFAULT_MAX_REGIONS,
    DEFAULT_MIN_AREA,
    DEFAULT_REGION_THRESHOLD,
    FileResult,
    predict_paths,
    write_csv,
    write_json,
)
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.assist import find_latest_checkpoint
from callisto_trainer.services.importer import find_fits_files
from callisto_trainer.settings import AppSettings
from callisto_trainer.ui.spectrogram_view import SpectrogramView, color_for_type

LOGGER = get_logger(__name__)

ResultRole = Qt.ItemDataRole.UserRole + 1


def _index_of(combo: QComboBox, value: Any) -> int:
    """Index of the item whose data equals ``value``, or -1.

    ``QComboBox.findData`` compares arbitrary Python objects by identity, so a
    freshly constructed ``Path`` never matches an equal one already stored in the
    combo. Comparing by value is what callers actually mean here.
    """
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


class _PredictWorker(QThread):
    progress = Signal(int, int, str)
    finished_with = Signal(object)
    failed = Signal(str)

    def __init__(
        self, paths, settings: AppSettings, binary, type_ckpt, unified,
        threshold, min_area, max_regions, adaptive, burst_threshold=None,
        type_prior_strength=None,
    ):
        super().__init__()
        self.paths = paths
        self.settings = settings
        self.binary = binary
        self.type_ckpt = type_ckpt
        self.unified = unified
        self.threshold = threshold
        self.min_area = min_area
        self.max_regions = max_regions
        self.adaptive = adaptive
        self.burst_threshold = burst_threshold
        self.type_prior_strength = type_prior_strength
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        def report(index: int, total: int, name: str) -> bool:
            self.progress.emit(index, total, name)
            return not self._cancelled

        try:
            results = predict_paths(
                self.paths,
                self.settings.pipeline,
                binary_checkpoint=self.binary,
                type_checkpoint=self.type_ckpt,
                unified_checkpoint=self.unified,
                region_threshold=self.threshold,
                min_area=self.min_area,
                max_regions=self.max_regions,
                adaptive_threshold=self.adaptive,
                progress=report,
                burst_threshold=self.burst_threshold,
                type_prior_strength=self.type_prior_strength,
            )
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            self.failed.emit(repr(exc))
            return
        self.finished_with.emit(results)


class PredictTab(QWidget):
    """Batch inference with a spectrogram preview of what was found."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._paths: list[Path] = []
        self._results: list[FileResult] = []
        self._worker: _PredictWorker | None = None
        # Whether the operator has picked each model themselves; see refresh_models.
        self._user_chose: dict[str, bool] = {"unified": False, "binary": False, "type": False}
        # The selected unified model's calibration: its threshold and the finder
        # settings the threshold was tuned with.
        self._calibration: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_input_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_results_group())
        splitter.addWidget(self._build_preview_group())
        splitter.setSizes([760, 780])
        layout.addWidget(splitter, 1)

        self.refresh_models()

    # -- construction ------------------------------------------------------

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("1. Models")
        outer = QVBoxLayout(group)

        unified_row = QHBoxLayout()
        self.unified_model = QComboBox()
        self.unified_model.setMinimumWidth(320)
        self.unified_model.setToolTip(
            "A single model over regions. It decides for itself whether each candidate "
            "region is background, interference (RFI) or a burst, and which type, so no "
            "separate burst/no-burst gate is needed."
        )
        self.refresh_button = QPushButton("Refresh")
        self.browse_button = QPushButton("Browse...")
        unified_row.addWidget(QLabel("<b>Unified model:</b>"))
        unified_row.addWidget(self.unified_model, 1)
        unified_row.addWidget(self.refresh_button)
        unified_row.addWidget(self.browse_button)
        outer.addLayout(unified_row)

        self.legacy_label = QLabel("Or the separate models (used only when no unified model is selected):")
        self.legacy_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        outer.addWidget(self.legacy_label)

        row = QHBoxLayout()
        self.binary_model = QComboBox()
        self.binary_model.setMinimumWidth(220)
        self.type_model = QComboBox()
        self.type_model.setMinimumWidth(220)

        row.addWidget(QLabel("Burst / no burst:"))
        row.addWidget(self.binary_model, 1)
        row.addWidget(QLabel("Burst type:"))
        row.addWidget(self.type_model, 1)
        outer.addLayout(row)

        tuning = QHBoxLayout()
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.05, 0.95)
        self.sensitivity.setSingleStep(0.05)
        self.sensitivity.setValue(DEFAULT_REGION_THRESHOLD)
        self.sensitivity.setToolTip(
            "Brightness a region must reach, in normalized units "
            "(0.45 is about +3 dB above background). Lower finds more, including "
            "more interference."
        )
        self.min_area = QSpinBox()
        self.min_area.setRange(4, 5000)
        self.min_area.setValue(DEFAULT_MIN_AREA)
        self.min_area.setToolTip("Smallest region, in pixels, that counts as a candidate.")
        self.max_regions = QSpinBox()
        self.max_regions.setRange(1, 50)
        self.max_regions.setValue(DEFAULT_MAX_REGIONS)

        tuning.addWidget(QLabel("Region brightness:"))
        tuning.addWidget(self.sensitivity)
        tuning.addWidget(QLabel("Min area (px):"))
        tuning.addWidget(self.min_area)
        tuning.addWidget(QLabel("Max regions:"))
        tuning.addWidget(self.max_regions)

        self.adaptive = QCheckBox("Adapt to each file")
        self.adaptive.setChecked(True)
        self.adaptive.setToolTip(
            "Stations differ a lot in gain, so the same event can peak at 1.0 in one "
            "recording and 0.43 in another. With this on, the brightness setting acts "
            "as a ceiling and each file's own bright tail lowers it when needed, so "
            "faint recordings are not skipped entirely."
        )
        tuning.addWidget(self.adaptive)
        tuning.addStretch(1)
        outer.addLayout(tuning)

        decision = QHBoxLayout()
        self.burst_threshold = QDoubleSpinBox()
        self.burst_threshold.setRange(0.01, 0.99)
        self.burst_threshold.setSingleStep(0.01)
        self.burst_threshold.setDecimals(3)
        self.burst_threshold.setValue(0.5)
        self.burst_threshold.setToolTip(
            "A region is a burst when its burst evidence (1 - P(not a burst)) reaches "
            "this value.\n"
            "Training tunes it on the validation files to a false-alarm budget; "
            "raise it for fewer false alarms, lower it to catch fainter bursts."
        )
        self.type_strength = QDoubleSpinBox()
        self.type_strength.setRange(0.0, 1.0)
        self.type_strength.setSingleStep(0.25)
        self.type_strength.setDecimals(2)
        self.type_strength.setValue(0.0)
        self.type_strength.setToolTip(
            "How strongly the burst type follows how often each type really occurs.\n"
            "0 decides types as the (balanced) model was trained; 1 uses the real-world "
            "odds in full, so an ambiguous burst is called the commoner type. Training "
            "chooses it on the validation data. It never changes whether a region is a "
            "burst, only which type."
        )
        self.show_rfi = QCheckBox("Show RFI")
        self.show_rfi.setChecked(True)
        self.show_rfi.setToolTip(
            "Draw the interference found in each file. RFI is detected separately and "
            "never decides the verdict: a file with a burst is Burst even with RFI in it, "
            "a file with only RFI is No_Burst."
        )
        decision.addWidget(QLabel("Burst threshold (unified):"))
        decision.addWidget(self.burst_threshold)
        decision.addWidget(QLabel("Type frequency correction:"))
        decision.addWidget(self.type_strength)
        decision.addWidget(self.show_rfi)
        decision.addStretch(1)
        outer.addLayout(decision)

        self.threshold_note = QLabel("")
        self.threshold_note.setWordWrap(True)
        self.threshold_note.setStyleSheet("color: #8b949e; font-size: 11px;")
        outer.addWidget(self.threshold_note)
        for widget in (self.sensitivity, self.min_area, self.max_regions):
            widget.valueChanged.connect(self._update_threshold_note)
        self.adaptive.toggled.connect(self._update_threshold_note)
        self.burst_threshold.valueChanged.connect(self._update_threshold_note)
        self.type_strength.valueChanged.connect(self._update_threshold_note)

        note = QLabel(
            "The burst / no-burst model judges the whole file. Where a burst <i>is</i> "
            "comes from a brightness heuristic, not a trained detector; the burst-type "
            "model then classifies each located region, cropped exactly as it was during "
            "training. Treat located regions as candidates to check, not as detections."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b949e;")
        outer.addWidget(note)

        self.gate_warning = QLabel("")
        self.gate_warning.setWordWrap(True)
        self.gate_warning.setStyleSheet("color: #d4a72c;")
        outer.addWidget(self.gate_warning)

        self.refresh_button.clicked.connect(self.refresh_models)
        self.browse_button.clicked.connect(self._browse_checkpoint)
        self.unified_model.currentIndexChanged.connect(lambda: self._on_model_chosen("unified"))
        self.binary_model.currentIndexChanged.connect(lambda: self._on_model_chosen("binary"))
        self.type_model.currentIndexChanged.connect(lambda: self._on_model_chosen("type"))
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("2. Files to predict")
        row = QHBoxLayout(group)

        self.add_folder = QPushButton("Add folder...")
        self.add_files = QPushButton("Add files...")
        self.clear_files = QPushButton("Clear")
        self.run_button = QPushButton("Run prediction")
        self.run_button.setMinimumHeight(32)
        self.run_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)

        self.input_label = QLabel("No files selected.")
        self.input_label.setStyleSheet("color: #8b949e;")

        for widget in (self.add_folder, self.add_files, self.clear_files):
            row.addWidget(widget)
        row.addWidget(self.input_label, 1)
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)

        self.add_folder.clicked.connect(self._choose_folder)
        self.add_files.clicked.connect(self._choose_files)
        self.clear_files.clicked.connect(self._clear_inputs)
        self.run_button.clicked.connect(self._run)
        self.cancel_button.clicked.connect(self._cancel)
        return group

    def _build_results_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Results")
        group_layout = QVBoxLayout(group)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        group_layout.addWidget(self.progress)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "Prediction", "Probability", "Regions found", "Alert"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_row_changed)
        group_layout.addWidget(self.table)

        actions = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #8b949e;")
        self.clear_results_button = QPushButton("Clear results")
        self.clear_results_button.setEnabled(False)
        self.clear_results_button.setToolTip(
            "Empty the results table and the preview, ready for another run. "
            "Nothing on disk is affected; export first if you want to keep these."
        )
        self.export_csv = QPushButton("Export CSV")
        self.export_json = QPushButton("Export JSON")
        self.export_csv.setEnabled(False)
        self.export_json.setEnabled(False)
        actions.addWidget(self.summary, 1)
        actions.addWidget(self.clear_results_button)
        actions.addWidget(self.export_csv)
        actions.addWidget(self.export_json)
        group_layout.addLayout(actions)

        self.clear_results_button.clicked.connect(self.clear_results)
        self.export_csv.clicked.connect(lambda: self._export("csv"))
        self.export_json.clicked.connect(lambda: self._export("json"))
        layout.addWidget(group)
        return container

    def _build_preview_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Selected file")
        group_layout = QVBoxLayout(group)
        self.preview_title = QLabel("Select a result to view it.")
        self.preview_title.setWordWrap(True)
        group_layout.addWidget(self.preview_title)

        self.canvas = SpectrogramView()
        self.canvas.set_draw_mode(False)
        group_layout.addWidget(self.canvas, 1)

        self.region_detail = QLabel("")
        self.region_detail.setWordWrap(True)
        self.region_detail.setStyleSheet("color: #8b949e; font-size: 11px;")
        group_layout.addWidget(self.region_detail)
        layout.addWidget(group)
        return container

    # -- models ------------------------------------------------------------

    def refresh_models(self) -> None:
        """Repopulate the model lists, defaulting to the newest trained checkpoint.

        Auto-selection happens only until the operator picks something themselves;
        after that a refresh preserves their choice, including a deliberate
        "(none)".
        """
        for combo, task in (
            (self.unified_model, "unified"),
            (self.binary_model, "binary"),
            (self.type_model, "type"),
        ):
            current = combo.currentData()
            chosen_by_user = self._user_chose.get(task, False)

            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)", None)
            for path in self._available_checkpoints(task):
                combo.addItem(f"{path.parent.parent.name} / {path.name}", path)

            index = -1
            if chosen_by_user:
                # findData(None) would match "(none)", which is the right answer
                # only when the operator actually chose it.
                index = _index_of(combo, current)
            if index < 0:
                latest = find_latest_checkpoint(self.settings.outputs_dir, task)
                index = _index_of(combo, latest) if latest else 0
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
        # Signals were blocked above, so adopt a newly auto-selected unified
        # model's calibration here -- but only when the selection actually
        # changed, so a refresh never undoes the operator's own settings.
        if self.unified_model.currentData() != getattr(self, "_calibrated_for", None):
            self._calibrated_for = self.unified_model.currentData()
            self._apply_unified_calibration()
        self._update_run_state()

    def _on_model_chosen(self, task: str) -> None:
        self._user_chose[task] = True
        if task == "unified":
            self._apply_unified_calibration()
        self._update_run_state()

    def _apply_unified_calibration(self) -> None:
        """Adopt the selected unified model's calibrated threshold and finder settings.

        The threshold was tuned for one finder configuration: examining more or
        smaller regions per file raises the false-alarm rate it was tuned to. So
        the finder controls are set to match, and a note appears if they drift.
        """
        from callisto_trainer.core.inference import checkpoint_inference_settings

        path = self.unified_model.currentData()
        self._calibrated_for = path
        self._calibration = checkpoint_inference_settings(path) if path else {}
        threshold = self._calibration.get("burst_threshold")
        finder = self._calibration.get("region_finder") or {}
        priors = self._calibration.get("type_priors") or {}
        widgets = (self.sensitivity, self.min_area, self.max_regions, self.adaptive,
                   self.burst_threshold, self.type_strength)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if threshold is not None:
                self.burst_threshold.setValue(float(threshold))
            self.type_strength.setValue(float(priors.get("strength") or 0.0))
            self.type_strength.setEnabled(bool(priors.get("adjustment")))
            if finder:
                self.sensitivity.setValue(float(finder.get("threshold", self.sensitivity.value())))
                self.min_area.setValue(int(finder.get("min_area", self.min_area.value())))
                self.max_regions.setValue(int(finder.get("max_regions", self.max_regions.value())))
                self.adaptive.setChecked(bool(finder.get("adaptive", self.adaptive.isChecked())))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_threshold_note()

    def _update_threshold_note(self) -> None:
        """Say where the burst threshold came from and whether it still applies."""
        if self.unified_model.currentData() is None:
            self.threshold_note.setText("")
            return
        threshold = self._calibration.get("burst_threshold")
        record = self._calibration.get("calibration") or {}
        if threshold is None:
            self.threshold_note.setText(
                "This model was not calibrated, so the threshold is a guess. Retrain it "
                "to tune the threshold to a false-alarm budget on held-out files."
            )
            return

        parts = [
            f"Calibrated to {float(threshold):.3f}: on {record.get('quiet_files', '?')} "
            f"held-out quiet files it flagged "
            f"{100 * float(record.get('false_alarm_rate') or 0):.1f}% "
            f"(budget {100 * float(record.get('max_false_alarm_rate') or 0):.0f}%) and "
            f"found {100 * float(record.get('burst_recall') or 0):.1f}% of burst files."
        ]
        if abs(self.burst_threshold.value() - float(threshold)) > 1e-6:
            parts.append("You changed the threshold, so those rates no longer apply.")
        finder = self._calibration.get("region_finder") or {}
        if finder and (
            abs(self.sensitivity.value() - float(finder.get("threshold", 0))) > 1e-6
            or self.min_area.value() != int(finder.get("min_area", 0))
            or self.max_regions.value() != int(finder.get("max_regions", 0))
            or self.adaptive.isChecked() != bool(finder.get("adaptive", True))
        ):
            parts.append(
                "The region settings differ from those the threshold was tuned with, "
                "so the false-alarm rate will differ too."
            )
        priors = self._calibration.get("type_priors") or {}
        if priors.get("adjustment"):
            chosen = float(priors.get("strength") or 0.0)
            parts.append(
                f"Types follow the observed type frequencies at strength "
                f"{self.type_strength.value():.2f}"
                + (f" (chosen on validation: {chosen:.2f})."
                   if abs(self.type_strength.value() - chosen) > 1e-6 else ", chosen on validation.")
            )
        self.threshold_note.setText(" ".join(parts))

    def _available_checkpoints(self, task: str) -> list[Path]:
        root = Path(self.settings.outputs_dir)
        if not root.exists():
            return []
        found = []
        for run in sorted(root.iterdir(), reverse=True):
            if not run.is_dir() or not run.name.startswith(f"{task}_"):
                continue
            for name in ("best.pt", "last.pt"):
                candidate = run / "checkpoints" / name
                if candidate.exists():
                    found.append(candidate)
        return found

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a checkpoint", str(self.settings.outputs_dir), "Checkpoints (*.pt)"
        )
        if not path:
            return
        target = self.type_model if "type" in Path(path).parts[-3:][0].lower() else self.binary_model
        target.addItem(Path(path).name, Path(path))
        target.setCurrentIndex(target.count() - 1)
        self._update_run_state()

    # -- inputs ------------------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of FITS files")
        if folder:
            self._add_paths(find_fits_files([Path(folder)]))

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS files", "", "FITS (*.fit.gz *.fits.gz *.fit *.fits)"
        )
        if files:
            self._add_paths([Path(f) for f in files])

    def _add_paths(self, paths: list[Path]) -> None:
        known = set(self._paths)
        self._paths.extend(p for p in paths if p not in known)
        self.input_label.setText(f"{len(self._paths):,} file(s) selected.")
        self._update_run_state()

    def _clear_inputs(self) -> None:
        self._paths.clear()
        self.input_label.setText("No files selected.")
        self._update_run_state()


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
        self._update_run_state()

    def _update_run_state(self) -> None:
        has_unified = self.unified_model.currentData() is not None
        has_binary = self.binary_model.currentData() is not None
        has_type = self.type_model.currentData() is not None
        self.run_button.setEnabled(
            bool(self._paths)
            and (has_unified or has_binary or has_type)
            and self._worker is None
        )

        # The separate models are only consulted when no unified model is chosen.
        for widget in (self.legacy_label, self.binary_model, self.type_model):
            widget.setEnabled(not has_unified)
        for widget in (self.burst_threshold, self.show_rfi):
            widget.setEnabled(has_unified)
        self.type_strength.setEnabled(
            has_unified and bool((self._calibration.get("type_priors") or {}).get("adjustment"))
        )
        if has_unified:
            self.gate_warning.setText("")
            return

        # Measured on this archive, quiet files yield as many bright regions as
        # burst files. Without the binary gate, typed regions say nothing about
        # whether a burst occurred, and that must not be discoverable only by
        # reading the source.
        if has_type and not has_binary:
            self.gate_warning.setText(
                "⚠ No burst / no-burst model selected. Regions will be typed for every "
                "file, including quiet ones — bright regions are just as common there, "
                "so the results will not tell you whether a burst occurred. Select a "
                "burst / no-burst model to gate the run."
            )
        elif has_binary and not has_type:
            self.gate_warning.setText(
                "No burst-type model selected: files will be classified burst / no burst "
                "only, with no regions located."
            )
        else:
            self.gate_warning.setText("")

    # -- running -----------------------------------------------------------

    def _run(self) -> None:
        if self._worker is not None or not self._paths:
            return
        unified = self.unified_model.currentData()
        binary = None if unified else self.binary_model.currentData()
        type_ckpt = None if unified else self.type_model.currentData()
        if unified is None and binary is None and type_ckpt is None:
            QMessageBox.information(
                self, "No model selected", "Choose at least one trained model to run."
            )
            return

        # Clear first: a new run must not leave the previous batch's rows or a
        # stale preview on screen while it works.
        self.clear_results()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self._paths))
        self.progress.setValue(0)

        self._worker = _PredictWorker(
            list(self._paths), self.settings, binary, type_ckpt, unified,
            self.sensitivity.value(), self.min_area.value(), self.max_regions.value(),
            self.adaptive.isChecked(),
            burst_threshold=self.burst_threshold.value() if unified else None,
            type_prior_strength=self.type_strength.value() if unified else None,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_with.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._release_worker)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.summary.setText("Cancelling...")

    def _on_progress(self, index: int, total: int, name: str) -> None:
        self.progress.setValue(index)
        self.summary.setText(f"Predicting {index:,} of {total:,}: {name}")

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.summary.setText(f"Prediction failed: {message}")
        self._update_run_state()

    def _on_finished(self, results: list[FileResult]) -> None:
        self.progress.setVisible(False)
        self.cancel_button.setEnabled(False)
        self._results = results
        self._populate(results)
        for button in (self.clear_results_button, self.export_csv, self.export_json):
            button.setEnabled(bool(results))
        self._update_run_state()

    # -- results -----------------------------------------------------------

    def _populate(self, results: list[FileResult]) -> None:
        self.table.setRowCount(len(results))
        bursts = 0
        errors = 0
        rfi = 0
        type_counts: dict[str, int] = {}

        for row, result in enumerate(results):
            if result.error:
                errors += 1
            if result.is_burst:
                bursts += 1
            rfi += len(result.rfi_sources)
            for region in result.regions:
                if region.burst_type:
                    type_counts[region.burst_type] = type_counts.get(region.burst_type, 0) + 1

            probability = (
                f"{result.burst_probability:.4f}" if result.burst_probability is not None else "-"
            )
            cells = [
                result.file_name,
                result.error and "error" or (result.predicted_label or "-"),
                probability,
                result.region_summary,
                result.alert_level or "-",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                item.setData(ResultRole, row)
                if result.error:
                    item.setForeground(QColor("#f85149"))
                    item.setToolTip(result.error)
                elif column == 1 and result.is_burst:
                    item.setForeground(QColor("#3fb950"))
                self.table.setItem(row, column, item)

        parts = [f"{len(results):,} file(s)"]
        if self.binary_model.currentData() is not None or self.unified_model.currentData():
            parts.append(f"{bursts:,} burst")
        if type_counts:
            parts.append(
                "regions: " + "  ".join(f"{k} {v:,}" for k, v in sorted(type_counts.items()))
            )
        if rfi:
            parts.append(f"{rfi:,} interference source(s) found")
        if errors:
            parts.append(f"{errors:,} unreadable")
        self.summary.setText("   ·   ".join(parts))

    def clear_results(self) -> None:
        """Empty the results table and preview, ready for another run.

        Only the view is cleared: the selected models, the tuning settings and any
        exported file are all left alone, since the usual reason to clear is to
        run the same configuration over a different batch.
        """
        self._results = []
        # Block signals so emptying the table does not fire a row-changed that
        # would try to render a result that no longer exists.
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.blockSignals(False)

        self.canvas.clear_boxes()
        self.canvas.clear()
        self.preview_title.setText("Select a result to view it.")
        self.region_detail.setText("")

        self.summary.setText("")
        self.progress.setVisible(False)
        for button in (self.clear_results_button, self.export_csv, self.export_json):
            button.setEnabled(False)

    def _on_row_changed(self, row: int, _column: int, _prev_row: int, _prev_col: int) -> None:
        if not (0 <= row < len(self._results)):
            return
        self._show_result(self._results[row])

    def _show_result(self, result: FileResult) -> None:
        if result.error:
            self.canvas.clear()
            self.preview_title.setText(f"{result.file_name} — could not be read: {result.error}")
            self.region_detail.setText("")
            return

        try:
            spectrum, metadata = read_fits_spectrum_and_axes(result.file_path)
            normalized = normalize_full_spectrum(spectrum, self.settings.pipeline)
        except Exception as exc:
            self.canvas.clear()
            self.preview_title.setText(f"{result.file_name} — {exc}")
            return

        axes = SpectrumAxes.from_metadata(metadata)
        self.canvas.set_spectrum(
            normalized,
            axes,
            rfi_channels=metadata.get("rfi_channels_mhz"),
            raw=spectrum,
            raw_levels=(float(np.nanpercentile(spectrum, 1)), float(np.nanpercentile(spectrum, 99.5))),
        )
        for index, region in enumerate(result.regions):
            self.canvas.add_box(
                index,
                region.row0, region.row1, region.col0, region.col1,
                region.burst_type or "Other",
                confirmed=False,  # dashed: these are candidates, not labels
            )
        if self.show_rfi.isChecked():
            # Interference, found separately from the bursts: drawn in the RFI
            # colour beside them, whatever the file's verdict.
            for offset, region in enumerate(result.rfi_regions):
                self.canvas.add_box(
                    len(result.regions) + offset,
                    region.row0, region.row1, region.col0, region.col1,
                    "RFI",
                    confirmed=False,
                )

        headline = result.predicted_label or "-"
        if result.burst_probability is not None:
            headline += f"  (p={result.burst_probability:.4f}, threshold {result.decision_threshold:.2f})"
        self.preview_title.setText(f"<b>{result.file_name}</b> — {headline}")

        rfi_lines = []
        for source in result.rfi_sources:
            span = ""
            if source.freq_lo_mhz is not None:
                span = f" · {source.freq_lo_mhz:.1f}-{source.freq_hi_mhz:.1f} MHz"
            segments = f" · {source.segments} segments" if source.segments > 1 else ""
            rfi_lines.append(
                f'<span style="color:{color_for_type("RFI")}">■</span> RFI ({source.kind})'
                f"{span}{segments}"
            )
        if not result.regions:
            self.region_detail.setText(
                "<br>".join(rfi_lines)
                if rfi_lines
                else (
                    "No candidate region was located above the brightness setting."
                    if result.is_burst
                    else ""
                )
            )
            return

        lines = []
        for index, region in enumerate(result.regions, 1):
            colour = color_for_type(region.burst_type or "Other")
            span = ""
            if region.freq_lo_mhz is not None:
                span = f" · {region.freq_lo_mhz:.1f}-{region.freq_hi_mhz:.1f} MHz"
            confidence = f" · {region.type_confidence:.2f}" if region.type_confidence else ""
            lines.append(
                f'<span style="color:{colour}">■</span> {index}. '
                f"{region.burst_type or 'untyped'}{confidence}{span} · {region.area:,} px"
            )
        self.region_detail.setText(
            "<br>".join(lines + rfi_lines)
            + "<br><i>Regions are heuristic candidates, drawn dashed.</i>"
        )

    # -- export ------------------------------------------------------------

    def _export(self, kind: str) -> None:
        if not self._results:
            return
        default = str(Path(self.settings.outputs_dir) / f"predictions.{kind}")
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save predictions as {kind.upper()}", default, f"{kind.upper()} (*.{kind})"
        )
        if not path:
            return
        writer = write_csv if kind == "csv" else write_json
        writer(self._results, path)
        self.summary.setText(f"Saved {len(self._results):,} prediction(s) to {path}")

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
