"""Train tab: pick a snapshot, tune the run, watch the curves live."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyqtgraph as pg
import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.train_runner import TrainingJob, TrainingRunner
from callisto_trainer.settings import AppSettings
from callisto_trainer.store.export import (
    SNAPSHOT_KINDS,
    list_snapshots,
    read_snapshot_info,
    task_for_kind,
)

LOGGER = get_logger(__name__)

# A split with fewer than this many samples of a class produces meaningless
# metrics, and an empty train split makes the class-weight computation raise.
MIN_SAMPLES_PER_SPLIT = 2

# The largest batch (regions of three views each) the deeper backbones train with
# inside an 8 GB GPU under mixed precision, measured on an RTX 5060. Above it
# Windows does not fail with out-of-memory: it quietly spills into system memory
# and training runs several times slower -- ConvNeXt-Tiny at 32 started at 7.5 GB,
# crept over after nine epochs and went from 3 to 13 minutes an epoch. Choosing
# one of these backbones lowers the batch size to fit; it can be raised again on
# a larger card.
MAX_BATCH_8GB = {"resnet50": 32, "convnext_tiny": 24}


class TrainTab(QWidget):
    """Configure and run training, with live loss and score curves."""

    training_finished = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.runner = TrainingRunner(settings.project_root, self)
        self._history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "score": [], "epoch": []
        }
        self._epoch_records: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(self._build_snapshot_group())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_settings_group())
        splitter.addWidget(self._build_monitor_group())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1100])
        layout.addWidget(splitter, 1)

        self.runner.started.connect(self._on_started)
        self.runner.progress.connect(self._on_progress)
        self.runner.output.connect(self._append_log)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(lambda message: self._append_log(f"ERROR: {message}"))

        self.refresh_snapshots()

    # -- construction ------------------------------------------------------

    def _build_snapshot_group(self) -> QGroupBox:
        group = QGroupBox("1. Choose a dataset snapshot")
        row = QHBoxLayout(group)

        self.kind = QComboBox()
        for directory, label, _task in SNAPSHOT_KINDS:
            self.kind.addItem(label, directory)
        self.snapshot = QComboBox()
        self.snapshot.setMinimumWidth(240)
        self.refresh_button = QPushButton("Refresh")

        row.addWidget(QLabel("Model:"))
        row.addWidget(self.kind)
        row.addWidget(QLabel("Snapshot:"))
        row.addWidget(self.snapshot, 1)
        row.addWidget(self.refresh_button)

        self.kind.currentIndexChanged.connect(self.refresh_snapshots)
        self.snapshot.currentIndexChanged.connect(self._on_snapshot_changed)
        self.refresh_button.clicked.connect(self.refresh_snapshots)
        return group

    def _build_settings_group(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("2. Training settings")
        form = QFormLayout(group)

        self.backbone = QComboBox()
        self.backbone.addItems(
            ["resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_b0",
             "mobilenet_v3_small", "simple_cnn"]
        )
        self.backbone.activated.connect(self._on_backbone_chosen)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(5)
        self.learning_rate.setRange(0.00001, 1.0)
        self.learning_rate.setSingleStep(0.0001)
        self.patience = QSpinBox()
        # 0 is the default and must be reachable: it means "run every epoch".
        self.patience.setRange(0, 500)
        self.patience.setSpecialValueText("Off - run all epochs")
        self.patience.setToolTip(
            "Stop early when the validation score has not improved for this many "
            "epochs.\n\n"
            "Off (0) runs the full epoch count you set above. The saved model is "
            "chosen by validation score either way, so training past a plateau "
            "cannot make the result worse -- it only costs time."
        )
        self.pretrained = QComboBox()
        self.pretrained.addItem("ImageNet pretrained", True)
        self.pretrained.addItem("Random init", False)
        self.augment = QComboBox()
        self.augment.addItem("On", True)
        self.augment.addItem("Off", False)
        # Unified models only: how far the file's station and date may move a
        # decision (see models/physics_model.py, StationDateCorrection).
        self.station_date = QComboBox()
        self.station_date.addItem("Off - image and region features only", 0.0)
        self.station_date.addItem("Light - log-odds shift up to 0.5", 0.5)
        self.station_date.addItem("Moderate - log-odds shift up to 1", 1.0)
        self.station_date.addItem("Strong - log-odds shift up to 2", 2.0)
        self.station_date.setToolTip(
            "Lets the model use the file's station and observation month/year as a "
            "bounded correction on top of what the spectrogram shows.\n\n"
            "The cap is the most it can move the odds of any decision: at 1, a region "
            "the image puts at 50% can end up between 27% and 73%, and one at 95% no "
            "lower than 87%. Labelled burst rates per station mostly reflect which "
            "files were picked for labelling, which is why the influence is capped.\n\n"
            "Unified snapshots only."
        )

        form.addRow("Backbone", self.backbone)
        form.addRow("Epochs", self.epochs)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Learning rate", self.learning_rate)
        form.addRow("Early-stop patience", self.patience)
        form.addRow("Weights", self.pretrained)
        form.addRow("Augmentation", self.augment)
        form.addRow("Station + date", self.station_date)
        outer.addWidget(group)

        self.dataset_summary = QLabel("")
        self.dataset_summary.setWordWrap(True)
        self.dataset_summary.setStyleSheet("color: #8b949e;")
        outer.addWidget(self.dataset_summary)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #d4a72c;")
        outer.addWidget(self.warning)

        row = QHBoxLayout()
        self.start_button = QPushButton("Start training")
        self.start_button.setMinimumHeight(36)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        row.addWidget(self.start_button, 1)
        row.addWidget(self.stop_button)
        outer.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)
        outer.addStretch(1)

        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self.runner.stop)
        return container

    def _build_monitor_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.loss_plot = pg.PlotWidget(title="Loss")
        self.loss_plot.addLegend()
        self.loss_plot.setLabel("bottom", "Epoch")
        self.train_curve = self.loss_plot.plot(pen=pg.mkPen("#58a6ff", width=2), name="train")
        self.val_curve = self.loss_plot.plot(pen=pg.mkPen("#ff6b35", width=2), name="validation")

        self.score_plot = pg.PlotWidget(title="Validation score")
        self.score_plot.setLabel("bottom", "Epoch")
        self.score_curve = self.score_plot.plot(pen=pg.mkPen("#3fb950", width=2))

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")

        splitter = QSplitter(Qt.Orientation.Vertical)
        plots = QWidget()
        plot_layout = QHBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.loss_plot)
        plot_layout.addWidget(self.score_plot)
        splitter.addWidget(plots)
        splitter.addWidget(self.log)
        splitter.setSizes([420, 260])
        layout.addWidget(splitter)
        return container

    # -- snapshots ---------------------------------------------------------

    def refresh_snapshots(self) -> None:
        kind = self.kind.currentData()
        current = self.snapshot.currentData()
        self.snapshot.blockSignals(True)
        self.snapshot.clear()
        for directory in list_snapshots(self.settings.datasets_dir, kind):
            info = read_snapshot_info(directory)
            self.snapshot.addItem(f"{directory.name}  ({info.get('samples', 0):,} samples)", directory)
        index = self.snapshot.findData(current)
        self.snapshot.setCurrentIndex(max(0, index))
        self.snapshot.blockSignals(False)
        self._on_snapshot_changed()

    def current_snapshot(self) -> Path | None:
        return self.snapshot.currentData()

    def _on_backbone_chosen(self) -> None:
        """Lower the batch size to one the chosen backbone trains with on 8 GB."""
        limit = MAX_BATCH_8GB.get(self.backbone.currentText())
        if limit is not None and self.batch_size.value() > limit:
            self.batch_size.setValue(limit)

    def _on_snapshot_changed(self) -> None:
        directory = self.current_snapshot()
        if directory is None:
            self.dataset_summary.setText("No snapshot yet. Export one from the Dataset tab.")
            self.warning.setText("")
            self.start_button.setEnabled(False)
            return

        info = read_snapshot_info(directory)
        classes = "   ".join(
            f"{name}: {count:,}" for name, count in sorted(info.get("class_counts", {}).items())
        )
        splits = "   ".join(
            f"{name}: {count:,}" for name, count in sorted(info.get("split_counts", {}).items())
        )
        self.dataset_summary.setText(
            f"{info.get('samples', 0):,} samples\n{classes}\n{splits}"
        )
        self._load_config_defaults(directory)
        problems = self._blocking_problems(info)
        self.warning.setText("\n".join(problems))
        self.start_button.setEnabled(not problems and not self.runner.is_running)

    def _blocking_problems(self, info: dict[str, Any]) -> list[str]:
        """Catch the cases that would otherwise crash or mislead mid-run."""
        problems: list[str] = []
        if not info.get("samples"):
            problems.append("This snapshot is empty.")
        for class_name, splits in sorted(info.get("class_split_counts", {}).items()):
            for split in ("train", "val", "test"):
                count = splits.get(split, 0)
                if count < MIN_SAMPLES_PER_SPLIT:
                    problems.append(
                        f"'{class_name}' has only {count} sample(s) in {split}. "
                        "Label more of this class, then export again."
                    )
        if info.get("event_leakage"):
            problems.append(
                f"{info['event_leakage']} event(s) span multiple splits; scores would be inflated."
            )
        return problems

    def _load_config_defaults(self, directory: Path) -> None:
        config_path = directory / "config.yaml"
        if not config_path.exists():
            return
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        model = config.get("model", {})
        training = config.get("training", {})
        index = self.backbone.findText(str(model.get("name", "resnet18")))
        self.backbone.setCurrentIndex(max(0, index))
        self.epochs.setValue(int(training.get("epochs", 40)))
        self.batch_size.setValue(int(training.get("batch_size", 64)))
        self.learning_rate.setValue(float(training.get("learning_rate", 0.0003)))
        self.patience.setValue(int(training.get("patience", 0)))
        self.pretrained.setCurrentIndex(0 if model.get("pretrained", True) else 1)
        self.augment.setCurrentIndex(
            0 if config.get("augmentation", {}).get("enabled", True) else 1
        )
        station_date = model.get("station_date") or {}
        cap = float(station_date.get("cap", 1.0)) if station_date.get("enabled") else 0.0
        index = self.station_date.findData(cap)
        self.station_date.setCurrentIndex(index if index >= 0 else 0)
        self.station_date.setEnabled(bool(model.get("use_physics", False)))

    def _write_run_config(self, directory: Path) -> Path:
        """Persist the edited settings beside the snapshot so the run is reproducible."""
        config_path = directory / "config.yaml"
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        config["model"]["name"] = self.backbone.currentText()
        config["model"]["pretrained"] = bool(self.pretrained.currentData())
        config["training"]["epochs"] = self.epochs.value()
        config["training"]["batch_size"] = self.batch_size.value()
        config["training"]["learning_rate"] = self.learning_rate.value()
        config["training"]["patience"] = self.patience.value()
        config["augmentation"]["enabled"] = bool(self.augment.currentData())
        if config["model"].get("use_physics", False):
            cap = float(self.station_date.currentData())
            section = dict(config["model"].get("station_date") or {})
            section["enabled"] = cap > 0
            if cap > 0:
                section["cap"] = cap
            config["model"]["station_date"] = section

        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
        return config_path

    # -- running -----------------------------------------------------------

    def _start(self) -> None:
        directory = self.current_snapshot()
        if directory is None or self.runner.is_running:
            return

        info = read_snapshot_info(directory)
        problems = self._blocking_problems(info)
        if problems:
            QMessageBox.warning(self, "Cannot train yet", "\n\n".join(problems))
            return

        config_path = self._write_run_config(directory)
        self._reset_history()
        self.log.clear()
        self._append_log(f"Config: {config_path}")

        self.runner.start(TrainingJob.train(task_for_kind(self.kind.currentData()), config_path))

    def _reset_history(self) -> None:
        self._history = {"train_loss": [], "val_loss": [], "score": [], "epoch": []}
        self._epoch_records = []

    def _on_started(self, job: TrainingJob) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, self.epochs.value())
        self.progress.setValue(0)
        self._append_log(f"Training started ({job.task}).")

    def _on_progress(self, record: dict) -> None:
        event = record.get("event")
        if event == "start":
            self.progress.setRange(0, int(record.get("total_epochs", self.epochs.value())))
            self._append_log(
                f"Device {record.get('device')} · train {record.get('train_size'):,} "
                f"/ val {record.get('val_size'):,} / test {record.get('test_size'):,}"
            )
            return

        if event == "epoch":
            epoch = int(record.get("epoch", 0))
            self._epoch_records.append(record)
            self._history["epoch"].append(epoch)
            self._history["train_loss"].append(float(record.get("train_loss", 0.0)))
            self._history["val_loss"].append(float(record.get("val_loss", 0.0)))
            self._history["score"].append(float(record.get("score", 0.0)))

            self.train_curve.setData(self._history["epoch"], self._history["train_loss"])
            self.val_curve.setData(self._history["epoch"], self._history["val_loss"])
            self.score_curve.setData(self._history["epoch"], self._history["score"])
            self.progress.setValue(epoch)

            if record.get("task") == "unified" and record.get("val_detection_ap") is not None:
                detail = (
                    f"detection_ap={record.get('val_detection_ap') or 0:.4f} "
                    f"type_f1={record.get('val_type_macro_f1') or 0:.4f} "
                    f"macro_f1={record.get('val_macro_f1', 0):.4f}"
                )
            elif record.get("task") in ("type", "unified"):
                detail = (
                    f"acc={record.get('val_accuracy', 0):.4f} "
                    f"macro_f1={record.get('val_macro_f1', 0):.4f}"
                )
            else:
                detail = (
                    f"f1={record.get('val_f1', 0):.4f} "
                    f"pr_auc={record.get('val_pr_auc', 0):.4f} "
                    f"thr={record.get('decision_threshold', 0):.3f}"
                )
            best = "  *best*" if record.get("is_best") else ""
            self._append_log(
                f"Epoch {epoch:3d}  train_loss={record.get('train_loss', 0):.4f}  "
                f"val_loss={record.get('val_loss', 0):.4f}  {detail}{best}"
            )
            return

        if event == "finished":
            # Say how much of the schedule actually ran. A run that ends short of
            # the configured epochs used to look identical to one that finished,
            # which made early stopping impossible to notice from the log.
            ran = record.get("epochs_run")
            planned = record.get("total_epochs")
            if ran is not None and planned:
                note = (
                    f"stopped early at {ran} of {planned}"
                    if record.get("stopped_early")
                    else f"all {planned} epochs completed"
                )
                self._append_log(f"Training finished: {note}.")
            self._append_log(
                f"Best epoch {record.get('best_epoch')} "
                f"(score {float(record.get('best_score', 0)):.4f})"
            )
            self._append_log(f"Checkpoint: {record.get('best_alias')}")
            self._warn_on_degenerate_threshold()
            if record.get("task") == "unified":
                self._append_log(
                    "Calibrating the burst threshold on the validation files "
                    "(the way Predict runs the model)..."
                )
            return

        if event == "calibrating":
            total = int(record.get("total", 0))
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(int(record.get("index", 0)) + 1)
            return

        if event == "calibrated":
            self._show_calibration(record)

        if event == "type_priors":
            self._show_type_priors(record)

    def _show_type_priors(self, record: dict) -> None:
        """Report how the burst types were corrected toward their real frequencies."""
        if record.get("error"):
            self._append_log(
                f"Type-frequency calibration failed ({record['error']}); types are decided "
                "as trained."
            )
            return

        def shares(values: dict) -> str:
            return ", ".join(f"{name} {100 * float(v):.1f}%" for name, v in values.items())

        self._append_log(
            f"Burst types: the model was trained on {shares(record.get('training_shares', {}))}; "
            f"they really occur as {shares(record.get('class_shares', {}))}. Types are "
            f"corrected toward the real frequencies at strength "
            f"{float(record.get('strength', 0)):.2f} (chosen on the validation regions)."
        )

    def _show_calibration(self, record: dict) -> None:
        """Report the file-level operating point the unified model was tuned to."""
        if record.get("error"):
            message = (
                f"Calibration failed ({record['error']}); the model decides by argmax. "
                "Evaluate it to see its file-level false-alarm rate."
            )
            self._append_log(message)
            self.warning.setText(message)
            return

        def percent(value) -> str:
            return "-" if value is None else f"{100 * float(value):.1f}%"

        self._append_log(
            f"Calibrated burst threshold {float(record.get('threshold', 0)):.3f} on "
            f"{record.get('quiet_files', 0)} quiet and {record.get('burst_files', 0)} burst "
            f"validation files: {percent(record.get('false_alarm_rate'))} of quiet files "
            f"flagged (budget {percent(record.get('max_false_alarm_rate'))}), "
            f"{percent(record.get('burst_recall'))} of burst files found."
        )
        if record.get("note"):
            self._append_log(f"NOTE: {record['note']}")
        if not record.get("met_budget", True):
            self.warning.setText(
                "The false-alarm budget could not be met on the validation files: "
                + (record.get("note") or "some quiet files score as high as bursts.")
            )

    def _warn_on_degenerate_threshold(self) -> None:
        """Flag a tuned decision threshold that sits far from 0.5.

        Legitimate, but it means the model's probabilities are not calibrated:
        it pushes one class to the extreme of the range. Worth saying, because
        such a model is brittle and its scores cannot be read as confidences.
        """
        if self.kind.currentData() != "binary":
            return
        threshold = next(
            (
                record.get("decision_threshold")
                for record in reversed(self._epoch_records)
                if record.get("decision_threshold") is not None
            ),
            None,
        )
        if threshold is None:
            return

        from callisto_trainer.core.metrics import threshold_is_extreme

        if threshold_is_extreme(float(threshold)):
            message = (
                f"NOTE: the tuned decision threshold is {float(threshold):g}, far from 0.5. "
                "The model separates the classes but its probabilities are not calibrated, "
                "so they cannot be read as confidences and small data changes can move many "
                "predictions. Usually a sign the minority class needs more examples."
            )
            self._append_log(message)
            self.warning.setText(message)

    def _on_finished(self, exit_code: int, job: TrainingJob | None) -> None:
        self.stop_button.setEnabled(False)
        self.progress.setVisible(False)
        self._on_snapshot_changed()
        if exit_code == 0:
            self._append_log("Training finished.")
        else:
            self._append_log(
                f"Training exited with code {exit_code}. "
                "The last checkpoint, if any, is still on disk."
            )
        self.training_finished.emit()

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def shutdown(self) -> None:
        if self.runner.is_running:
            self.runner.stop()
