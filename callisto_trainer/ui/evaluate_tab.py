"""Evaluate tab: run a checkpoint on the held-out split and read the results.

The misclassified list is clickable and jumps straight back to that file in the
Label tab. Evaluation is most useful when it feeds re-labelling: the fastest way
to improve a model early on is to look at what it got wrong and check whether the
label was right in the first place.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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

FilePathRole = Qt.ItemDataRole.UserRole + 1


class EvaluateTab(QWidget):
    """Run evaluation and show metrics, confusion matrix and mistakes."""

    inspect_file_requested = Signal(str)  # absolute source path

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.runner = TrainingRunner(settings.project_root, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._build_selection_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_results_group())
        splitter.addWidget(self._build_mistakes_group())
        splitter.setSizes([760, 620])
        layout.addWidget(splitter, 1)

        self.runner.output.connect(self._append_log)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(lambda message: self._append_log(f"ERROR: {message}"))

        self.refresh_snapshots()

    # -- construction ------------------------------------------------------

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Choose what to evaluate")
        row = QHBoxLayout(group)

        self.kind = QComboBox()
        for directory, label, _task in SNAPSHOT_KINDS:
            self.kind.addItem(label, directory)
        self.snapshot = QComboBox()
        self.snapshot.setMinimumWidth(220)
        self.checkpoint = QComboBox()
        self.checkpoint.setMinimumWidth(200)
        self.split = QComboBox()
        self.split.addItems(["test", "val", "train"])

        self.run_button = QPushButton("Run evaluation")
        self.run_button.setMinimumHeight(30)
        self.export_button = QPushButton("Export model...")
        self.export_button.setToolTip(
            "Write a self-contained bundle: weights, the full input contract, a "
            "TorchScript graph and a standalone predict script."
        )
        self.refresh_button = QPushButton("Refresh")

        for label, widget in (
            ("Model:", self.kind),
            ("Snapshot:", self.snapshot),
            ("Checkpoint:", self.checkpoint),
            ("Split:", self.split),
        ):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addWidget(self.run_button)
        row.addWidget(self.export_button)
        row.addWidget(self.refresh_button)
        row.addStretch(1)

        self.kind.currentIndexChanged.connect(self.refresh_snapshots)
        self.snapshot.currentIndexChanged.connect(self._refresh_checkpoints)
        self.refresh_button.clicked.connect(self.refresh_snapshots)
        self.run_button.clicked.connect(self._run)
        self.export_button.clicked.connect(self._export_model)
        return group

    def _export_model(self) -> None:
        """Write the selected checkpoint out as a portable bundle."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from callisto_trainer.services.model_export import export_model_bundle

        checkpoint = self.checkpoint.currentData()
        snapshot = self.snapshot.currentData()
        if checkpoint is None:
            QMessageBox.information(
                self, "No checkpoint", "Train a model first, then select its checkpoint."
            )
            return

        destination = QFileDialog.getExistingDirectory(
            self, "Where should the model bundle go?", str(self.settings.outputs_dir)
        )
        if not destination:
            return

        task = task_for_kind(self.kind.currentData())
        try:
            bundle = export_model_bundle(
                checkpoint,
                task,
                destination,
                snapshot_dir=snapshot,
                include_torchscript=True,
                make_zip=True,
            )
        except Exception as exc:
            LOGGER.exception("Model export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        message = [
            f"Bundle written to:\n{bundle.directory}",
            "",
            "Contents: " + ", ".join(sorted(bundle.files)),
        ]
        if bundle.zip_path:
            message.append(f"\nZipped: {bundle.zip_path.name}")
        if bundle.torchscript_error:
            message.append(
                f"\nTorchScript could not be produced ({bundle.torchscript_error}). "
                "The bundle is still usable through weights.pt and this project."
            )
        else:
            message.append(
                "\nRun it anywhere with:\n  python predict.py path/to/file.fit.gz"
            )
        self._append_log(f"Exported model bundle to {bundle.directory}")
        QMessageBox.information(self, "Model exported", "\n".join(message))

    def _build_results_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        headline_group = QGroupBox("Headline metrics")
        headline_layout = QVBoxLayout(headline_group)
        self.headline = QLabel("Run an evaluation to see results.")
        self.headline.setStyleSheet("font-size: 14px;")
        self.headline.setWordWrap(True)
        headline_layout.addWidget(self.headline)

        self.metric_warning = QLabel("")
        self.metric_warning.setWordWrap(True)
        self.metric_warning.setStyleSheet("color: #d4a72c; font-size: 12px;")
        headline_layout.addWidget(self.metric_warning)
        layout.addWidget(headline_group)

        matrix_group = QGroupBox("Confusion matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix = QTableWidget(0, 0)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.matrix.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        matrix_layout.addWidget(self.matrix)
        layout.addWidget(matrix_group, 1)

        per_class_group = QGroupBox("Per class")
        per_class_layout = QVBoxLayout(per_class_group)
        self.per_class = QTableWidget(0, 5)
        self.per_class.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1", "Support"]
        )
        self.per_class.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.per_class.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        per_class_layout.addWidget(self.per_class)
        layout.addWidget(per_class_group, 1)
        return container

    def _build_mistakes_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Misclassified — double-click to inspect in the Label tab")
        group_layout = QVBoxLayout(group)
        self.mistakes = QListWidget()
        self.mistakes.itemDoubleClicked.connect(self._on_mistake_activated)
        group_layout.addWidget(self.mistakes)
        layout.addWidget(group, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(180)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.log)
        return container

    # -- selection ---------------------------------------------------------

    def refresh_snapshots(self) -> None:
        kind = self.kind.currentData()
        current = self.snapshot.currentData()
        self.snapshot.blockSignals(True)
        self.snapshot.clear()
        for directory in list_snapshots(self.settings.datasets_dir, kind):
            info = read_snapshot_info(directory)
            self.snapshot.addItem(f"{directory.name}  ({info.get('samples', 0):,})", directory)
        index = self.snapshot.findData(current)
        self.snapshot.setCurrentIndex(max(0, index))
        self.snapshot.blockSignals(False)
        self._refresh_checkpoints()

    def _refresh_checkpoints(self) -> None:
        self.checkpoint.clear()
        directory = self.snapshot.currentData()
        if directory is None:
            self.run_button.setEnabled(False)
            return

        import yaml

        config_path = Path(directory) / "config.yaml"
        if not config_path.exists():
            self.run_button.setEnabled(False)
            return
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        candidates = []
        if checkpoint_dir.exists():
            for name in ("best.pt", "last.pt"):
                if (checkpoint_dir / name).exists():
                    candidates.append(checkpoint_dir / name)
            candidates.extend(sorted(checkpoint_dir.glob("best_epoch_*.pt"), reverse=True)[:5])

        for path in candidates:
            self.checkpoint.addItem(path.name, path)
        self.run_button.setEnabled(bool(candidates) and not self.runner.is_running)
        if not candidates:
            self.headline.setText("No checkpoint for this snapshot yet — train one first.")

    # -- running -----------------------------------------------------------

    def _run(self) -> None:
        directory = self.snapshot.currentData()
        checkpoint = self.checkpoint.currentData()
        if directory is None or checkpoint is None or self.runner.is_running:
            return

        self.log.clear()
        self.run_button.setEnabled(False)
        self._append_log(f"Evaluating {Path(checkpoint).name} on the {self.split.currentText()} split...")

        task = task_for_kind(self.kind.currentData())
        self.runner.start(
            TrainingJob.evaluate(
                task, Path(directory) / "config.yaml", checkpoint, self.split.currentText()
            )
        )

    def _on_finished(self, exit_code: int, job: TrainingJob | None) -> None:
        self.run_button.setEnabled(True)
        if exit_code != 0:
            self._append_log(f"Evaluation exited with code {exit_code}.")
            return
        self._append_log("Evaluation finished.")
        self._load_reports()

    def _load_reports(self) -> None:
        directory = self.snapshot.currentData()
        if directory is None:
            return

        import yaml

        with (Path(directory) / "config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        reports = Path(config["paths"]["reports_dir"])
        split = self.split.currentText()
        is_multiclass = self.kind.currentData() in ("types", "unified")

        metrics_path = reports / (f"{split}_type_metrics.json" if is_multiclass else f"{split}_metrics.json")
        if not metrics_path.exists():
            self._append_log(f"No metrics file at {metrics_path}")
            return
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

        self._show_metrics(metrics, is_multiclass)
        mistakes_path = reports / (
            f"{split}_type_misclassified_files.csv" if is_multiclass
            else f"{split}_misclassified_files.csv"
        )
        self._show_mistakes(mistakes_path)

        file_metrics = reports / f"{split}_file_metrics.json"
        if self.kind.currentData() == "unified" and file_metrics.exists():
            with file_metrics.open(encoding="utf-8") as handle:
                self._show_file_level(json.load(handle))

    def _show_file_level(self, report: dict[str, Any]) -> None:
        """What the model does to whole files -- the number that matters in use.

        Crop metrics hide the aggregation: a file is flagged if any of its dozen
        candidate regions is called a burst, so a small per-region error rate
        becomes a large per-file one. This block states the per-file outcome at
        the threshold the model will actually be used with.
        """
        quiet = int(report.get("quiet_files", 0))
        bursts = int(report.get("burst_files", 0))
        alarms = int(report.get("false_alarms", 0))
        found = int(report.get("bursts_detected", 0))
        calibrated = report.get("threshold_calibrated", False)
        lines = [
            self.headline.text(),
            "<span style='font-size:13px'><b>Per file</b> (as Predict runs it, threshold "
            f"{float(report.get('threshold', 0.5)):.3f}"
            f"{', calibrated' if calibrated else ', not calibrated'}): "
            f"false alarms <b>{alarms} of {quiet}</b> quiet files"
            + (f" ({alarms / quiet:.1%})" if quiet else "")
            + f" · bursts found <b>{found} of {bursts}</b>"
            + (f" ({found / bursts:.1%})" if bursts else "")
            + f" · {int(report.get('stray_detections', 0))} stray detection(s) in burst files"
            "</span>",
        ]
        self.headline.setText("<br>".join(lines))

        noisy = sorted(
            (
                (station, entry)
                for station, entry in (report.get("false_alarms_by_station") or {}).items()
                if entry.get("false_alarms")
            ),
            key=lambda pair: -pair[1]["false_alarms"],
        )
        if noisy:
            detail = ", ".join(
                f"{station} {entry['false_alarms']}/{entry['quiet_files']}"
                for station, entry in noisy[:6]
            )
            existing = self.metric_warning.text()
            self.metric_warning.setText(
                "\n".join(filter(None, [
                    existing,
                    f"False alarms by station: {detail}. Check those files for an "
                    "unlabelled burst; re-exporting with hard-negative mining targets "
                    "exactly the rest.",
                ]))
            )

        # File-level mistakes first: they are what an operator would see.
        entries = [
            ("false alarm", item) for item in report.get("false_alarm_files", [])
        ] + [("missed burst", item) for item in report.get("missed_burst_files", [])]
        for position, (kind, item) in enumerate(entries[:200]):
            name = Path(item.get("file_path", "")).name
            list_item = QListWidgetItem(
                f"{name}\n    {kind} · {item.get('station', '')} · "
                f"evidence {float(item.get('score', 0)):.3f}"
            )
            list_item.setData(FilePathRole, item.get("file_path", ""))
            list_item.setForeground(QColor("#f85149" if kind == "false alarm" else "#d4a72c"))
            self.mistakes.insertItem(position, list_item)
        if entries:
            self._append_log(
                f"{len(report.get('false_alarm_files', [])):,} false-alarm file(s), "
                f"{len(report.get('missed_burst_files', [])):,} missed burst file(s)."
            )

    # -- rendering ---------------------------------------------------------

    def _show_metrics(self, metrics: dict[str, Any], is_type: bool) -> None:
        self.metric_warning.setText("")
        if is_type:
            self.headline.setText(
                f"Accuracy {metrics.get('accuracy', 0):.4f}    "
                f"Macro-F1 {metrics.get('macro_f1', 0):.4f}    "
                f"Weighted-F1 {metrics.get('weighted_f1', 0):.4f}    "
                f"({metrics.get('num_samples', 0):,} samples)"
            )
            class_names = metrics.get("class_names", [])
            self._show_confusion(metrics.get("confusion_matrix", []), class_names)
            self._show_per_class(metrics.get("per_class", {}))
            self._show_burst_rollup(metrics.get("burst_rollup"))
            self._show_real_world_typing(metrics.get("real_world_typing"))
        else:
            self._show_binary_metrics(metrics)

    def _show_binary_metrics(self, metrics: dict[str, Any]) -> None:
        """Binary results, always next to what a constant predictor would score.

        On an imbalanced set F1 and accuracy are dominated by the majority class,
        so a headline of 0.91 can be *worse* than always answering "Burst". The
        baseline and the missed-burst count are shown so that cannot pass unnoticed.
        """
        tp = int(metrics.get("tp", metrics.get("true_positives", 0)))
        fn = int(metrics.get("fn", metrics.get("false_negatives", 0)))
        fp = int(metrics.get("fp", metrics.get("false_positives", 0)))
        tn = int(metrics.get("tn", metrics.get("true_negatives", 0)))
        bursts = tp + fn
        total = tp + fn + fp + tn

        f1 = float(metrics.get("f1", 0.0))
        balanced = metrics.get("balanced_accuracy")
        # Older reports predate balanced_accuracy; derive it from the matrix.
        if balanced is None and bursts and (tn + fp):
            balanced = 0.5 * (tp / bursts + tn / (tn + fp))

        baseline_f1 = 0.0
        if total:
            share = max(bursts, total - bursts) / total
            baseline_f1 = (2 * share / (1 + share)) if bursts >= total - bursts else 0.0

        lines = [
            f"F1 {f1:.4f}    Precision {metrics.get('precision', 0):.4f}    "
            f"Recall {metrics.get('recall', 0):.4f}    "
            f"ROC-AUC {metrics.get('roc_auc', 0):.4f}    "
            f"PR-AUC {metrics.get('pr_auc', 0):.4f}"
            + (f"    Balanced acc {balanced:.4f}" if balanced is not None else "")
        ]
        if bursts:
            lines.append(
                f"<span style='font-size:12px'>Missed <b>{fn}</b> of {bursts} real bursts "
                f"({fn / bursts:.0%}) at threshold {metrics.get('threshold', 0.5):.3f}"
                f" · false alarms {fp}</span>"
            )
        if total and baseline_f1 > 0:
            verdict = (
                "<span style='color:#f85149'><b>at or below</b></span>"
                if f1 <= baseline_f1 + 1e-9
                else "<span style='color:#3fb950'>above</span>"
            )
            lines.append(
                f"<span style='font-size:12px'>Always-answer-majority baseline F1 = "
                f"{baseline_f1:.4f} — this model is {verdict} it.</span>"
            )
        self.headline.setText("<br>".join(lines))

        warnings = []
        if f1 <= baseline_f1 + 1e-9 and baseline_f1 > 0:
            warnings.append(
                "This model does not beat a constant predictor on F1. The split is too "
                "imbalanced for F1 to mean anything here — judge it on balanced accuracy "
                "and the missed-burst count, and label more of the minority class."
            )
        if balanced is not None and balanced < 0.6:
            warnings.append(
                f"Balanced accuracy {balanced:.3f} is close to the 0.5 a coin-flip scores; "
                "the model has learned little that generalises across both classes."
            )
        from callisto_trainer.core.metrics import threshold_is_extreme

        threshold = float(metrics.get("threshold", 0.5))
        if threshold_is_extreme(threshold):
            warnings.append(
                f"The decision threshold is {threshold:g}, far from 0.5. The model "
                "separates the classes but its probabilities are not calibrated, so they "
                "cannot be read as confidences and a small change in the data could move "
                "many predictions. More minority-class examples would help."
            )
        self.metric_warning.setText("\n".join(warnings))

        self._show_confusion([[tn, fp], [fn, tp]], ["No_Burst", "Burst"])
        self.per_class.setRowCount(0)

    def _show_burst_rollup(self, rollup: dict[str, Any] | None) -> None:
        """The unified model's burst-vs-background summary.

        Macro-F1 answers "did it name the right type". This answers the question
        that matters more operationally: did it notice a burst at all. Confusing
        Type II with Type III is recoverable; calling a real burst background is
        not.
        """
        if not rollup:
            return

        missed = int(rollup.get("false_negatives", 0))
        detected = int(rollup.get("true_positives", 0))
        total = missed + detected
        lines = [
            self.headline.text(),
            "<span style='font-size:12px'>Burst vs background: "
            f"recall <b>{rollup.get('burst_recall', 0):.3f}</b> · "
            f"precision {rollup.get('burst_precision', 0):.3f} · "
            f"background rejected {rollup.get('no_burst_specificity', 0):.3f}</span>",
        ]
        if total:
            lines.append(
                f"<span style='font-size:12px'>Missed <b>{missed}</b> of {total} real "
                f"burst regions ({missed / total:.0%}); of those it did find, the type "
                f"was right {rollup.get('type_accuracy_given_detected', 0):.0%} "
                "of the time.</span>"
            )
        self.headline.setText("<br>".join(lines))

        warnings = []
        if rollup.get("burst_recall", 1.0) < 0.85:
            warnings.append(
                f"Burst recall {rollup['burst_recall']:.3f}: roughly "
                f"{1 - rollup['burst_recall']:.0%} of real burst regions are being called "
                "background. More labelled bursts, or a lower region-brightness setting, "
                "would help."
            )
        if rollup.get("no_burst_specificity", 1.0) < 0.85:
            warnings.append(
                f"Background rejection {rollup['no_burst_specificity']:.3f}: interference "
                "is being typed as bursts. More No_Burst examples would help."
            )
        if warnings:
            existing = self.metric_warning.text()
            self.metric_warning.setText("\n".join(filter(None, [existing, *warnings])))

    def _show_real_world_typing(self, report: dict[str, Any] | None) -> None:
        """Typing as it would be at the observed type frequencies.

        The labelled split holds the rare types far out of proportion, so plain
        per-class scores flatter a model that over-calls them. This weighs each
        type by how often it really occurs, with and without the type-frequency
        correction the model was calibrated with.
        """
        if not report:
            return
        trained, corrected = report.get("as_trained") or {}, report.get("corrected") or {}
        if not trained:
            return
        lines = [
            self.headline.text(),
            "<span style='font-size:12px'>Typing at the observed type frequencies: "
            f"macro-F1 {trained.get('real_world_macro_f1', 0):.3f} as trained → "
            f"<b>{corrected.get('real_world_macro_f1', 0):.3f}</b> corrected "
            f"(strength {corrected.get('strength', 0):.2f}); "
            f"accuracy {trained.get('real_world_accuracy', 0):.0%} → "
            f"{corrected.get('real_world_accuracy', 0):.0%}</span>",
        ]
        per_type = corrected.get("per_type") or {}
        if per_type:
            lines.append(
                "<span style='font-size:12px'>"
                + " · ".join(
                    f"{name}: found {values.get('recall', 0):.0%}, "
                    f"right when called {values.get('real_world_precision', 0):.0%}"
                    for name, values in per_type.items()
                )
                + "</span>"
            )
        self.headline.setText("<br>".join(lines))

    def _show_confusion(self, matrix: list[list[int]], class_names: list[str]) -> None:
        size = len(class_names)
        self.matrix.setRowCount(size)
        self.matrix.setColumnCount(size)
        self.matrix.setHorizontalHeaderLabels([f"pred {name}" for name in class_names])
        self.matrix.setVerticalHeaderLabels([f"true {name}" for name in class_names])
        self.matrix.verticalHeader().setVisible(True)

        peak = max((max(row) for row in matrix if row), default=1) or 1
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                item = QTableWidgetItem(f"{int(value):,}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                intensity = value / peak
                if i == j:
                    item.setBackground(QColor(63, 185, 80, int(40 + 150 * intensity)))
                elif value:
                    item.setBackground(QColor(248, 81, 73, int(40 + 150 * intensity)))
                self.matrix.setItem(i, j, item)

    def _show_per_class(self, per_class: dict[str, dict[str, Any]]) -> None:
        self.per_class.setRowCount(len(per_class))
        for row, (name, scores) in enumerate(sorted(per_class.items())):
            values = [
                name,
                f"{scores.get('precision', 0):.4f}",
                f"{scores.get('recall', 0):.4f}",
                f"{scores.get('f1', 0):.4f}",
                f"{scores.get('support', 0):,}",
            ]
            for column, text in enumerate(values):
                self.per_class.setItem(row, column, QTableWidgetItem(text))

    def _show_mistakes(self, path: Path) -> None:
        self.mistakes.clear()
        if not path.exists():
            return
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows[:500]:
            label = row.get("label", "?")
            predicted = row.get("predicted_label", "?")
            name = Path(row.get("file_path", "")).name
            item = QListWidgetItem(f"{name}\n    true: {label}    predicted: {predicted}")
            item.setData(FilePathRole, row.get("file_path", ""))
            self.mistakes.addItem(item)

        if len(rows) > 500:
            self.mistakes.addItem(f"... and {len(rows) - 500:,} more")
        self._append_log(f"{len(rows):,} misclassified sample(s).")

    def _on_mistake_activated(self, item: QListWidgetItem) -> None:
        path = item.data(FilePathRole)
        if path:
            self.inspect_file_requested.emit(str(path))

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def shutdown(self) -> None:
        if self.runner.is_running:
            self.runner.stop()
