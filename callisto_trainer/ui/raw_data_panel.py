"""Raw-data inspection: the FITS header, and the numbers under the cursor.

The labelling canvas necessarily shows an *interpretation* of the file -- the
background-subtracted, dB-windowed view the model is trained on. These two
widgets expose what is underneath it, so a decision never has to rest on trust
in the preprocessing alone.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PixelInspector(QGroupBox):
    """The same sample shown at three stages of the pipeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Value under cursor", parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setVerticalSpacing(3)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)

        self._values: dict[str, QLabel] = {}
        rows = [
            ("time", "Time (UTC)"),
            ("mhz", "Frequency"),
            ("pixel", "Row / column"),
            ("raw", "Raw digit"),
            ("db", "dB above bkg"),
            ("normalized", "Model input"),
        ]
        for index, (key, caption) in enumerate(rows):
            name = QLabel(caption)
            name.setStyleSheet("color: #8b949e;")
            value = QLabel("-")
            value.setFont(mono)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(name, index, 0)
            layout.addWidget(value, index, 1)
            self._values[key] = value

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #d4a72c; font-size: 11px;")
        layout.addWidget(self.note, len(rows), 0, 1, 2)

    def show_sample(self, sample: dict[str, Any]) -> None:
        self._values["time"].setText(sample.get("time_label", "-"))
        self._values["mhz"].setText(f"{sample['mhz']:.3f} MHz")
        self._values["pixel"].setText(f"{sample['row']}, {sample['column']}")
        raw = sample.get("raw")
        raw_invalid = raw is not None and not math.isfinite(raw)
        if raw is None:
            self._values["raw"].setText("-")
        elif raw_invalid:
            self._values["raw"].setText("NaN" if math.isnan(raw) else "Inf")
        else:
            self._values["raw"].setText(f"{raw:.1f}")
        self._values["raw"].setStyleSheet(
            "color: #f85149; font-weight: bold;" if raw_invalid else ""
        )
        self._values["db"].setText(f"{sample['db']:+.3f} dB")
        self._values["normalized"].setText(f"{sample['normalized']:.4f}")

        if raw_invalid:
            self._values["normalized"].setStyleSheet("")
            self.note.setText(
                "This sample is invalid in the file. Preprocessing replaces it with "
                "the finite median, so the model sees a substituted value here, not "
                "a measurement."
            )
        elif sample.get("clipped"):
            self._values["normalized"].setStyleSheet("color: #d4a72c; font-weight: bold;")
            self.note.setText(
                "Saturated: this sample sits at the edge of the -1..8 dB training "
                "window, so the model cannot tell it apart from anything beyond it."
            )
        else:
            self._values["normalized"].setStyleSheet("")
            self.note.setText("")

    def clear(self) -> None:
        for label in self._values.values():
            label.setText("-")
            label.setStyleSheet("")
        self.note.setText("")


class HeaderPanel(QWidget):
    """The file's FITS header, extension list and axis provenance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(self.path_label)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.text.setFont(mono)
        layout.addWidget(self.text, 1)

        self.copy_button = QPushButton("Copy header to clipboard")
        self.copy_button.clicked.connect(self._copy)
        layout.addWidget(self.copy_button)

    def show_header(self, path: str, header_text: str) -> None:
        self.path_label.setText(path)
        self.text.setPlainText(header_text or "No header available for this file.")

    def clear(self) -> None:
        self.path_label.setText("")
        self.text.setPlainText("")

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.text.toPlainText())
