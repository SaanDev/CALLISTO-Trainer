"""Right-hand labelling controls: verdict, burst types, box list, crop preview.

The crop preview is the important piece. It renders the exact 224x224 tensor the
selected box will produce -- same normalization, same geometry, same
interpolation as the exporter -- so the operator can see whether a box actually
captures the burst before committing it. Without it, a box that looks fine on a
40,000-column canvas can silently become an unusable training sample.

The crop is the drawn region and nothing else (see :mod:`core.crops`), so the
preview is a literal picture of the training sample, not an approximation of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.coords import SpectrumAxes, format_time_label
from callisto_trainer.core.crops import (
    CropConfig,
    PixelBox,
    context_box,
    crop_context,
    crop_from_normalized,
    expand_box,
)
from callisto_trainer.store.repository import (
    BOX_TYPES,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    VERDICT_UNSURE,
    BoxRecord,
)
from callisto_trainer.ui.spectrogram_view import color_for_type

BoxIdRole = Qt.ItemDataRole.UserRole + 1


class CropPreview(QWidget):
    """Shows the exact tensor a box will contribute to training."""

    def __init__(self, parent: QWidget | None = None, minimum_height: int = 180) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMinimumHeight(minimum_height)
        self.plot.getViewBox().setAspectLocked(True)
        self.plot.getViewBox().invertY(True)
        self.plot.getViewBox().setMouseEnabled(x=False, y=False)

        self.image = pg.ImageItem()
        try:
            colormap = pg.colormap.get("inferno")
            self.image.setLookupTable(colormap.getLookupTable(nPts=256))
        except Exception:
            pass
        self.image.setLevels((0.0, 1.0))
        self.plot.addItem(self.image)

        self.caption = QLabel("No box selected")
        self.caption.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.caption.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.plot)
        layout.addWidget(self.caption)

    def show_crop(self, tensor: np.ndarray, caption: str) -> None:
        self.image.setImage(tensor[0], autoLevels=False, levels=(0.0, 1.0))
        self.caption.setText(caption)

    def clear(self) -> None:
        self.image.clear()
        self.caption.setText("No box selected")


class LabelPanel(QWidget):
    """Verdict, burst-type palette, per-file box list and crop preview."""

    verdict_changed = Signal(str)          # burst | no_burst | unsure
    type_assigned = Signal(str)            # applies to the selected box
    box_selected = Signal(int)             # box_id, or -1
    box_delete_requested = Signal(int)
    notes_changed = Signal(str)

    def __init__(self, crop_config: CropConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.crop_config = crop_config
        self._normalized: np.ndarray | None = None
        self._quiet: np.ndarray | None = None
        self._axes: SpectrumAxes | None = None
        self._boxes: list[BoxRecord] = []
        self._suspend = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_verdict_group())
        layout.addWidget(self._build_type_group())
        layout.addWidget(self._build_box_group(), 1)
        layout.addWidget(self._build_preview_group())
        layout.addWidget(self._build_physics_group())
        layout.addWidget(self._build_notes_group())

    # -- construction ------------------------------------------------------

    def _build_verdict_group(self) -> QGroupBox:
        group = QGroupBox("Does this file contain a burst?")
        row = QHBoxLayout(group)
        self.verdict_buttons = QButtonGroup(self)

        for label, verdict, shortcut, color in (
            ("Burst", VERDICT_BURST, "B", "#3fb950"),
            ("No burst", VERDICT_NO_BURST, "N", "#8b949e"),
            ("Unsure", VERDICT_UNSURE, "U", "#d4a72c"),
        ):
            button = QPushButton(f"{label}  ({shortcut})")
            button.setCheckable(True)
            button.setMinimumHeight(34)
            button.setStyleSheet(
                f"QPushButton:checked {{ background-color: {color}; color: #0d1117; font-weight: bold; }}"
            )
            button.clicked.connect(lambda _checked, v=verdict: self._on_verdict_clicked(v))
            self.verdict_buttons.addButton(button)
            row.addWidget(button)
            setattr(self, f"_verdict_{verdict}", button)
        return group

    def _build_type_group(self) -> QGroupBox:
        """The five burst types, in key order: 1 II, 2 III, 3 IIIG, 4 IV, 5 Other.

        There is no RFI button: interference is never drawn. Everything outside
        the burst boxes, and every no-burst file, already teaches the model what
        is not a burst, and the exporter names the interference in it itself.
        """
        group = QGroupBox("Type for selected box")
        grid = QGridLayout(group)
        grid.setSpacing(4)
        tooltips = {
            "Type IIIG": "A group of three or more Type III bursts in quick succession.",
            "Type IV": (
                "Broadband continuum lasting minutes to hours, stationary or slowly "
                "drifting, often with fine structure (pulsations, zebra, fibres)."
            ),
            "Other": "A solar burst that is none of the types above (Type I, V, U, J, ...).",
        }
        for index, box_type in enumerate(BOX_TYPES):
            button = QPushButton(f"{box_type}  ({index + 1})")
            button.setMinimumHeight(30)
            color = color_for_type(box_type)
            button.setStyleSheet(
                f"QPushButton {{ border-left: 4px solid {color}; padding-left: 6px; }}"
            )
            if box_type in tooltips:
                button.setToolTip(tooltips[box_type])
            button.clicked.connect(lambda _checked, t=box_type: self.type_assigned.emit(t))
            # Three types on the first row; Type IV and Other on the second.
            grid.addWidget(button, index // 3, index % 3)
        return group

    def _build_box_group(self) -> QGroupBox:
        group = QGroupBox("Bursts in this file")
        layout = QVBoxLayout(group)
        self.box_list = QListWidget()
        self.box_list.setAlternatingRowColors(True)
        self.box_list.currentItemChanged.connect(self._on_box_row_changed)
        layout.addWidget(self.box_list)

        self.delete_button = QPushButton("Delete selected box  (Del)")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self.delete_button)

        self.hint = QLabel("Drag on the spectrum to mark a burst.")
        self.hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        return group

    def _build_preview_group(self) -> QGroupBox:
        """The views the unified model is given: the crop, its context, and the
        context on the quiet-part background.

        The context view is what lets the model see that a carrier continues
        past the box or that an impulse spans the whole band; the quiet one keeps
        a continuum lasting most of the file (Type IV) visible. Both are shown
        rather than left as invisible parts of the training sample.
        """
        group = QGroupBox("Training tensors for this box")
        layout = QHBoxLayout(group)
        self.preview = CropPreview()
        self.context_preview = CropPreview(minimum_height=180)
        self.context_preview.caption.setText("Context view")
        self.quiet_preview = CropPreview(minimum_height=180)
        self.quiet_preview.caption.setText("Quiet background")
        layout.addWidget(self.preview)
        layout.addWidget(self.context_preview)
        layout.addWidget(self.quiet_preview)
        return group

    def _build_physics_group(self) -> QGroupBox:
        """Burst parameters of the selected box, for Type II and Type III.

        Calculated from the box itself: its height is the frequency range and
        its width the duration, and df/dt = (f_start - f_end) / (t_start - t_end)
        with the burst starting at the top of the box (see
        ``burst_physics.box_parameters``). Drift rate is what physically separates
        the two -- a Type III is an electron beam, a Type II a shock front two
        orders of magnitude slower -- so showing it as the box is drawn lets an
        inconsistent label be caught immediately rather than at training time.
        """
        group = QGroupBox("Burst parameters (Type II / III)")
        layout = QVBoxLayout(group)
        layout.setSpacing(3)

        self.physics_labels: dict[str, QLabel] = {}
        for key, caption in (
            ("frequency", "f_start → f_end"),
            ("time", "t_start → t_end"),
            ("extent", "Duration / bandwidth"),
            ("drift", "Drift rate df/dt"),
            ("relative", "Relative drift"),
            ("bursts", "Separate bursts"),
        ):
            row = QHBoxLayout()
            name = QLabel(caption)
            name.setStyleSheet("color: #8b949e; font-size: 11px;")
            name.setMinimumWidth(120)
            value = QLabel("-")
            value.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(name)
            row.addWidget(value, 1)
            layout.addLayout(row)
            self.physics_labels[key] = value

        self.physics_note = QLabel("")
        self.physics_note.setWordWrap(True)
        self.physics_note.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(self.physics_note)

        self.physics_warning = QLabel("")
        self.physics_warning.setWordWrap(True)
        self.physics_warning.setStyleSheet("color: #d4a72c; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.physics_warning)

        # Type III vs Type IIIG: how many separate bursts the box holds.
        self.group_hint = QLabel("")
        self.group_hint.setWordWrap(True)
        self.group_hint.setStyleSheet("color: #3a86ff; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.group_hint)
        return group

    def show_physics(self, physics: Any | None, burst_type: str | None) -> None:
        """Display a measurement, and flag it if it contradicts the assigned type."""
        from callisto_trainer.core.burst_physics import consistency_warning, group_hint

        count = int(getattr(physics, "burst_count", 0) or 0) if physics is not None else 0
        hint = group_hint(physics, burst_type)
        self.group_hint.setText(f"ⓘ {hint}" if hint else "")

        if physics is None or not getattr(physics, "measured", False):
            for label in self.physics_labels.values():
                label.setText("-")
            self.physics_labels["bursts"].setText(str(count) if count else "-")
            self.physics_note.setText(
                "" if physics is None else (physics.note or "No clean burst track in this region.")
            )
            self.physics_warning.setText("")
            return

        self.physics_labels["frequency"].setText(
            f"{physics.freq_start_mhz:.2f} → {physics.freq_end_mhz:.2f} MHz"
        )
        self.physics_labels["time"].setText(
            f"+{physics.time_start_s:.2f} → +{physics.time_end_s:.2f} s"
        )
        self.physics_labels["extent"].setText(
            f"{physics.duration_s:.2f} s  /  {physics.bandwidth_mhz:.2f} MHz"
        )
        self.physics_labels["drift"].setText(f"{physics.drift_mhz_per_s:+.4f} MHz/s")
        self.physics_labels["relative"].setText(
            f"{physics.relative_drift_per_s:+.5f} s⁻¹"
            if physics.relative_drift_per_s is not None
            else "-"
        )
        self.physics_labels["bursts"].setText(str(count) if count else "-")

        if getattr(physics, "from_box", False):
            self.physics_note.setText(
                "From the box: its height is the frequency range and its width the "
                "duration, so df/dt = (f_start − f_end) / (t_start − t_end). Draw it from "
                "the burst's start to its end."
            )
        else:
            colour = {"good": "#3fb950", "fair": "#d4a72c"}.get(physics.confidence, "#f85149")
            self.physics_note.setText(
                f"<span style='color:{colour}'>{physics.confidence} fit</span> · "
                f"{physics.track_samples} samples along {physics.track_axis}"
                + (" · burst touches the box edge, extent may be cut off"
                   if physics.edge_clipped else "")
            )
        warning = consistency_warning(physics, burst_type or "")
        self.physics_warning.setText(f"⚠ {warning}" if warning else "")

    def _build_notes_group(self) -> QGroupBox:
        group = QGroupBox("Notes")
        layout = QVBoxLayout(group)
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(56)
        self.notes.setPlaceholderText("Optional note about this file...")
        self.notes.textChanged.connect(
            lambda: None if self._suspend else self.notes_changed.emit(self.notes.toPlainText())
        )
        layout.addWidget(self.notes)
        return group

    # -- state -------------------------------------------------------------

    def set_file(
        self,
        normalized: np.ndarray | None,
        axes: SpectrumAxes | None,
        verdict: str | None,
        notes: str,
        boxes: list[BoxRecord],
        quiet: np.ndarray | None = None,
    ) -> None:
        self._normalized = normalized
        self._quiet = quiet
        self._axes = axes
        self.set_verdict(verdict)
        self._suspend = True
        self.notes.setPlainText(notes or "")
        self._suspend = False
        self.set_boxes(boxes)

    def set_notes(self, text: str) -> None:
        """Set the notes box without emitting a change back to the store."""
        self._suspend = True
        self.notes.setPlainText(text or "")
        self._suspend = False

    def set_verdict(self, verdict: str | None) -> None:
        self.verdict_buttons.setExclusive(False)
        for name in (VERDICT_BURST, VERDICT_NO_BURST, VERDICT_UNSURE):
            button = getattr(self, f"_verdict_{name}", None)
            if button is not None:
                button.setChecked(name == verdict)
        self.verdict_buttons.setExclusive(True)

    def set_boxes(self, boxes: list[BoxRecord], select_id: int | None = None) -> None:
        self._boxes = boxes
        self._suspend = True
        self.box_list.clear()
        for index, box in enumerate(boxes, start=1):
            item = QListWidgetItem(self._describe(index, box))
            item.setData(BoxIdRole, box.id)
            item.setForeground(QColor(color_for_type(box.burst_type)))
            if not box.confirmed:
                font = QFont()
                font.setItalic(True)
                item.setFont(font)
            self.box_list.addItem(item)
        self._suspend = False

        self.hint.setText(
            "Drag on the spectrum to mark a burst."
            if boxes
            else "No bursts marked yet. Drag on the spectrum to add one."
        )
        if select_id is not None:
            self.select_box(select_id)
        elif boxes:
            self.select_box(boxes[-1].id)
        else:
            self._clear_previews()

    def _describe(self, index: int, box: BoxRecord) -> str:
        parts = [f"{index}. {box.burst_type}"]
        if box.freq_lo_mhz is not None and box.freq_hi_mhz is not None:
            parts.append(f"{box.freq_lo_mhz:.0f}-{box.freq_hi_mhz:.0f} MHz")
        if self._axes is not None and box.t_start_s is not None:
            parts.append(format_time_label(self._axes, box.t_start_s))
        if not box.confirmed:
            parts.append("(proposed)")
        return "   ".join(parts)

    def select_box(self, box_id: int | None) -> None:
        self._suspend = True
        matched = False
        for row in range(self.box_list.count()):
            item = self.box_list.item(row)
            if item.data(BoxIdRole) == box_id:
                self.box_list.setCurrentItem(item)
                matched = True
                break
        if not matched:
            self.box_list.clearSelection()
            self.box_list.setCurrentRow(-1)
        self._suspend = False
        self._update_preview(box_id if matched else None)

    def current_box_id(self) -> int | None:
        item = self.box_list.currentItem()
        return item.data(BoxIdRole) if item else None

    # -- preview -----------------------------------------------------------

    def refresh_preview(self) -> None:
        self._update_preview(self.current_box_id())

    def _clear_previews(self) -> None:
        self.preview.clear()
        self.context_preview.clear()
        self.context_preview.caption.setText("Context view")
        self.quiet_preview.clear()
        self.quiet_preview.caption.setText("Quiet background")

    def _update_preview(self, box_id: int | None) -> None:
        if box_id is None or self._normalized is None:
            self._clear_previews()
            return
        box = next((b for b in self._boxes if b.id == box_id), None)
        if box is None:
            self._clear_previews()
            return

        drawn = PixelBox(box.row0, box.row1, box.col0, box.col1)
        try:
            tensor = crop_from_normalized(self._normalized, drawn, self.crop_config)
            context = crop_context(self._normalized, drawn, self.crop_config)
        except ValueError:
            self._clear_previews()
            return
        around = context_box(drawn, self._normalized.shape, self.crop_config)
        self.context_preview.show_crop(
            context,
            f"Context · full band, {around.n_cols:,} columns around the box",
        )
        if self._quiet is not None and self._quiet.shape == self._normalized.shape:
            self.quiet_preview.show_crop(
                crop_context(self._quiet, drawn, self.crop_config),
                "Quiet background · the same context, background from each channel's "
                "quietest tenth",
            )
        else:
            self.quiet_preview.clear()

        # Report the region actually sliced rather than the configured margin, so
        # any geometry that grows the box beyond the drawing says so out loud
        # instead of the caption quietly claiming a crop it did not take.
        cropped = expand_box(drawn, self._normalized.shape, self.crop_config)
        size = f"{drawn.n_rows} x {drawn.n_cols} px"
        extent = (
            f"{size} exactly"
            if cropped.as_tuple() == drawn.as_tuple()
            else f"{size} → {cropped.n_rows} x {cropped.n_cols} px with context"
        )
        caption = f"{box.burst_type} · {extent} → {tensor.shape[1]}x{tensor.shape[2]}"
        self.preview.show_crop(tensor, caption)

    # -- signals -----------------------------------------------------------

    def _on_verdict_clicked(self, verdict: str) -> None:
        if not self._suspend:
            self.verdict_changed.emit(verdict)

    def _on_box_row_changed(self, current: QListWidgetItem, _previous) -> None:
        if self._suspend:
            return
        box_id = current.data(BoxIdRole) if current else None
        self._update_preview(box_id)
        self.box_selected.emit(box_id if box_id is not None else -1)

    def _on_delete_clicked(self) -> None:
        box_id = self.current_box_id()
        if box_id is not None:
            self.box_delete_requested.emit(box_id)
