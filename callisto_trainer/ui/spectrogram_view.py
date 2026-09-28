"""Interactive dynamic-spectrum canvas with burst box editing.

## Coordinate system

The view works in **native pixel coordinates**: x is the original time-sample
column, y is the frequency-channel row, always at full resolution regardless of
what is currently rendered. Boxes are therefore created, moved and stored in the
same units the crop extractor uses, with no scaling round-trip to get wrong.

Frequency channels are not evenly spaced and normally run high-to-low, so the
axes are *not* linear transforms of the pixel grid. Tick labels are produced by
looking each value up in the real axis arrays, which handles non-uniform and
descending channels exactly. The y axis is inverted so row 0 -- the highest
frequency -- sits at the top, matching how e-CALLISTO spectra are conventionally
read.

## Level of detail

Files reach ~40,000 time samples. Rendering that whole array on every pan would
be wasteful, so only the visible column span is uploaded, max-pooled down to at
most ``MAX_RENDER_COLS``. Max, never mean: a Type III lane can be two columns
wide and averaging would erase it. Because the image is positioned with
``setRect`` in original column units, the pooling is invisible to everything
else -- zooming in simply re-renders the smaller span at full resolution.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QWidget

from callisto_trainer.core.coords import SpectrumAxes, format_time_label
from callisto_trainer.services.cache import decimate_for_display

# Render at most this many columns at once; more cannot be resolved on screen.
MAX_RENDER_COLS = 4000
# Coalesce range-change redraws so a drag does not trigger one slice per frame.
REDRAW_DEBOUNCE_MS = 25
# Smallest box the user can create by dragging, in pixels; anything smaller is
# treated as a click (deselect) rather than an accidental zero-area box.
MIN_DRAG_PIXELS = 3

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "#101418")
pg.setConfigOption("foreground", "#c8d0d8")

# Colours per box type, reused by the legend and the box list. Type IIIG is a
# deeper blue-green than Type III so a group reads as related but distinct; RFI
# (found by the model, never drawn) is a magenta that no burst type uses, so
# interference is never mistaken for one.
TYPE_COLORS: dict[str, str] = {
    "Type II": "#ff6b35",
    "Type III": "#4ecdc4",
    "Type IIIG": "#3a86ff",
    "Type IV": "#9b5de5",
    "Other": "#ffd166",
    "RFI": "#e040fb",
}
DEFAULT_TYPE_COLOR = "#c0c0c0"


def color_for_type(burst_type: str) -> str:
    return TYPE_COLORS.get(burst_type, DEFAULT_TYPE_COLOR)


class FrequencyAxis(pg.AxisItem):
    """Left axis labelled in MHz by looking rows up in the real frequency array."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._axes: SpectrumAxes | None = None

    def set_axes(self, axes: SpectrumAxes | None) -> None:
        self._axes = axes
        self.picture = None
        self.update()

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if self._axes is None or self._axes.n_freq == 0:
            return [""] * len(values)
        labels = []
        for value in values:
            row = int(round(value))
            if 0 <= row < self._axes.n_freq:
                labels.append(f"{float(self._axes.freq_mhz[row]):.1f}")
            else:
                labels.append("")
        return labels


class TimeAxis(pg.AxisItem):
    """Bottom axis labelled with absolute UTC when the date is known."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._axes: SpectrumAxes | None = None

    def set_axes(self, axes: SpectrumAxes | None) -> None:
        self._axes = axes
        self.picture = None
        self.update()

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if self._axes is None or self._axes.n_time == 0:
            return [""] * len(values)
        labels = []
        for value in values:
            column = int(round(value))
            column = max(0, min(column, self._axes.n_time - 1))
            labels.append(format_time_label(self._axes, float(self._axes.time_s[column])))
        return labels


class BurstROI(pg.RectROI):
    """A resizable box bound to one stored annotation."""

    def __init__(self, box_id: int, rect: QRectF, burst_type: str, confirmed: bool = True) -> None:
        pen = pg.mkPen(color_for_type(burst_type), width=2)
        if not confirmed:
            # Proposals stay visually provisional until the operator accepts them.
            pen.setStyle(Qt.PenStyle.DashLine)
        super().__init__(
            pos=[rect.x(), rect.y()],
            size=[rect.width(), rect.height()],
            pen=pen,
            hoverPen=pg.mkPen(color_for_type(burst_type), width=3),
            handlePen=pg.mkPen("#ffffff", width=1),
            movable=True,
            rotatable=False,
            resizable=True,
        )
        self.box_id = box_id
        self.burst_type = burst_type
        self.confirmed = confirmed

        # Handles on all four corners and edges for comfortable adjustment.
        for x, y in ((0, 0), (1, 0), (0, 1)):
            self.addScaleHandle([x, y], [1 - x, 1 - y])
        self.addScaleHandle([0.5, 0], [0.5, 1])
        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([0, 0.5], [1, 0.5])
        self.addScaleHandle([1, 0.5], [0, 0.5])

        self.label = pg.TextItem(burst_type, color=color_for_type(burst_type), anchor=(0, 1))
        self.label.setFont(QFont("Segoe UI", 8))
        self.label.setParentItem(self)
        self.label.setPos(0, 0)

    def set_burst_type(self, burst_type: str) -> None:
        self.burst_type = burst_type
        pen = pg.mkPen(color_for_type(burst_type), width=2)
        if not self.confirmed:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.hoverPen = pg.mkPen(color_for_type(burst_type), width=3)
        self.label.setText(burst_type)
        self.label.setColor(color_for_type(burst_type))

    def set_selected(self, selected: bool) -> None:
        pen = pg.mkPen(color_for_type(self.burst_type), width=3 if selected else 2)
        if not self.confirmed:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

    def pixel_box(self) -> tuple[int, int, int, int]:
        """Current geometry as ``(row0, row1, col0, col1)``, normalised and integral."""
        position, size = self.pos(), self.size()
        col0, col1 = sorted((position.x(), position.x() + size.x()))
        row0, row1 = sorted((position.y(), position.y() + size.y()))
        return (
            int(round(row0)),
            max(int(round(row1)), int(round(row0)) + 1),
            int(round(col0)),
            max(int(round(col1)), int(round(col0)) + 1),
        )


class SpectrogramViewBox(pg.ViewBox):
    """ViewBox where a plain left-drag draws a new box instead of panning.

    Drawing is the primary action in this tool, so it gets the primary gesture.
    Panning stays available on middle-drag and with Space held, and the wheel
    still zooms, so navigation is never lost.
    """

    region_drawn = Signal(QRectF)
    clicked_empty = Signal()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.draw_mode = True
        self._origin: QPointF | None = None
        self._rubber_band = QRectF()

        self._preview = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("#ffffff", width=1))
        self._preview.setVisible(False)
        self._preview.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        for handle in list(self._preview.handles):
            self._preview.removeHandle(handle["item"])
        self.addItem(self._preview, ignoreBounds=True)

    def _panning_requested(self, event: Any) -> bool:
        modifiers = event.modifiers()
        return (
            not self.draw_mode
            or bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            or event.button() == Qt.MouseButton.MiddleButton
        )

    def mouseDragEvent(self, event: Any, axis: int | None = None) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._panning_requested(event):
            super().mouseDragEvent(event, axis=axis)
            return

        event.accept()
        position = self.mapToView(event.pos())

        if event.isStart():
            self._origin = position
            self._preview.setVisible(True)

        if self._origin is None:
            return

        x0, x1 = sorted((self._origin.x(), position.x()))
        y0, y1 = sorted((self._origin.y(), position.y()))
        self._rubber_band = QRectF(x0, y0, max(x1 - x0, 1e-6), max(y1 - y0, 1e-6))
        self._preview.setPos([x0, y0], finish=False)
        self._preview.setSize([self._rubber_band.width(), self._rubber_band.height()], finish=False)

        if event.isFinish():
            self._preview.setVisible(False)
            self._origin = None
            span = self.mapViewToDevice(
                QPointF(self._rubber_band.width(), self._rubber_band.height())
            ) - self.mapViewToDevice(QPointF(0, 0))
            if abs(span.x()) < MIN_DRAG_PIXELS and abs(span.y()) < MIN_DRAG_PIXELS:
                self.clicked_empty.emit()
            else:
                self.region_drawn.emit(self._rubber_band)


class SpectrogramView(QWidget):
    """The labelling canvas: image, axes, crosshair readout and burst boxes."""

    box_created = Signal(int, int, int, int)  # row0, row1, col0, col1
    box_edited = Signal(int, int, int, int, int)  # box_id, row0, row1, col0, col1
    box_selected = Signal(int)  # box_id, or -1 for none
    cursor_moved = Signal(dict)  # sample_at() readout
    cursor_left = Signal()
    levels_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._array: np.ndarray | None = None
        self._raw: np.ndarray | None = None
        self._quiet: np.ndarray | None = None
        self._axes: SpectrumAxes | None = None
        self._rois: dict[int, BurstROI] = {}
        self._selected_id: int | None = None
        self._suspend_edit_signal = False
        self._levels: tuple[float, float] = (0.0, 1.0)
        # Which array is on screen. Boxes are always stored against the same
        # pixel grid, so switching views never moves an annotation.
        self._view_mode = "normalized"
        self._raw_levels: tuple[float, float] = (0.0, 1.0)

        self._freq_axis = FrequencyAxis(orientation="left")
        self._time_axis = TimeAxis(orientation="bottom")
        self._viewbox = SpectrogramViewBox()

        self.plot = pg.PlotWidget(
            viewBox=self._viewbox,
            axisItems={"left": self._freq_axis, "bottom": self._time_axis},
        )
        self.plot.setLabel("left", "Frequency", units="MHz")
        self.plot.setLabel("bottom", "Time (UTC)")
        self.plot.setMenuEnabled(False)
        self._viewbox.invertY(True)  # row 0 = highest frequency, drawn at the top
        self._viewbox.setAspectLocked(False)

        self.image = pg.ImageItem()
        self.image.setAutoDownsample(False)  # LOD is handled explicitly below
        self.plot.addItem(self.image)
        self._apply_colormap()

        self._rfi_lines: list[pg.InfiniteLine] = []
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#5a6570", width=1))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#5a6570", width=1))
        for line in (self._crosshair_v, self._crosshair_h):
            line.setVisible(False)
            self.plot.addItem(line, ignoreBounds=True)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(REDRAW_DEBOUNCE_MS)
        self._redraw_timer.timeout.connect(self._render_visible)

        self._viewbox.sigRangeChanged.connect(lambda *_: self._redraw_timer.start())
        self._viewbox.region_drawn.connect(self._on_region_drawn)
        self._viewbox.clicked_empty.connect(lambda: self.select_box(None, notify=True))
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    # -- appearance --------------------------------------------------------

    def _apply_colormap(self, name: str = "inferno") -> None:
        try:
            colormap = pg.colormap.get(name)
        except Exception:
            colormap = pg.colormap.get("viridis")
        if colormap is not None:
            self.image.setLookupTable(colormap.getLookupTable(nPts=256))

    def set_colormap(self, name: str) -> None:
        self._apply_colormap(name)

    def set_levels(self, low: float, high: float) -> None:
        """Set display contrast. ``(0, 1)`` is exactly what the model receives."""
        if high <= low:
            return
        self._levels = (float(low), float(high))
        if self._view_mode in ("normalized", "quiet"):
            self.image.setLevels(self._levels)
        self.levels_changed.emit(*self._levels)

    @property
    def levels(self) -> tuple[float, float]:
        return self._levels

    @property
    def levels_match_model(self) -> bool:
        return abs(self._levels[0]) < 1e-9 and abs(self._levels[1] - 1.0) < 1e-9

    def set_draw_mode(self, enabled: bool) -> None:
        self._viewbox.draw_mode = bool(enabled)

    # -- data --------------------------------------------------------------

    def set_spectrum(
        self,
        array: np.ndarray,
        axes: SpectrumAxes,
        rfi_channels: np.ndarray | None = None,
        reset_view: bool = True,
        raw: np.ndarray | None = None,
        raw_levels: tuple[float, float] | None = None,
        quiet: np.ndarray | None = None,
    ) -> None:
        """Display a normalized ``[frequency, time]`` array with its axes.

        ``raw`` is the same spectrum as stored in the FITS; supplying it enables
        the raw/preprocessed toggle and the pixel inspector's raw column.
        ``quiet`` is the quiet-part background of the same file, on the model's
        dB scale; supplying it enables the quiet-background view.
        """
        self._array = array
        self._raw = raw
        self._quiet = quiet
        self._raw_levels = raw_levels or (0.0, 1.0)
        self._axes = axes
        self._freq_axis.set_axes(axes)
        self._time_axis.set_axes(axes)
        self.clear_boxes()

        n_freq, n_time = array.shape
        if reset_view:
            self._viewbox.setLimits(
                xMin=0, xMax=n_time, yMin=0, yMax=n_freq, minXRange=4, minYRange=2
            )
            self._viewbox.setRange(
                xRange=(0, n_time), yRange=(0, n_freq), padding=0.0, update=False
            )
        self._set_rfi_overlay(rfi_channels)
        self._render_visible()

    @property
    def has_raw(self) -> bool:
        return self._raw is not None

    @property
    def view_mode(self) -> str:
        return self._view_mode

    def set_view_mode(self, mode: str) -> None:
        """Switch between ``"normalized"`` (what the model sees), ``"raw"`` and
        ``"quiet"`` (the quiet-part background, where long continua stay bright).

        Only the displayed pixels change. Zoom, boxes and every coordinate stay
        put, because all the arrays share the same row/column grid.
        """
        if mode not in ("normalized", "raw", "quiet"):
            raise ValueError(f"Unknown view mode: {mode}")
        if (mode == "raw" and self._raw is None) or (mode == "quiet" and self._quiet is None):
            return
        self._view_mode = mode
        self._render_visible()

    def _display_array(self) -> np.ndarray | None:
        if self._view_mode == "raw":
            return self._raw
        if self._view_mode == "quiet":
            return self._quiet
        return self._array

    def _display_levels(self) -> tuple[float, float]:
        return self._raw_levels if self._view_mode == "raw" else self._levels

    def clear(self) -> None:
        self._array = None
        self._raw = None
        self._quiet = None
        self._axes = None
        self.clear_boxes()
        self._set_rfi_overlay(None)
        self.image.clear()
        self._freq_axis.set_axes(None)
        self._time_axis.set_axes(None)

    def _set_rfi_overlay(self, channels: np.ndarray | None) -> None:
        for line in self._rfi_lines:
            self.plot.removeItem(line)
        self._rfi_lines.clear()
        if channels is None or self._axes is None or channels.size == 0:
            return

        from callisto_trainer.core.coords import mhz_to_row

        pen = pg.mkPen("#7a4fff", width=1, style=Qt.PenStyle.DotLine)
        for mhz in np.atleast_1d(channels)[:64]:  # cap: some stations flag hundreds
            row = float(mhz_to_row(self._axes, float(mhz)))
            if 0 <= row < self._axes.n_freq:
                line = pg.InfiniteLine(pos=row, angle=0, movable=False, pen=pen)
                line.setZValue(-5)
                line.setVisible(False)
                self.plot.addItem(line, ignoreBounds=True)
                self._rfi_lines.append(line)

    def set_rfi_visible(self, visible: bool) -> None:
        for line in self._rfi_lines:
            line.setVisible(visible)

    # -- level of detail ---------------------------------------------------

    def _render_visible(self) -> None:
        source = self._display_array()
        if source is None:
            return
        n_freq, n_time = source.shape
        (x_min, x_max), _ = self._viewbox.viewRange()

        col0 = max(0, int(np.floor(x_min)))
        col1 = min(n_time, int(np.ceil(x_max)) + 1)
        if col1 <= col0:
            col0, col1 = 0, n_time

        window = source[:, col0:col1]
        pooled, factor = decimate_for_display(
            window, MAX_RENDER_COLS, nan_aware=self._view_mode == "raw"
        )

        self.image.setImage(pooled, autoLevels=False, levels=self._display_levels())
        # Positioning in original column units keeps every other coordinate in
        # this class independent of whatever pooling just happened.
        self.image.setRect(QRectF(col0, 0, col1 - col0, n_freq))

    # -- boxes -------------------------------------------------------------

    def add_box(
        self, box_id: int, row0: int, row1: int, col0: int, col1: int,
        burst_type: str, confirmed: bool = True,
    ) -> None:
        rect = QRectF(col0, row0, max(col1 - col0, 1), max(row1 - row0, 1))
        roi = BurstROI(box_id, rect, burst_type, confirmed=confirmed)
        roi.sigRegionChangeFinished.connect(lambda item=roi: self._on_roi_changed(item))
        roi.sigClicked.connect(lambda item=roi: self.select_box(item.box_id, notify=True))
        self.plot.addItem(roi)
        self._rois[box_id] = roi

    def update_box_type(self, box_id: int, burst_type: str) -> None:
        roi = self._rois.get(box_id)
        if roi is not None:
            roi.set_burst_type(burst_type)

    def remove_box(self, box_id: int) -> None:
        roi = self._rois.pop(box_id, None)
        if roi is not None:
            self.plot.removeItem(roi)
        if self._selected_id == box_id:
            self._selected_id = None

    def clear_boxes(self) -> None:
        for roi in self._rois.values():
            self.plot.removeItem(roi)
        self._rois.clear()
        self._selected_id = None

    def select_box(self, box_id: int | None, notify: bool = False) -> None:
        for identifier, roi in self._rois.items():
            roi.set_selected(identifier == box_id)
        self._selected_id = box_id
        if notify:
            self.box_selected.emit(box_id if box_id is not None else -1)

    @property
    def selected_box_id(self) -> int | None:
        return self._selected_id

    def _on_region_drawn(self, rect: QRectF) -> None:
        if self._array is None:
            return
        n_freq, n_time = self._array.shape
        row0 = int(np.clip(np.floor(rect.top()), 0, n_freq - 1))
        row1 = int(np.clip(np.ceil(rect.bottom()), row0 + 1, n_freq))
        col0 = int(np.clip(np.floor(rect.left()), 0, n_time - 1))
        col1 = int(np.clip(np.ceil(rect.right()), col0 + 1, n_time))
        self.box_created.emit(row0, row1, col0, col1)

    def _on_roi_changed(self, roi: BurstROI) -> None:
        if self._suspend_edit_signal or self._array is None:
            return
        n_freq, n_time = self._array.shape
        row0, row1, col0, col1 = roi.pixel_box()
        row0 = int(np.clip(row0, 0, n_freq - 1))
        row1 = int(np.clip(row1, row0 + 1, n_freq))
        col0 = int(np.clip(col0, 0, n_time - 1))
        col1 = int(np.clip(col1, col0 + 1, n_time))
        self.box_edited.emit(roi.box_id, row0, row1, col0, col1)

    # -- crosshair ---------------------------------------------------------

    def _on_mouse_moved(self, position: QPointF) -> None:
        if self._array is None or self._axes is None:
            return
        if not self.plot.sceneBoundingRect().contains(position):
            self._crosshair_v.setVisible(False)
            self._crosshair_h.setVisible(False)
            self.cursor_left.emit()
            return

        point = self._viewbox.mapSceneToView(position)
        n_freq, n_time = self._array.shape
        row = int(round(point.y()))
        column = int(round(point.x()))
        if not (0 <= row < n_freq and 0 <= column < n_time):
            self._crosshair_v.setVisible(False)
            self._crosshair_h.setVisible(False)
            self.cursor_left.emit()
            return

        self._crosshair_v.setPos(point.x())
        self._crosshair_h.setPos(point.y())
        self._crosshair_v.setVisible(True)
        self._crosshair_h.setVisible(True)

        self.cursor_moved.emit(self.sample_at(row, column))

    def sample_at(self, row: int, column: int) -> dict[str, Any]:
        """Every representation of one sample, for the pixel inspector.

        The three value columns are the same measurement at three stages of the
        pipeline: the receiver digit as stored, its level in dB above the
        per-frequency background, and the clipped 0-1 number the network is fed.
        Seeing them together makes it obvious when a feature is being clipped
        away by the display window.
        """
        normalized = float(self._array[row, column]) if self._array is not None else float("nan")
        raw = float(self._raw[row, column]) if self._raw is not None else None
        axes = self._axes
        return {
            "row": row,
            "column": column,
            "raw": raw,
            "normalized": normalized,
            "db": self._to_db(normalized),
            "mhz": float(axes.freq_mhz[row]) if axes is not None else float("nan"),
            "seconds": float(axes.time_s[column]) if axes is not None else float("nan"),
            "time_label": format_time_label(axes, float(axes.time_s[column])) if axes else "",
            "clipped": normalized <= 0.0 or normalized >= 1.0,
        }

    @staticmethod
    def _to_db(normalized_value: float, vmin: float = -1.0, vmax: float = 8.0) -> float:
        """Invert the db_window normalization for the readout."""
        return normalized_value * (vmax - vmin) + vmin

    # -- view helpers ------------------------------------------------------

    def reset_view(self) -> None:
        if self._array is None:
            return
        n_freq, n_time = self._array.shape
        self._viewbox.setRange(xRange=(0, n_time), yRange=(0, n_freq), padding=0.0)

    def zoom_to_box(self, row0: int, row1: int, col0: int, col1: int, margin: float = 0.4) -> None:
        col_pad = max((col1 - col0) * margin, 8)
        row_pad = max((row1 - row0) * margin, 4)
        self._viewbox.setRange(
            xRange=(col0 - col_pad, col1 + col_pad),
            yRange=(row0 - row_pad, row1 + row_pad),
            padding=0.0,
        )
