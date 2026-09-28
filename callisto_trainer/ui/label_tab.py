"""The labelling workspace: queue, spectrum canvas and label controls.

Everything the operator does here commits to SQLite immediately -- there is no
save button, because the tool has to survive being killed mid-session without
losing a label. On startup the queue position and filters are restored from the
same store, so work resumes exactly where it stopped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.core.coords import box_to_physical
from callisto_trainer.core.crops import CropConfig
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.cache import SpectrumBundle
from callisto_trainer.services.prep_worker import LoadRequest, SpectrumLoader, prefetch_window
from callisto_trainer.settings import AppSettings
from callisto_trainer.core.taxonomy import TYPE_III
from callisto_trainer.store.repository import (
    BOX_TYPES,
    BURST_TYPES,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
)
from callisto_trainer.ui.file_queue import FileQueuePanel
from callisto_trainer.ui.label_panel import LabelPanel
from callisto_trainer.ui.raw_data_panel import HeaderPanel, PixelInspector
from callisto_trainer.ui.spectrogram_view import SpectrogramView

LOGGER = get_logger(__name__)

STATE_QUEUE_FILE = "queue.current_file_id"
STATE_LEVELS = "view.levels"
STATE_COLORMAP = "view.colormap"

# Fraction of a suggestion's area that may overlap an existing box before the
# suggestion is treated as a duplicate and dropped.
SUGGESTION_OVERLAP_LIMIT = 0.3


def _overlaps(proposal, box) -> bool:
    """True when a suggestion substantially covers an already-marked region."""
    row_overlap = max(0, min(proposal.row1, box.row1) - max(proposal.row0, box.row0))
    col_overlap = max(0, min(proposal.col1, box.col1) - max(proposal.col0, box.col0))
    if row_overlap == 0 or col_overlap == 0:
        return False
    intersection = row_overlap * col_overlap
    area = (proposal.row1 - proposal.row0) * (proposal.col1 - proposal.col0)
    return area > 0 and intersection / area > SUGGESTION_OVERLAP_LIMIT


class LabelTab(QWidget):
    """Import-to-annotation workspace."""

    dataset_changed = Signal()

    def __init__(
        self,
        repository: AnnotationRepository,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository = repository
        self.settings = settings
        self.crop_config = CropConfig.from_config(settings.pipeline)

        self.loader = SpectrumLoader(
            settings.pipeline,
            cache_dir=str(settings.display_cache_dir),
            max_workers=settings.decode_workers,
            memory_budget_bytes=settings.memory_cache_bytes,
            parent=self,
        )
        self.loader.loaded.connect(self._on_bundle_loaded)
        self.loader.failed.connect(self._on_load_failed)

        self._current_file_id: int | None = None
        self._current_bundle: SpectrumBundle | None = None
        # Lazily loaded model used to label suggestions (the newest unified
        # model, else the newest type model); loading costs ~1 s so it is kept
        # for the session once obtained.
        self._scorer: Any | None = None
        # The type a new box gets when nothing in the file suggests one: the
        # last type the operator chose, so a run of files of one kind needs no
        # re-picking.
        self._sticky_type: str = TYPE_III
        # Verdicts that drawing a burst box changed implicitly, by file, so they
        # can be put back if the burst box is deleted.
        self._implicit_verdict: dict[int, str | None] = {}

        self.queue = FileQueuePanel(repository)
        self.canvas = SpectrogramView()
        self.panel = LabelPanel(self.crop_config)
        self.inspector = PixelInspector()
        self.header_panel = HeaderPanel()

        self._build_layout()
        self._connect()
        self._install_shortcuts()
        self._restore_view_state()

    # -- construction ------------------------------------------------------

    def _build_layout(self) -> None:
        self.title = QLabel("No file loaded")
        self.title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.readout = QLabel("")
        self.readout.setStyleSheet("color: #8b949e; font-family: Consolas, monospace;")

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(4, 4, 4, 4)
        centre_layout.setSpacing(4)
        centre_layout.addWidget(self.title)
        centre_layout.addWidget(self._build_toolbar())
        centre_layout.addWidget(self.canvas, 1)
        centre_layout.addWidget(self._build_footer())

        # The right column carries two views of the same file: the labelling
        # controls, and the underlying data the preprocessing was derived from.
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.panel, "Label")

        raw_page = QWidget()
        raw_layout = QVBoxLayout(raw_page)
        raw_layout.setContentsMargins(6, 6, 6, 6)
        raw_layout.setSpacing(6)
        raw_layout.addWidget(self.header_panel, 1)
        raw_layout.addWidget(self.inspector)
        self.right_tabs.addTab(raw_page, "Raw data")

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.queue)
        self.splitter.addWidget(centre)
        self.splitter.addWidget(self.right_tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([260, 900, 340])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        self.draw_mode = QCheckBox("Draw boxes (drag)")
        self.draw_mode.setChecked(True)
        self.draw_mode.setToolTip("Off: left-drag pans instead. Shift-drag always pans.")
        row.addWidget(self.draw_mode)

        self.show_rfi = QCheckBox("Show flagged RFI")
        self.show_rfi.setToolTip("Interference channels listed in the file's RFI_FREQ table")
        row.addWidget(self.show_rfi)

        row.addSpacing(12)
        row.addWidget(QLabel("View:"))
        self.view_mode = QComboBox()
        self.view_mode.addItem("Preprocessed (model view)", "normalized")
        self.view_mode.addItem("Raw FITS data", "raw")
        self.view_mode.addItem("Quiet background (long continua)", "quiet")
        self.view_mode.setToolTip(
            "Preprocessed: background-subtracted and mapped to the -1..8 dB window "
            "the model is trained on.\n"
            "Raw: the array exactly as stored in the file, uncalibrated receiver "
            "digits with no background removed.\n"
            "Quiet background: each channel's background taken from its quietest "
            "tenth instead of its median, on the same dB window, so a continuum "
            "lasting most of the file (Type IV) stays bright. The model sees it too.\n\n"
            "Boxes and zoom are unaffected by switching. V cycles through the views."
        )
        row.addWidget(self.view_mode)

        row.addSpacing(12)
        row.addWidget(QLabel("Contrast:"))
        self.level_low = QDoubleSpinBox()
        self.level_high = QDoubleSpinBox()
        for spin, value in ((self.level_low, 0.0), (self.level_high, 1.0)):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setFixedWidth(64)
            row.addWidget(spin)

        self.reset_levels = QPushButton("Model view")
        self.reset_levels.setToolTip("Reset contrast to exactly what the model receives")
        row.addWidget(self.reset_levels)

        self.level_warning = QLabel("")
        self.level_warning.setStyleSheet("color: #d4a72c; font-weight: bold;")
        row.addWidget(self.level_warning)

        row.addSpacing(12)
        self.colormap = QComboBox()
        self.colormap.addItems(["inferno", "viridis", "magma", "plasma", "CET-L2", "grey"])
        row.addWidget(self.colormap)

        self.reset_view = QPushButton("Reset zoom (R)")
        row.addWidget(self.reset_view)

        row.addSpacing(12)
        self.propose_button = QPushButton("Suggest bursts (P)")
        self.propose_button.setToolTip(
            "Finds bright connected regions and, if a burst-type model has been "
            "trained, labels each one with its prediction.\n\n"
            "This is a brightness heuristic, not a trained detector: it will also "
            "flag RFI and instrument artifacts. Suggestions are drawn dashed and "
            "do not count as labels until you give one a type."
        )
        row.addWidget(self.propose_button)
        row.addStretch(1)
        return bar

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        row = QHBoxLayout(footer)
        row.setContentsMargins(0, 0, 0, 0)

        self.prev_button = QPushButton("◀ Previous (A)")
        self.next_button = QPushButton("Next (D) ▶")
        self.next_unreviewed = QPushButton("Next unreviewed (F)")
        for button in (self.prev_button, self.next_button, self.next_unreviewed):
            row.addWidget(button)

        row.addSpacing(16)
        row.addWidget(self.readout, 1)

        self.autosave = QLabel("Saved")
        self.autosave.setStyleSheet("color: #3fb950;")
        row.addWidget(self.autosave)
        return footer

    def _connect(self) -> None:
        self.queue.file_activated.connect(self.load_file)
        self.canvas.box_created.connect(self._on_box_created)
        self.canvas.box_edited.connect(self._on_box_edited)
        self.canvas.box_selected.connect(self._on_canvas_box_selected)
        self.canvas.cursor_moved.connect(self._on_cursor_moved)

        self.panel.verdict_changed.connect(self._on_verdict_changed)
        self.panel.type_assigned.connect(self._on_type_assigned)
        self.panel.box_selected.connect(self._on_panel_box_selected)
        self.panel.box_delete_requested.connect(self._on_box_deleted)
        self.panel.notes_changed.connect(self._on_notes_changed)

        self.draw_mode.toggled.connect(self.canvas.set_draw_mode)
        self.show_rfi.toggled.connect(self.canvas.set_rfi_visible)
        self.view_mode.currentIndexChanged.connect(self._on_view_mode_changed)
        self.canvas.cursor_left.connect(self.inspector.clear)
        self.level_low.valueChanged.connect(self._on_levels_changed)
        self.level_high.valueChanged.connect(self._on_levels_changed)
        self.reset_levels.clicked.connect(self._reset_levels)
        self.colormap.currentTextChanged.connect(self._on_colormap_changed)
        self.reset_view.clicked.connect(self.canvas.reset_view)
        self.propose_button.clicked.connect(self.propose_boxes)

        self.prev_button.clicked.connect(lambda: self.queue.step(-1))
        self.next_button.clicked.connect(lambda: self.queue.step(1))
        self.next_unreviewed.clicked.connect(self.queue.next_unreviewed)

    def _install_shortcuts(self) -> None:
        def bind(sequence: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)

        bind("A", lambda: self.queue.step(-1))
        bind("D", lambda: self.queue.step(1))
        bind("Left", lambda: self.queue.step(-1))
        bind("Right", lambda: self.queue.step(1))
        bind("F", self.queue.next_unreviewed)
        bind("B", lambda: self._on_verdict_changed(VERDICT_BURST))
        bind("N", lambda: self._on_verdict_changed(VERDICT_NO_BURST))
        bind("U", lambda: self._on_verdict_changed("unsure"))
        bind("R", self.canvas.reset_view)
        bind("P", self.propose_boxes)
        bind("V", self._toggle_view_mode)
        bind("Del", self._delete_selected_box)
        # 1 Type II, 2 Type III, 3 Type IIIG, 4 Type IV, 5 Other.
        for index, box_type in enumerate(BOX_TYPES, start=1):
            bind(str(index), lambda t=box_type: self._on_type_assigned(t))

    # -- queue / loading ---------------------------------------------------

    def refresh_queue(self, keep_selection: bool = True) -> None:
        self.queue.refresh_stations()
        self.queue.reload(self._current_file_id if keep_selection else None)

    def restore_session(self) -> None:
        """Reopen the file the operator was last working on."""
        self.refresh_queue(keep_selection=False)
        stored = self.repository.get_state(STATE_QUEUE_FILE)
        if stored and stored.isdigit() and self.queue.select_file(int(stored)):
            return
        self.queue.select_row(0)

    def load_file(self, file_id: int) -> None:
        record = self.repository.get_file(file_id)
        if record is None:
            return

        self._current_file_id = file_id
        self.repository.set_state(STATE_QUEUE_FILE, str(file_id))
        self._update_title(record)

        bundle = self.loader.request(LoadRequest(file_id, record.path, record.content_hash))
        if bundle is not None:
            self._display(bundle)
        else:
            self.canvas.clear()
            self.readout.setText("Loading...")
        self._queue_prefetch()

    def _queue_prefetch(self) -> None:
        ids = self.queue.model.file_ids
        row = self.queue.current_row()
        if row < 0:
            return
        requests = []
        for file_id in prefetch_window(ids, row, ahead=self.settings.prefetch_ahead):
            record = self.repository.get_file(file_id)
            if record and record.status != "error":
                requests.append(LoadRequest(file_id, record.path, record.content_hash))
        self.loader.prefetch(requests)

    def _on_bundle_loaded(self, bundle: SpectrumBundle) -> None:
        if bundle.file_id == self._current_file_id:
            self._display(bundle)

    def _on_load_failed(self, file_id: int, message: str) -> None:
        self.repository.set_error(file_id, message)
        self.queue.refresh_file(file_id)
        if file_id == self._current_file_id:
            self.canvas.clear()
            self.readout.setText(f"Could not read this file: {message}")

    def _display(self, bundle: SpectrumBundle) -> None:
        self._current_bundle = bundle
        record = self.repository.get_file(bundle.file_id)
        self.canvas.set_spectrum(
            bundle.normalized,
            bundle.axes,
            rfi_channels=bundle.metadata.get("rfi_channels_mhz"),
            raw=bundle.raw,
            raw_levels=bundle.raw_levels(),
            quiet=bundle.quiet,
        )
        self.canvas.set_rfi_visible(self.show_rfi.isChecked())
        # Keep whichever view the operator chose when moving between files.
        self.canvas.set_view_mode(self.view_mode.currentData() or "normalized")
        self.header_panel.show_header(bundle.path, bundle.header_text)
        self.inspector.clear()

        boxes = self.repository.boxes_for_file(bundle.file_id)
        # Boxes measured by an earlier method (a pixel fit) are recalculated from
        # their geometry the first time their file is opened.
        from callisto_trainer.core.burst_physics import box_parameters_current

        stale = [box for box in boxes if not box_parameters_current(box.physics, box.burst_type)]
        if stale:
            for box in stale:
                self._measure_physics(box.id)
            boxes = self.repository.boxes_for_file(bundle.file_id)
        for box in boxes:
            self.canvas.add_box(
                box.id, box.row0, box.row1, box.col0, box.col1, box.burst_type, box.confirmed
            )
        self.panel.set_file(
            bundle.normalized,
            bundle.axes,
            record.verdict if record else None,
            record.notes if record else "",
            boxes,
            quiet=bundle.quiet,
        )
        if boxes:
            self.canvas.select_box(boxes[-1].id)
        self._update_title(record)
        # Clear the "Loading..." placeholder; the readout fills in on hover.
        self.readout.setText(
            "Drag on the spectrum to mark a burst   ·   hover for time, frequency and dB"
        )

    def _update_title(self, record) -> None:
        if record is None:
            self.title.setText("No file loaded")
            return
        pieces = [record.file_name]
        if record.station:
            pieces.append(record.station)
        if record.obs_date:
            pieces.append(f"{record.obs_date} {record.obs_time or ''}".strip())
        if record.n_freq and record.n_time:
            pieces.append(f"{record.n_freq}x{record.n_time}")
        if record.freq_axis_is_approximate:
            pieces.append("frequency axis approximate (no AXES table)")
        row = self.queue.current_row()
        if row >= 0:
            pieces.append(f"{row + 1} of {self.queue.model.rowCount():,}")
        self.title.setText("   ·   ".join(pieces))

    # -- annotation --------------------------------------------------------

    def _measure_physics(self, box_id: int) -> None:
        """Measure and store the burst parameters for one box."""
        if self._current_bundle is None:
            return
        from callisto_trainer.services.physics_service import measure_and_store

        box = next(
            (b for b in self.repository.boxes_for_file(self._current_file_id) if b.id == box_id),
            None,
        )
        if box is not None:
            measure_and_store(
                self.repository, self._current_bundle.normalized, self._current_bundle.axes, box
            )

    def _show_physics_for(self, box_id: int | None) -> None:
        if box_id is None or self._current_file_id is None:
            self.panel.show_physics(None, None)
            return
        from callisto_trainer.core.burst_physics import physics_from_row

        box = next(
            (b for b in self.repository.boxes_for_file(self._current_file_id) if b.id == box_id),
            None,
        )
        if box is None:
            self.panel.show_physics(None, None)
            return
        self.panel.show_physics(physics_from_row(box.physics), box.burst_type)

    def _physical_for(self, row0: int, row1: int, col0: int, col1: int) -> dict[str, float]:
        if self._current_bundle is None:
            return {}
        return box_to_physical(self._current_bundle.axes, row0, row1, col0, col1)

    def _default_box_type(self) -> str:
        """The type a freshly drawn box starts with.

        The type of the last box in the file -- almost always right when marking
        several bursts of one kind -- and on a file with none, the last type
        chosen this session.
        """
        boxes = self.repository.boxes_for_file(self._current_file_id)
        return boxes[-1].burst_type if boxes else self._sticky_type

    def _sync_verdict_with_boxes(self) -> None:
        """Keep the file's verdict consistent with the boxes on it.

        Drawing a burst is an implicit statement that the file has one, so a
        confirmed burst box makes the verdict Burst, as drawing one always has.
        If the only burst box on a file is deleted, a verdict that was set
        implicitly goes back to what it was. Pressing B / N / U is an explicit
        decision and forgets what the implicit change replaced.
        """
        file_id = self._current_file_id
        record = self.repository.get_file(file_id) if file_id is not None else None
        if record is None:
            return
        has_burst = any(
            box.confirmed and box.burst_type in BURST_TYPES
            for box in self.repository.boxes_for_file(file_id)
        )
        if has_burst and record.verdict != VERDICT_BURST:
            self._implicit_verdict.setdefault(file_id, record.verdict)
            self.repository.set_verdict(file_id, VERDICT_BURST)
            self.panel.set_verdict(VERDICT_BURST)
        elif not has_burst and record.verdict == VERDICT_BURST and file_id in self._implicit_verdict:
            previous = self._implicit_verdict.pop(file_id)
            self.repository.set_verdict(file_id, previous)
            self.panel.set_verdict(previous)

    def _on_box_created(self, row0: int, row1: int, col0: int, col1: int) -> None:
        if self._current_file_id is None:
            return
        box_type = self._default_box_type()

        box_id = self.repository.add_box(
            self._current_file_id,
            row0, row1, col0, col1,
            box_type,
            physical=self._physical_for(row0, row1, col0, col1),
        )
        self._sync_verdict_with_boxes()

        self._measure_physics(box_id)
        self.canvas.add_box(box_id, row0, row1, col0, col1, box_type)
        self.canvas.select_box(box_id)
        self._reload_boxes(select_id=box_id)
        self._show_physics_for(box_id)
        self._mark_saved()

    def _on_box_edited(self, box_id: int, row0: int, row1: int, col0: int, col1: int) -> None:
        self.repository.update_box(
            box_id,
            row0=row0, row1=row1, col0=col0, col1=col1,
            physical=self._physical_for(row0, row1, col0, col1),
        )
        # The box *is* the measurement: a new height or width is a new
        # frequency range, duration and drift rate.
        self._measure_physics(box_id)
        self._reload_boxes(select_id=box_id)
        self._show_physics_for(box_id)
        self._mark_saved()

    def _on_box_deleted(self, box_id: int) -> None:
        self.repository.delete_box(box_id)
        self.canvas.remove_box(box_id)
        self._sync_verdict_with_boxes()
        self._reload_boxes()
        self._mark_saved()

    def _delete_selected_box(self) -> None:
        box_id = self.panel.current_box_id() or self.canvas.selected_box_id
        if box_id is not None:
            self._on_box_deleted(box_id)

    def _on_type_assigned(self, burst_type: str) -> None:
        box_id = self.panel.current_box_id() or self.canvas.selected_box_id
        # Only burst types are drawn; interference is found automatically.
        if box_id is None or burst_type not in BOX_TYPES:
            return
        # Typing a suggestion is the act of accepting it: the operator has looked
        # at the region and committed to what it is.
        self.repository.update_box(box_id, burst_type=burst_type, confirmed=True)
        # Parameters depend on the type (Type II and III only), so recalculate.
        self._measure_physics(box_id)
        self._sticky_type = burst_type
        self._sync_verdict_with_boxes()
        self.canvas.update_box_type(box_id, burst_type)
        self._reload_boxes(select_id=box_id)
        self._show_physics_for(box_id)
        self._refresh_canvas_boxes()
        self._mark_saved()

    def _refresh_canvas_boxes(self) -> None:
        """Redraw ROIs so confirmed/proposed styling stays in step with the store."""
        if self._current_file_id is None:
            return
        selected = self.canvas.selected_box_id
        self.canvas.clear_boxes()
        for box in self.repository.boxes_for_file(self._current_file_id):
            self.canvas.add_box(
                box.id, box.row0, box.row1, box.col0, box.col1, box.burst_type, box.confirmed
            )
        if selected is not None:
            self.canvas.select_box(selected)

    # -- assisted pre-labelling -------------------------------------------

    def _load_suggestion_model(self) -> Any | None:
        """The newest unified model, else the newest type model, else nothing."""
        from callisto_trainer.core.inference import CascadePredictor
        from callisto_trainer.services.assist import CheckpointScorer, find_latest_checkpoint

        unified = find_latest_checkpoint(self.settings.outputs_dir, "unified")
        if unified is not None:
            try:
                model = CascadePredictor(self.settings.pipeline, unified_checkpoint=unified)
                LOGGER.info("Loaded unified checkpoint for suggestions: %s", unified)
                return model
            except Exception as exc:
                LOGGER.warning("Could not load unified checkpoint %s: %r", unified, exc)
        checkpoint = find_latest_checkpoint(self.settings.outputs_dir, "type")
        if checkpoint is not None:
            try:
                scorer = CheckpointScorer(checkpoint)
                LOGGER.info("Loaded type checkpoint for suggestions: %s", checkpoint)
                return scorer
            except Exception as exc:
                LOGGER.warning("Could not load type checkpoint %s: %r", checkpoint, exc)
        return None

    def _unified_proposals(self) -> tuple[list[Any], int]:
        """Regions the unified model calls a burst, and how many it rejected."""
        from callisto_trainer.core.region_finder import Proposal

        bundle = self._current_bundle
        regions = self._scorer.examine(
            bundle.normalized, bundle.axes, rfi_channels=bundle.metadata.get("rfi_channels_mhz"),
            quiet=self._scorer.quiet_for(bundle.raw),
        )
        proposals, rejected = [], 0
        for region in regions:
            if self._scorer.decide(region):
                proposals.append(
                    Proposal(
                        region.row0, region.row1, region.col0, region.col1,
                        area=region.area, peak=region.peak,
                        burst_type=region.burst_type, probability=region.type_confidence,
                    )
                )
            else:
                rejected += 1
        return proposals, rejected

    def propose_boxes(self) -> None:
        """Suggest candidate regions for the current file.

        Honest about what this is: a brightness heuristic for *where*, plus a
        trained model for *what*. With a unified model, suggestions are the
        regions it calls a burst; what it calls background or interference is
        left out. Suggestions land unconfirmed and are drawn dashed.
        """
        from callisto_trainer.core.inference import CascadePredictor
        from callisto_trainer.services.assist import propose_from_normalized

        if self._current_file_id is None or self._current_bundle is None:
            return

        if self._scorer is None:
            self._scorer = self._load_suggestion_model()

        from callisto_trainer.core.inference import resolve_threshold

        rejected = 0
        if isinstance(self._scorer, CascadePredictor):
            proposals, rejected = self._unified_proposals()
        else:
            # Same adaptive threshold the Predict tab uses, so a suggestion here
            # and a prediction there see the same regions.
            proposals = propose_from_normalized(
                self._current_bundle.normalized,
                self.settings.pipeline,
                scorer=self._scorer,
                threshold=resolve_threshold(self._current_bundle.normalized),
            )
        if not proposals:
            self.readout.setText(
                f"The model called all {rejected} candidate region(s) background or interference."
                if rejected
                else "No candidate regions found above the brightness threshold."
            )
            return

        existing = self.repository.boxes_for_file(self._current_file_id)
        added = 0
        for proposal in proposals:
            if any(_overlaps(proposal, box) for box in existing):
                continue
            box_type = proposal.burst_type if proposal.burst_type in BOX_TYPES else TYPE_III
            box_id = self.repository.add_box(
                self._current_file_id,
                proposal.row0, proposal.row1, proposal.col0, proposal.col1,
                box_type,
                physical=self._physical_for(
                    proposal.row0, proposal.row1, proposal.col0, proposal.col1
                ),
                source="assisted",
                confirmed=False,
            )
            self._measure_physics(box_id)
            added += 1

        self._refresh_canvas_boxes()
        self._reload_boxes()
        typed = "typed by the model" if self._scorer else "untyped (no model trained yet)"
        skipped = (
            f" {rejected} region(s) it called background or interference were left out."
            if rejected else ""
        )
        self.readout.setText(
            f"{added} suggestion(s) added, {typed}.{skipped} They are dashed and are not "
            "counted as labels until you give each one a type."
        )

    def _on_verdict_changed(self, verdict: str) -> None:
        if self._current_file_id is None:
            return

        boxes = self.repository.boxes_for_file(self._current_file_id)
        bursts = [box for box in boxes if box.burst_type in BURST_TYPES]
        if verdict == VERDICT_NO_BURST and bursts:
            # Burst boxes on a no_burst file would be silently dropped at export
            # time. Make that trade-off explicit rather than losing the work
            # quietly.
            choice = QMessageBox.question(
                self,
                "Discard marked bursts?",
                f"This file has {len(bursts)} marked burst(s), but you are marking it "
                "as containing no burst.\n\nOnly bursts on files marked 'Burst' are "
                "exported for training.\n\nDelete the marked bursts and continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if choice != QMessageBox.StandardButton.Yes:
                self.panel.set_verdict(self.repository.get_file(self._current_file_id).verdict)
                return
            self.repository.delete_burst_boxes_for_file(self._current_file_id)
            self._refresh_canvas_boxes()

        self._implicit_verdict.pop(self._current_file_id, None)
        self.repository.set_verdict(self._current_file_id, verdict)
        self.panel.set_verdict(verdict)
        self._reload_boxes()
        self._mark_saved()
        self.dataset_changed.emit()

    def _on_notes_changed(self, text: str) -> None:
        if self._current_file_id is not None:
            self.repository.set_notes(self._current_file_id, text)
            self._mark_saved()

    def _reload_boxes(self, select_id: int | None = None) -> None:
        if self._current_file_id is None:
            return
        boxes = self.repository.boxes_for_file(self._current_file_id)
        self.panel.set_boxes(boxes, select_id=select_id)
        self.queue.refresh_file(self._current_file_id)
        self.dataset_changed.emit()

    def _on_cursor_moved(self, sample: dict) -> None:
        self.inspector.show_sample(sample)
        raw = sample.get("raw")
        raw_text = "" if raw is None else f"   raw {raw:.0f}"
        self.readout.setText(
            f"{sample['time_label']}   {sample['mhz']:.2f} MHz   "
            f"{sample['db']:+.2f} dB{raw_text}   (row {sample['row']}, col {sample['column']})"
        )

    def _on_view_mode_changed(self) -> None:
        mode = self.view_mode.currentData() or "normalized"
        self.canvas.set_view_mode(mode)
        showing_raw = mode == "raw"
        # Contrast controls act on the model's dB window, which has no meaning
        # for uncalibrated raw digits.
        for widget in (self.level_low, self.level_high, self.reset_levels):
            widget.setEnabled(not showing_raw)
        self.level_warning.setText(
            "⚠ raw data, not the model's view"
            if showing_raw
            else ("" if self.canvas.levels_match_model else "⚠ not the model's view")
        )

    def _on_canvas_box_selected(self, box_id: int) -> None:
        self.panel.select_box(box_id if box_id >= 0 else None)
        self._show_physics_for(box_id if box_id >= 0 else None)

    def _on_panel_box_selected(self, box_id: int) -> None:
        self.canvas.select_box(box_id if box_id >= 0 else None)
        self._show_physics_for(box_id if box_id >= 0 else None)

    # -- view state --------------------------------------------------------

    def _on_levels_changed(self) -> None:
        low, high = self.level_low.value(), self.level_high.value()
        if high <= low:
            return
        self.canvas.set_levels(low, high)
        self.repository.set_state(STATE_LEVELS, f"{low},{high}")
        self.level_warning.setText(
            "" if self.canvas.levels_match_model else "⚠ not the model's view"
        )

    def _reset_levels(self) -> None:
        self.level_low.setValue(0.0)
        self.level_high.setValue(1.0)

    def _toggle_view_mode(self) -> None:
        self.view_mode.setCurrentIndex((self.view_mode.currentIndex() + 1) % self.view_mode.count())

    def _on_colormap_changed(self, name: str) -> None:
        self.canvas.set_colormap(name)
        self.repository.set_state(STATE_COLORMAP, name)

    def _restore_view_state(self) -> None:
        stored = self.repository.get_state(STATE_LEVELS)
        if stored:
            try:
                low, high = (float(part) for part in stored.split(","))
                self.level_low.setValue(low)
                self.level_high.setValue(high)
            except ValueError:
                pass
        colormap = self.repository.get_state(STATE_COLORMAP)
        if colormap:
            index = self.colormap.findText(colormap)
            if index >= 0:
                self.colormap.setCurrentIndex(index)

    def _mark_saved(self) -> None:
        self.autosave.setText("Saved")

    # -- reset -------------------------------------------------------------

    def reset_view_and_layout(self) -> None:
        """Restore display defaults. Never touches labels."""
        self.view_mode.setCurrentIndex(0)
        self._reset_levels()
        self.colormap.setCurrentIndex(0)
        self.draw_mode.setChecked(True)
        self.show_rfi.setChecked(False)
        self.canvas.set_colormap(self.colormap.currentText())
        self.canvas.reset_view()
        self.splitter.setSizes([260, 900, 340])
        self.right_tabs.setCurrentIndex(0)

        self.queue.status_filter.setCurrentIndex(0)
        self.queue.station_filter.setCurrentIndex(0)
        self.queue.order_by.setCurrentIndex(0)
        self.queue.search.clear()

        for key in (STATE_LEVELS, STATE_COLORMAP):
            self.repository.set_state(key, "")
        self.readout.setText("View and layout reset to defaults.")

    def clear_current_file_labels(self) -> bool:
        """Drop every box and the verdict for the file on screen."""
        if self._current_file_id is None:
            return False
        self.repository.clear_file_labels(self._current_file_id)
        self.canvas.clear_boxes()
        self.panel.set_verdict(None)
        self.panel.set_boxes([])
        # set_notes, not setPlainText: the latter would echo an empty string
        # straight back into the row we just cleared.
        self.panel.set_notes("")
        self.queue.refresh_file(self._current_file_id)
        self.dataset_changed.emit()
        self.readout.setText("Labels cleared for this file; it is unreviewed again.")
        return True

    def reload_after_reset(self) -> None:
        """Return to an empty state after the whole dataset was deleted."""
        self._current_file_id = None
        self._current_bundle = None
        self._scorer = None
        self.loader.cache.clear()
        self.canvas.clear()
        self.panel.set_file(None, None, None, "", [])
        self.header_panel.clear()
        self.inspector.clear()
        self.title.setText("No file loaded")
        self.readout.setText("")
        self.refresh_queue(keep_selection=False)

    def shutdown(self) -> None:
        self.loader.shutdown()
