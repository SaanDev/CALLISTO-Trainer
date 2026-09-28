"""The review queue: a virtualized list over potentially 100k+ files.

The model holds only integer ids. Row content is fetched from SQLite on demand
inside ``data()``, which Qt calls for visible rows only, with a small LRU so
repaints do not re-query. That keeps memory flat and startup instant no matter
how large the imported archive is.
"""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QVBoxLayout,
    QWidget,
)

from callisto_trainer.store.repository import (
    STATUS_ERROR,
    STATUS_LABELED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    VERDICT_UNSURE,
    AnnotationRepository,
    FileRecord,
    QueueFilter,
)

STATUS_COLORS = {
    STATUS_PENDING: "#7d8590",
    "in_review": "#d4a72c",
    STATUS_LABELED: "#3fb950",
    STATUS_SKIPPED: "#6e7681",
    STATUS_ERROR: "#f85149",
}

VERDICT_MARKS = {
    VERDICT_BURST: "●",      # filled circle
    VERDICT_NO_BURST: "○",   # hollow circle
    VERDICT_UNSURE: "?",
}

FileIdRole = Qt.ItemDataRole.UserRole + 1


class FileQueueModel(QAbstractListModel):
    """List model backed by the annotation store."""

    def __init__(self, repository: AnnotationRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self._ids: list[int] = []
        self._cache: OrderedDict[int, FileRecord] = OrderedDict()
        self._cache_limit = 512

    # -- Qt model API ------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._ids)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._ids)):
            return None

        file_id = self._ids[index.row()]
        if role == FileIdRole:
            return file_id

        record = self._record(file_id)
        if record is None:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            mark = VERDICT_MARKS.get(record.verdict or "", " ")
            boxes = f"  [{record.box_count}]" if record.box_count else ""
            return f"{mark} {record.file_name}{boxes}"
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor(STATUS_COLORS.get(record.status, "#c8d0d8"))
        if role == Qt.ItemDataRole.FontRole and record.status == STATUS_PENDING:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [
                record.path,
                f"Station: {record.station or 'unknown'}",
                f"Date: {record.obs_date or '?'} {record.obs_time or ''}",
                f"Status: {record.status}   Verdict: {record.verdict or '-'}",
                f"Boxes: {record.box_count}",
            ]
            if record.freq_min_mhz is not None:
                approx = " (approximate)" if record.freq_axis_is_approximate else ""
                lines.append(
                    f"Frequency: {record.freq_min_mhz:.2f}-{record.freq_max_mhz:.2f} MHz{approx}"
                )
            if record.burst_probability is not None:
                lines.append(f"Predicted burst probability: {record.burst_probability:.3f}")
            if record.error:
                lines.append(f"Error: {record.error}")
            return "\n".join(lines)
        return None

    # -- population --------------------------------------------------------

    def set_filter(self, filters: QueueFilter) -> None:
        self.beginResetModel()
        self._ids = self.repository.file_ids(filters)
        self._cache.clear()
        self.endResetModel()

    def refresh_file(self, file_id: int) -> None:
        """Invalidate one row after an edit so its status/box count redraws."""
        self._cache.pop(file_id, None)
        try:
            row = self._ids.index(file_id)
        except ValueError:
            return
        index = self.index(row, 0)
        self.dataChanged.emit(index, index)

    def _record(self, file_id: int) -> FileRecord | None:
        record = self._cache.get(file_id)
        if record is not None:
            self._cache.move_to_end(file_id)
            return record

        record = self.repository.get_file(file_id)
        if record is not None:
            self._cache[file_id] = record
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return record

    # -- lookups -----------------------------------------------------------

    @property
    def file_ids(self) -> list[int]:
        return self._ids

    def row_of(self, file_id: int) -> int:
        try:
            return self._ids.index(file_id)
        except ValueError:
            return -1

    def id_at(self, row: int) -> int | None:
        return self._ids[row] if 0 <= row < len(self._ids) else None


class FileQueuePanel(QWidget):
    """Queue list plus its filter controls."""

    file_activated = Signal(int)  # file_id

    def __init__(self, repository: AnnotationRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repository = repository
        self.model = FileQueueModel(repository, self)

        self.status_filter = QComboBox()
        self.status_filter.addItem("All files", [])
        self.status_filter.addItem("Unreviewed", [STATUS_PENDING, "in_review"])
        self.status_filter.addItem("Labeled", [STATUS_LABELED])
        self.status_filter.addItem("Skipped", [STATUS_SKIPPED])
        self.status_filter.addItem("Errors", [STATUS_ERROR])

        self.station_filter = QComboBox()
        self.station_filter.addItem("All stations", "")

        self.order_by = QComboBox()
        self.order_by.addItem("Name", "name")
        self.order_by.addItem("Import order", "imported")
        self.order_by.addItem("Burst probability", "probability")

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search file name...")
        self.search.setClearButtonEnabled(True)

        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setUniformItemSizes(True)  # required for smooth 100k-row scrolling
        self.view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)

        self.summary = QLabel("No files imported")
        self.summary.setStyleSheet("color: #8b949e; padding: 2px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(self.status_filter, 1)
        row.addWidget(self.station_filter, 1)
        layout.addLayout(row)
        layout.addWidget(self.order_by)
        layout.addWidget(self.search)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.summary)

        self.status_filter.currentIndexChanged.connect(self.reload)
        self.station_filter.currentIndexChanged.connect(self.reload)
        self.order_by.currentIndexChanged.connect(self.reload)
        self.search.textChanged.connect(self.reload)
        self.view.selectionModel().currentChanged.connect(self._on_current_changed)

    # -- data --------------------------------------------------------------

    def current_filter(self) -> QueueFilter:
        station = self.station_filter.currentData() or ""
        return QueueFilter(
            statuses=self.status_filter.currentData() or [],
            stations=[station] if station else [],
            search=self.search.text().strip(),
            order_by=self.order_by.currentData() or "name",
        )

    def reload(self, keep_file_id: int | None = None) -> None:
        keep_file_id = keep_file_id if isinstance(keep_file_id, int) else self.current_file_id()
        self.model.set_filter(self.current_filter())
        self.refresh_summary()
        if keep_file_id is not None:
            self.select_file(keep_file_id)

    def refresh_stations(self) -> None:
        current = self.station_filter.currentData()
        self.station_filter.blockSignals(True)
        self.station_filter.clear()
        self.station_filter.addItem("All stations", "")
        for station in self.repository.stations():
            self.station_filter.addItem(station, station)
        index = self.station_filter.findData(current)
        self.station_filter.setCurrentIndex(max(0, index))
        self.station_filter.blockSignals(False)

    def refresh_summary(self) -> None:
        counts = self.repository.status_counts()
        total = sum(counts.values())
        labeled = counts.get(STATUS_LABELED, 0)
        boxes = self.repository.total_boxes()
        showing = self.model.rowCount()
        self.summary.setText(
            f"{showing:,} shown · {labeled:,}/{total:,} reviewed · {boxes:,} bursts"
        )

    def refresh_file(self, file_id: int) -> None:
        self.model.refresh_file(file_id)
        self.refresh_summary()

    # -- selection ---------------------------------------------------------

    def current_file_id(self) -> int | None:
        index = self.view.currentIndex()
        return self.model.id_at(index.row()) if index.isValid() else None

    def current_row(self) -> int:
        index = self.view.currentIndex()
        return index.row() if index.isValid() else -1

    def select_file(self, file_id: int) -> bool:
        row = self.model.row_of(file_id)
        if row < 0:
            return False
        self.select_row(row)
        return True

    def select_row(self, row: int) -> None:
        if 0 <= row < self.model.rowCount():
            index = self.model.index(row, 0)
            self.view.setCurrentIndex(index)
            self.view.scrollTo(index, QListView.ScrollHint.EnsureVisible)

    def step(self, delta: int) -> None:
        row = self.current_row()
        if row < 0:
            self.select_row(0)
            return
        self.select_row(max(0, min(row + delta, self.model.rowCount() - 1)))

    def next_unreviewed(self) -> None:
        """Jump to the next file that still needs a decision."""
        start = self.current_row() + 1
        for row in range(start, self.model.rowCount()):
            file_id = self.model.id_at(row)
            record = self.repository.get_file(file_id) if file_id else None
            if record and not record.is_reviewed:
                self.select_row(row)
                return

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        file_id = self.model.id_at(current.row()) if current.isValid() else None
        if file_id is not None:
            self.file_activated.emit(file_id)
