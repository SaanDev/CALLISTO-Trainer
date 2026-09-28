"""Typed access to the annotation store.

Every mutation commits immediately. That is deliberate: the app must be able to
die at any instant without losing a label, so there is no "save" step and no
in-memory buffer of pending edits.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from callisto_trainer.core.taxonomy import BOX_TYPES, BURST_TYPES  # noqa: F401 (re-exported)
from callisto_trainer.store.db import PHYSICS_COLUMNS, Database

# Verdicts a file can carry.
VERDICT_BURST = "burst"
VERDICT_NO_BURST = "no_burst"
VERDICT_UNSURE = "unsure"
VERDICTS = (VERDICT_BURST, VERDICT_NO_BURST, VERDICT_UNSURE)

# Review lifecycle.
STATUS_PENDING = "pending"
STATUS_IN_REVIEW = "in_review"
STATUS_LABELED = "labeled"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"

# BURST_TYPES and BOX_TYPES are defined in core/taxonomy.py and imported above
# so the callers that have always imported them from the store still can.

_HASH_BLOCK = 256 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(path: str | Path) -> str:
    """Cheap content fingerprint: size plus the head and tail of the file.

    Hashing whole gzipped spectra across an archive this size would dominate
    import time. Size plus 512 KB of content is more than enough to detect the
    duplicate imports and moved files this is used for.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.blake2b(str(size).encode("ascii"), digest_size=16)
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_BLOCK))
        if size > 2 * _HASH_BLOCK:
            handle.seek(-_HASH_BLOCK, 2)
            digest.update(handle.read(_HASH_BLOCK))
    return digest.hexdigest()


@dataclass
class FileRecord:
    """One imported FITS file."""

    id: int
    path: str
    file_name: str
    content_hash: str | None = None
    station: str | None = None
    obs_date: str | None = None
    obs_time: str | None = None
    n_freq: int | None = None
    n_time: int | None = None
    freq_min_mhz: float | None = None
    freq_max_mhz: float | None = None
    freq_axis_source: str = "none"
    legacy_freq_min_mhz: float | None = None
    legacy_freq_max_mhz: float | None = None
    cadence_s: float | None = None
    duration_s: float | None = None
    status: str = STATUS_PENDING
    verdict: str | None = None
    burst_probability: float | None = None
    notes: str | None = None
    error: str | None = None
    imported_at: str = ""
    labeled_at: str | None = None
    box_count: int = 0

    @property
    def is_reviewed(self) -> bool:
        return self.status in (STATUS_LABELED, STATUS_SKIPPED)

    @property
    def freq_axis_is_approximate(self) -> bool:
        return self.freq_axis_source != "axes_table"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FileRecord":
        keys = set(row.keys())
        return cls(
            **{f.name: row[f.name] for f in _FILE_FIELDS if f.name in keys},
        )


_FILE_FIELDS = [f for f in FileRecord.__dataclass_fields__.values()]


@dataclass
class BoxRecord:
    """One labelled burst region within a file."""

    id: int
    file_id: int
    row0: int
    row1: int
    col0: int
    col1: int
    burst_type: str
    freq_lo_mhz: float | None = None
    freq_hi_mhz: float | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    confidence: str = "certain"
    source: str = "manual"
    confirmed: bool = True
    created_at: str = ""
    updated_at: str = ""
    # Measured burst physics (schema v2). None until measured.
    physics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "BoxRecord":
        keys = set(row.keys())
        return cls(
            id=row["id"],
            file_id=row["file_id"],
            row0=row["row0"],
            row1=row["row1"],
            col0=row["col0"],
            col1=row["col1"],
            burst_type=row["burst_type"],
            freq_lo_mhz=row["freq_lo_mhz"],
            freq_hi_mhz=row["freq_hi_mhz"],
            t_start_s=row["t_start_s"],
            t_end_s=row["t_end_s"],
            confidence=row["confidence"],
            source=row["source"],
            confirmed=bool(row["confirmed"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            physics={
                name: row[name] for name, _type in PHYSICS_COLUMNS if name in keys
            },
        )

    @property
    def drift_mhz_per_s(self) -> float | None:
        return self.physics.get("drift_mhz_per_s")

    @property
    def physics_confidence(self) -> str:
        return str(self.physics.get("physics_confidence") or "none")


@dataclass
class QueueFilter:
    """Filters applied to the review queue."""

    statuses: Sequence[str] = ()
    verdicts: Sequence[str] = ()
    stations: Sequence[str] = ()
    search: str = ""
    order_by: str = "name"  # name | imported | probability

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.statuses:
            clauses.append(f"status IN ({','.join('?' * len(self.statuses))})")
            params.extend(self.statuses)
        if self.verdicts:
            clauses.append(f"verdict IN ({','.join('?' * len(self.verdicts))})")
            params.extend(self.verdicts)
        if self.stations:
            clauses.append(f"station IN ({','.join('?' * len(self.stations))})")
            params.extend(self.stations)
        if self.search:
            clauses.append("file_name LIKE ?")
            params.append(f"%{self.search}%")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def order(self) -> str:
        if self.order_by == "imported":
            return " ORDER BY imported_at, id"
        if self.order_by == "probability":
            # Highest burst probability first; unscored files sink to the bottom
            # so an assisted triage pass reviews the likely bursts first.
            return " ORDER BY burst_probability IS NULL, burst_probability DESC, file_name"
        return " ORDER BY file_name, id"


class AnnotationRepository:
    """CRUD over files, boxes and session state."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- import ------------------------------------------------------------

    def add_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> int | None:
        """Insert one file. Returns the new id, or ``None`` if already present."""
        path = Path(path).resolve()
        metadata = metadata or {}
        connection = self.db.connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO files (
                    path, file_name, content_hash, station, obs_date, obs_time,
                    n_freq, n_time, freq_min_mhz, freq_max_mhz, freq_axis_source,
                    legacy_freq_min_mhz, legacy_freq_max_mhz, cadence_s, duration_s,
                    status, error, imported_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(path),
                    path.name,
                    metadata.get("content_hash"),
                    metadata.get("station"),
                    metadata.get("date"),
                    metadata.get("start_time"),
                    metadata.get("n_freq"),
                    metadata.get("n_time"),
                    metadata.get("freq_min_mhz"),
                    metadata.get("freq_max_mhz"),
                    metadata.get("freq_axis_source", "none"),
                    metadata.get("legacy_freq_min_mhz"),
                    metadata.get("legacy_freq_max_mhz"),
                    metadata.get("cadence_s"),
                    metadata.get("duration_s"),
                    STATUS_ERROR if metadata.get("error") else STATUS_PENDING,
                    metadata.get("error"),
                    _now(),
                ),
            )
        except sqlite3.IntegrityError:
            return None
        return int(cursor.lastrowid)

    def path_exists(self, path: str | Path) -> bool:
        row = self.db.connect().execute(
            "SELECT 1 FROM files WHERE path = ?", (str(Path(path).resolve()),)
        ).fetchone()
        return row is not None

    def file_id_for_path(self, path: str | Path) -> int | None:
        """Look up a file by absolute path (used to jump from a report to a file)."""
        row = self.db.connect().execute(
            "SELECT id FROM files WHERE path = ?", (str(Path(path).resolve()),)
        ).fetchone()
        return int(row["id"]) if row else None

    def hash_exists(self, digest: str) -> bool:
        row = self.db.connect().execute(
            "SELECT 1 FROM files WHERE content_hash = ?", (digest,)
        ).fetchone()
        return row is not None

    def existing_paths(self) -> set[str]:
        """All known paths, for fast bulk dedupe during an import scan."""
        return {row["path"] for row in self.db.connect().execute("SELECT path FROM files")}

    def existing_hashes(self) -> set[str]:
        return {
            row["content_hash"]
            for row in self.db.connect().execute(
                "SELECT content_hash FROM files WHERE content_hash IS NOT NULL"
            )
        }

    # -- queue -------------------------------------------------------------

    def file_ids(self, filters: QueueFilter | None = None) -> list[int]:
        filters = filters or QueueFilter()
        where, params = filters.where()
        rows = self.db.connect().execute(
            f"SELECT id FROM files{where}{filters.order()}", params
        )
        return [int(row["id"]) for row in rows]

    def files(self, filters: QueueFilter | None = None, limit: int | None = None) -> list[FileRecord]:
        filters = filters or QueueFilter()
        where, params = filters.where()
        query = (
            "SELECT f.*, (SELECT COUNT(*) FROM boxes b WHERE b.file_id = f.id) AS box_count "
            f"FROM files f{where}{filters.order()}"
        )
        if limit is not None:
            query += " LIMIT ?"
            params = [*params, int(limit)]
        return [FileRecord.from_row(row) for row in self.db.connect().execute(query, params)]

    def get_file(self, file_id: int) -> FileRecord | None:
        row = self.db.connect().execute(
            "SELECT f.*, (SELECT COUNT(*) FROM boxes b WHERE b.file_id = f.id) AS box_count "
            "FROM files f WHERE f.id = ?",
            (file_id,),
        ).fetchone()
        return FileRecord.from_row(row) if row else None

    def stations(self) -> list[str]:
        rows = self.db.connect().execute(
            "SELECT DISTINCT station FROM files WHERE station IS NOT NULL ORDER BY station"
        )
        return [row["station"] for row in rows]

    def status_counts(self) -> dict[str, int]:
        rows = self.db.connect().execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status")
        return {row["status"]: int(row["n"]) for row in rows}

    def verdict_counts(self) -> dict[str, int]:
        rows = self.db.connect().execute(
            "SELECT verdict, COUNT(*) AS n FROM files WHERE verdict IS NOT NULL GROUP BY verdict"
        )
        return {row["verdict"]: int(row["n"]) for row in rows}

    def box_type_counts(self) -> dict[str, int]:
        rows = self.db.connect().execute(
            "SELECT burst_type, COUNT(*) AS n FROM boxes WHERE confirmed = 1 GROUP BY burst_type"
        )
        return {row["burst_type"]: int(row["n"]) for row in rows}

    def total_files(self) -> int:
        return int(self.db.connect().execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])

    # -- labelling ---------------------------------------------------------

    def set_verdict(self, file_id: int, verdict: str | None) -> None:
        """Record burst / no_burst / unsure and advance the review status."""
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"Unknown verdict: {verdict}")
        status = STATUS_LABELED if verdict else STATUS_IN_REVIEW
        self.db.connect().execute(
            "UPDATE files SET verdict = ?, status = ?, labeled_at = ? WHERE id = ?",
            (verdict, status, _now() if verdict else None, file_id),
        )

    def set_status(self, file_id: int, status: str) -> None:
        self.db.connect().execute(
            "UPDATE files SET status = ? WHERE id = ?", (status, file_id)
        )

    def set_notes(self, file_id: int, notes: str) -> None:
        self.db.connect().execute("UPDATE files SET notes = ? WHERE id = ?", (notes, file_id))

    def set_burst_probability(self, file_id: int, probability: float | None) -> None:
        self.db.connect().execute(
            "UPDATE files SET burst_probability = ? WHERE id = ?", (probability, file_id)
        )

    def update_file_axes(self, file_id: int, metadata: dict[str, Any]) -> None:
        """Refresh the axis-derived columns after re-reading a file's header."""
        self.db.connect().execute(
            """
            UPDATE files SET
                freq_min_mhz = ?, freq_max_mhz = ?, freq_axis_source = ?,
                legacy_freq_min_mhz = ?, legacy_freq_max_mhz = ?,
                cadence_s = ?, duration_s = ?, n_freq = ?, n_time = ?
            WHERE id = ?
            """,
            (
                metadata.get("freq_min_mhz"),
                metadata.get("freq_max_mhz"),
                metadata.get("freq_axis_source", "none"),
                metadata.get("legacy_freq_min_mhz"),
                metadata.get("legacy_freq_max_mhz"),
                metadata.get("cadence_s"),
                metadata.get("duration_s"),
                metadata.get("n_freq"),
                metadata.get("n_time"),
                file_id,
            ),
        )

    def update_box_physical(self, box_id: int, physical: dict[str, float]) -> None:
        """Refresh a box's derived physical bounds after an axis correction."""
        self.db.connect().execute(
            "UPDATE boxes SET freq_lo_mhz = ?, freq_hi_mhz = ?, t_start_s = ?, "
            "t_end_s = ?, updated_at = ? WHERE id = ?",
            (
                physical.get("freq_lo_mhz"),
                physical.get("freq_hi_mhz"),
                physical.get("t_start_s"),
                physical.get("t_end_s"),
                _now(),
                box_id,
            ),
        )

    def all_files(self) -> list[FileRecord]:
        """Every imported file, for maintenance passes."""
        return self.files(QueueFilter())

    def set_error(self, file_id: int, message: str) -> None:
        self.db.connect().execute(
            "UPDATE files SET status = ?, error = ? WHERE id = ?",
            (STATUS_ERROR, message, file_id),
        )

    # -- boxes -------------------------------------------------------------

    def add_box(
        self,
        file_id: int,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        burst_type: str,
        physical: dict[str, float] | None = None,
        confidence: str = "certain",
        source: str = "manual",
        confirmed: bool = True,
    ) -> int:
        physical = physical or {}
        timestamp = _now()
        cursor = self.db.connect().execute(
            """
            INSERT INTO boxes (
                file_id, row0, row1, col0, col1,
                freq_lo_mhz, freq_hi_mhz, t_start_s, t_end_s,
                burst_type, confidence, source, confirmed, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                file_id,
                int(row0),
                int(row1),
                int(col0),
                int(col1),
                physical.get("freq_lo_mhz"),
                physical.get("freq_hi_mhz"),
                physical.get("t_start_s"),
                physical.get("t_end_s"),
                burst_type,
                confidence,
                source,
                int(bool(confirmed)),
                timestamp,
                timestamp,
            ),
        )
        return int(cursor.lastrowid)

    def update_box(
        self,
        box_id: int,
        row0: int | None = None,
        row1: int | None = None,
        col0: int | None = None,
        col1: int | None = None,
        burst_type: str | None = None,
        physical: dict[str, float] | None = None,
        confidence: str | None = None,
        confirmed: bool | None = None,
    ) -> None:
        assignments: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("row0", row0),
            ("row1", row1),
            ("col0", col0),
            ("col1", col1),
            ("burst_type", burst_type),
            ("confidence", confidence),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if confirmed is not None:
            assignments.append("confirmed = ?")
            params.append(int(bool(confirmed)))
        for key in ("freq_lo_mhz", "freq_hi_mhz", "t_start_s", "t_end_s"):
            if physical and key in physical:
                assignments.append(f"{key} = ?")
                params.append(physical[key])
        if not assignments:
            return

        assignments.append("updated_at = ?")
        params.extend([_now(), box_id])
        self.db.connect().execute(
            f"UPDATE boxes SET {', '.join(assignments)} WHERE id = ?", params
        )

    def set_box_physics(self, box_id: int, physics: dict[str, Any]) -> None:
        """Store measured physics for one box. Unknown keys are ignored."""
        names = [name for name, _type in PHYSICS_COLUMNS if name in physics]
        if not names:
            return
        values = []
        for name in names:
            value = physics[name]
            values.append(int(value) if isinstance(value, bool) else value)
        assignments = ", ".join(f"{name} = ?" for name in names)
        self.db.connect().execute(
            f"UPDATE boxes SET {assignments} WHERE id = ?", [*values, box_id]
        )

    def boxes_missing_physics(self) -> list[int]:
        """Ids of boxes with no measurement yet, for backfilling."""
        rows = self.db.connect().execute(
            "SELECT id FROM boxes WHERE physics_confidence IS NULL ORDER BY file_id, id"
        )
        return [int(row["id"]) for row in rows]

    def delete_box(self, box_id: int) -> None:
        self.db.connect().execute("DELETE FROM boxes WHERE id = ?", (box_id,))

    def delete_boxes_for_file(self, file_id: int) -> None:
        self.db.connect().execute("DELETE FROM boxes WHERE file_id = ?", (file_id,))

    def delete_burst_boxes_for_file(self, file_id: int) -> None:
        """Drop every burst box on a file: the boxes a "No burst" verdict contradicts."""
        placeholders = ",".join("?" * len(BURST_TYPES))
        self.db.connect().execute(
            f"DELETE FROM boxes WHERE file_id = ? AND burst_type IN ({placeholders})",
            (file_id, *BURST_TYPES),
        )

    def boxes_for_file(self, file_id: int) -> list[BoxRecord]:
        rows = self.db.connect().execute(
            "SELECT * FROM boxes WHERE file_id = ? ORDER BY id", (file_id,)
        )
        return [BoxRecord.from_row(row) for row in rows]

    def total_boxes(self, confirmed_only: bool = True) -> int:
        query = "SELECT COUNT(*) AS n FROM boxes"
        if confirmed_only:
            query += " WHERE confirmed = 1"
        return int(self.db.connect().execute(query).fetchone()["n"])

    # -- export queries ----------------------------------------------------

    def iter_labeled_boxes(self) -> Iterator[tuple[FileRecord, BoxRecord]]:
        """Every confirmed box on a file the user marked as containing a burst."""
        rows = self.db.connect().execute(
            """
            SELECT b.*, f.path AS f_path
            FROM boxes b
            JOIN files f ON f.id = b.file_id
            WHERE b.confirmed = 1 AND f.verdict = ?
            ORDER BY f.file_name, b.id
            """,
            (VERDICT_BURST,),
        )
        cache: dict[int, FileRecord] = {}
        for row in rows:
            box = BoxRecord.from_row(row)
            record = cache.get(box.file_id)
            if record is None:
                record = self.get_file(box.file_id)
                if record is None:
                    continue
                cache[box.file_id] = record
            yield record, box

    def iter_training_boxes(self) -> Iterator[tuple[FileRecord, BoxRecord]]:
        """Every confirmed burst box on a file marked Burst.

        Interference is never drawn: the exporter finds it itself, outside these
        boxes and in no-burst files. A box of any other type (an RFI box from an
        earlier version) is left out. Unsure files contribute nothing.
        """
        placeholders = ",".join("?" * len(BURST_TYPES))
        rows = self.db.connect().execute(
            f"""
            SELECT b.*
            FROM boxes b
            JOIN files f ON f.id = b.file_id
            WHERE b.confirmed = 1 AND f.verdict = ? AND b.burst_type IN ({placeholders})
            ORDER BY f.file_name, b.id
            """,
            (VERDICT_BURST, *BURST_TYPES),
        )
        cache: dict[int, FileRecord] = {}
        for row in rows:
            box = BoxRecord.from_row(row)
            record = cache.get(box.file_id)
            if record is None:
                record = self.get_file(box.file_id)
                if record is None:
                    continue
                cache[box.file_id] = record
            yield record, box

    def files_with_verdict(self, verdicts: Iterable[str]) -> list[FileRecord]:
        verdicts = list(verdicts)
        placeholders = ",".join("?" * len(verdicts))
        rows = self.db.connect().execute(
            "SELECT f.*, (SELECT COUNT(*) FROM boxes b WHERE b.file_id = f.id) AS box_count "
            f"FROM files f WHERE f.verdict IN ({placeholders}) ORDER BY f.file_name",
            verdicts,
        )
        return [FileRecord.from_row(row) for row in rows]

    # -- reset -------------------------------------------------------------

    def clear_file_labels(self, file_id: int) -> None:
        """Return one file to unreviewed: drop its boxes, verdict and notes."""
        connection = self.db.connect()
        with connection:
            connection.execute("DELETE FROM boxes WHERE file_id = ?", (file_id,))
            connection.execute(
                "UPDATE files SET verdict = NULL, status = ?, labeled_at = NULL, notes = NULL "
                "WHERE id = ?",
                (STATUS_PENDING, file_id),
            )

    def reset_dataset(self, backup_path: str | Path | None = None) -> Path | None:
        """Delete every file, box and session value. Irreversible.

        When ``backup_path`` is given the current database is copied there first,
        so an accidental reset is still recoverable.
        """
        backup = self.db.backup(backup_path) if backup_path else None
        self.db.reset()
        return backup

    # -- session state -----------------------------------------------------

    def set_state(self, key: str, value: str) -> None:
        self.db.connect().execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.db.connect().execute(
            "SELECT value FROM app_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default
