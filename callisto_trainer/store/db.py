"""SQLite connection management and schema for the annotation store.

Why SQLite rather than a JSON sidecar: the tool has to survive being killed at
any moment with zero lost labels (objective 5), and has to answer "how many
files are still pending" instantly over a queue that can hold 100k+ rows. A
single JSON document would rewrite the whole file on every box edit and would be
corrupt if the process died mid-write.

Concurrency model: WAL journalling with one connection per thread. WAL lets the
decode/prefetch workers read while the UI thread writes, and a single-row write
is microseconds, so the UI never blocks on the database.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 3

# v2 adds measured burst physics to each box: extent, drift rate and fit quality.
# See callisto_trainer/core/burst_physics.py. Added as columns rather than a side
# table because there is exactly one measurement per box and it is read on every
# export and every label-panel refresh.
#
# v3 adds ``burst_count``, the number of separate bursts inside a box, which is
# what distinguishes a Type III group (Type IIIG) from a single Type III. The new
# box types themselves (Type IIIG, RFI) need no schema change: ``burst_type`` is
# free text, validated against core/taxonomy.py by the code that writes it.
PHYSICS_COLUMNS: list[tuple[str, str]] = [
    ("freq_start_mhz", "REAL"),
    ("freq_end_mhz", "REAL"),
    ("freq_high_mhz", "REAL"),
    ("freq_low_mhz", "REAL"),
    ("time_start_s", "REAL"),
    ("time_end_s", "REAL"),
    ("duration_s", "REAL"),
    ("bandwidth_mhz", "REAL"),
    ("drift_mhz_per_s", "REAL"),
    ("relative_drift_per_s", "REAL"),
    ("fit_quality", "REAL"),
    ("track_samples", "INTEGER"),
    ("track_axis", "TEXT"),
    ("edge_clipped", "INTEGER"),
    ("physics_confidence", "TEXT"),
    ("burst_count", "INTEGER"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- One row per imported FITS file. Files are referenced in place; `path` is
-- absolute and `content_hash` allows re-linking if the archive is moved.
CREATE TABLE IF NOT EXISTS files (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    path                 TEXT    NOT NULL UNIQUE,
    file_name            TEXT    NOT NULL,
    content_hash         TEXT,
    station              TEXT,
    obs_date             TEXT,
    obs_time             TEXT,
    n_freq               INTEGER,
    n_time               INTEGER,
    freq_min_mhz         REAL,
    freq_max_mhz         REAL,
    freq_axis_source     TEXT    NOT NULL DEFAULT 'none',
    legacy_freq_min_mhz  REAL,
    legacy_freq_max_mhz  REAL,
    cadence_s            REAL,
    duration_s           REAL,
    status               TEXT    NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','in_review','labeled','skipped','error')),
    verdict              TEXT
                         CHECK (verdict IS NULL OR verdict IN ('burst','no_burst','unsure')),
    burst_probability    REAL,
    notes                TEXT,
    error                TEXT,
    imported_at          TEXT    NOT NULL,
    labeled_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_status  ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_verdict ON files(verdict);
CREATE INDEX IF NOT EXISTS idx_files_station ON files(station);
CREATE INDEX IF NOT EXISTS idx_files_hash    ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_name    ON files(file_name);

-- One row per drawn burst. Pixel indices are authoritative and always present;
-- the physical columns are derived and make the annotation portable.
CREATE TABLE IF NOT EXISTS boxes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    row0         INTEGER NOT NULL,
    row1         INTEGER NOT NULL,
    col0         INTEGER NOT NULL,
    col1         INTEGER NOT NULL,
    freq_lo_mhz  REAL,
    freq_hi_mhz  REAL,
    t_start_s    REAL,
    t_end_s      REAL,
    burst_type   TEXT    NOT NULL,
    confidence   TEXT    NOT NULL DEFAULT 'certain'
                 CHECK (confidence IN ('certain','probable')),
    source       TEXT    NOT NULL DEFAULT 'manual'
                 CHECK (source IN ('manual','assisted')),
    confirmed    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    CHECK (row1 > row0),
    CHECK (col1 > col0)
);

CREATE INDEX IF NOT EXISTS idx_boxes_file ON boxes(file_id);
CREATE INDEX IF NOT EXISTS idx_boxes_type ON boxes(burst_type);

-- Session state so the app resumes exactly where it was left.
CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Database:
    """Thread-safe SQLite handle for the annotation store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        with self._init_lock:
            self._create_schema()

    # -- connections -------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, creating it on first use."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,  # autocommit; explicit transactions below
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        """Close this thread's connection."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    # -- schema ------------------------------------------------------------

    def _create_schema(self) -> None:
        connection = self.connect()
        connection.executescript(SCHEMA)
        row = connection.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            connection.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        else:
            self._migrate(connection, int(row["version"]))
        # Runs for fresh databases as well as migrated ones, so the two paths
        # cannot drift apart.
        self._ensure_physics_columns(connection)

    def _migrate(self, connection: sqlite3.Connection, from_version: int) -> None:
        """Upgrade an older database in place.

        Opening a database created by a *newer* build is refused rather than
        silently misread.
        """
        if from_version == SCHEMA_VERSION:
            self._ensure_physics_columns(connection)
            return
        if from_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was written by a newer version of CALLISTO Trainer "
                f"(schema v{from_version}, this build understands v{SCHEMA_VERSION}). "
                "Update the app rather than risk misreading your annotations."
            )

        if from_version < 3:
            # Additive only: existing boxes keep their geometry and labels, and
            # their physics columns (v2) and burst count (v3) stay NULL until
            # measured or backfilled.
            self._ensure_physics_columns(connection)

        connection.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def _ensure_physics_columns(self, connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(boxes)")}
        for name, sql_type in PHYSICS_COLUMNS:
            if name not in existing:
                connection.execute(f"ALTER TABLE boxes ADD COLUMN {name} {sql_type}")

    @property
    def version(self) -> int:
        row = self.connect().execute("SELECT version FROM schema_version").fetchone()
        return int(row["version"]) if row else 0

    # -- maintenance -------------------------------------------------------

    def backup(self, destination: str | Path) -> Path:
        """Copy the live database to ``destination`` using SQLite's backup API.

        Safe on an open database with WAL entries outstanding, unlike copying the
        file, which can capture a torn state.
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination)
        try:
            with target:
                self.connect().backup(target)
        finally:
            target.close()
        return destination

    def reset(self) -> None:
        """Delete all annotations and imported files, leaving an empty schema.

        Irreversible. Callers are expected to have taken a backup and obtained
        explicit confirmation first.
        """
        connection = self.connect()
        with connection:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM boxes")
            connection.execute("DELETE FROM files")
            connection.execute("DELETE FROM app_state")
            connection.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('files','boxes')"
            )
        connection.execute("VACUUM")
