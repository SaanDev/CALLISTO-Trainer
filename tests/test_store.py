"""Annotation store: CRUD, dedupe, resume-after-kill, and concurrent access."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from callisto_trainer.store.db import SCHEMA_VERSION, Database
from callisto_trainer.store.repository import (
    STATUS_LABELED,
    STATUS_PENDING,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
    QueueFilter,
    content_hash,
)


@pytest.fixture
def repo(tmp_path: Path) -> AnnotationRepository:
    return AnnotationRepository(Database(tmp_path / "annotations.db"))


def _add(repo: AnnotationRepository, tmp_path: Path, name: str, **metadata) -> int:
    path = tmp_path / name
    path.write_bytes(b"not really fits, but a stable byte string for hashing")
    base = {
        "station": "ALASKA-ANCHORAGE",
        "date": "2023-06-13",
        "start_time": "23:00:59",
        "n_freq": 181,
        "n_time": 1200,
        "freq_min_mhz": 5.875,
        "freq_max_mhz": 65.875,
        "freq_axis_source": "axes_table",
    }
    base.update(metadata)
    file_id = repo.add_file(path, base)
    assert file_id is not None
    return file_id


def test_schema_is_created_and_versioned(tmp_path: Path) -> None:
    database = Database(tmp_path / "a.db")
    assert database.version == SCHEMA_VERSION


def test_reopening_an_existing_database_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "a.db"
    repo = AnnotationRepository(Database(path))
    _add(repo, tmp_path, "one.fit.gz")

    reopened = AnnotationRepository(Database(path))
    assert reopened.total_files() == 1


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    """Opening a database from a future build must fail loudly, not misread it."""
    path = tmp_path / "a.db"
    Database(path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 5,))
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="newer version"):
        Database(path)


def test_add_file_and_read_back(repo: AnnotationRepository, tmp_path: Path) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    record = repo.get_file(file_id)

    assert record is not None
    assert record.file_name == "one.fit.gz"
    assert record.status == STATUS_PENDING
    assert record.verdict is None
    assert record.freq_min_mhz == pytest.approx(5.875)
    assert not record.freq_axis_is_approximate


def test_duplicate_path_is_rejected(repo: AnnotationRepository, tmp_path: Path) -> None:
    _add(repo, tmp_path, "one.fit.gz")
    path = tmp_path / "one.fit.gz"
    assert repo.add_file(path, {}) is None
    assert repo.total_files() == 1


def test_existing_paths_and_hashes_support_bulk_dedupe(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    path = tmp_path / "one.fit.gz"
    digest = None
    _add(repo, tmp_path, "one.fit.gz")
    digest = content_hash(path)
    repo.db.connect().execute(
        "UPDATE files SET content_hash = ? WHERE file_name = 'one.fit.gz'", (digest,)
    )

    assert str(path.resolve()) in repo.existing_paths()
    assert digest in repo.existing_hashes()
    assert repo.hash_exists(digest)


def test_content_hash_is_stable_and_distinguishes(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 5000)
    b.write_bytes(b"y" * 5000)

    assert content_hash(a) == content_hash(a)
    assert content_hash(a) != content_hash(b)


def test_verdict_updates_status(repo: AnnotationRepository, tmp_path: Path) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    repo.set_verdict(file_id, VERDICT_BURST)

    record = repo.get_file(file_id)
    assert record.verdict == VERDICT_BURST
    assert record.status == STATUS_LABELED
    assert record.labeled_at


def test_unknown_verdict_is_rejected(repo: AnnotationRepository, tmp_path: Path) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    with pytest.raises(ValueError):
        repo.set_verdict(file_id, "maybe-ish")


def test_box_lifecycle(repo: AnnotationRepository, tmp_path: Path) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    physical = {
        "freq_lo_mhz": 20.0,
        "freq_hi_mhz": 45.0,
        "t_start_s": 12.5,
        "t_end_s": 40.0,
    }
    box_id = repo.add_box(file_id, 10, 50, 100, 260, "Type III", physical=physical)

    boxes = repo.boxes_for_file(file_id)
    assert len(boxes) == 1
    assert boxes[0].burst_type == "Type III"
    assert boxes[0].freq_hi_mhz == pytest.approx(45.0)
    assert boxes[0].confirmed

    repo.update_box(box_id, burst_type="Type II", row1=60)
    updated = repo.boxes_for_file(file_id)[0]
    assert updated.burst_type == "Type II"
    assert updated.row1 == 60

    repo.delete_box(box_id)
    assert repo.boxes_for_file(file_id) == []


def test_multiple_boxes_per_file(repo: AnnotationRepository, tmp_path: Path) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    repo.add_box(file_id, 0, 20, 0, 100, "Type II")
    repo.add_box(file_id, 40, 90, 300, 500, "Type III")
    repo.add_box(file_id, 100, 150, 700, 900, "Other")

    assert len(repo.boxes_for_file(file_id)) == 3
    assert repo.box_type_counts() == {"Type II": 1, "Type III": 1, "Other": 1}


def test_degenerate_box_is_rejected_by_the_schema(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_box(file_id, 50, 50, 0, 10, "Other")


def test_deleting_a_file_cascades_to_its_boxes(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    repo.add_box(file_id, 0, 20, 0, 100, "Type II")
    repo.db.connect().execute("DELETE FROM files WHERE id = ?", (file_id,))
    assert repo.total_boxes() == 0


def test_queue_filters(repo: AnnotationRepository, tmp_path: Path) -> None:
    a = _add(repo, tmp_path, "AAA_20230101_0000_0005.fit.gz", station="AAA")
    b = _add(repo, tmp_path, "BBB_20230101_0000_0005.fit.gz", station="BBB")
    _add(repo, tmp_path, "CCC_20230101_0000_0005.fit.gz", station="CCC")
    repo.set_verdict(a, VERDICT_BURST)
    repo.set_verdict(b, VERDICT_NO_BURST)

    assert len(repo.file_ids(QueueFilter(statuses=[STATUS_PENDING]))) == 1
    assert repo.file_ids(QueueFilter(verdicts=[VERDICT_BURST])) == [a]
    assert repo.file_ids(QueueFilter(stations=["BBB"])) == [b]
    assert len(repo.file_ids(QueueFilter(search="CCC"))) == 1
    assert repo.status_counts()[STATUS_LABELED] == 2
    assert repo.verdict_counts() == {VERDICT_BURST: 1, VERDICT_NO_BURST: 1}


def test_probability_ordering_puts_likely_bursts_first(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    low = _add(repo, tmp_path, "low.fit.gz")
    high = _add(repo, tmp_path, "high.fit.gz")
    unscored = _add(repo, tmp_path, "unscored.fit.gz")
    repo.set_burst_probability(low, 0.10)
    repo.set_burst_probability(high, 0.97)

    assert repo.file_ids(QueueFilter(order_by="probability")) == [high, low, unscored]


def test_box_count_is_reported_with_the_file(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    file_id = _add(repo, tmp_path, "one.fit.gz")
    repo.add_box(file_id, 0, 20, 0, 100, "Type II")
    repo.add_box(file_id, 30, 50, 0, 100, "Type III")
    assert repo.get_file(file_id).box_count == 2


def test_iter_labeled_boxes_only_yields_confirmed_bursts(
    repo: AnnotationRepository, tmp_path: Path
) -> None:
    burst = _add(repo, tmp_path, "burst.fit.gz")
    quiet = _add(repo, tmp_path, "quiet.fit.gz")
    repo.set_verdict(burst, VERDICT_BURST)
    repo.set_verdict(quiet, VERDICT_NO_BURST)

    repo.add_box(burst, 0, 20, 0, 100, "Type III")
    repo.add_box(burst, 30, 50, 0, 100, "Type II", source="assisted", confirmed=False)
    repo.add_box(quiet, 0, 20, 0, 100, "Other")  # orphaned by the no_burst verdict

    pairs = list(repo.iter_labeled_boxes())
    assert len(pairs) == 1
    assert pairs[0][1].burst_type == "Type III"
    assert pairs[0][0].file_name == "burst.fit.gz"


def test_session_state_round_trip(repo: AnnotationRepository) -> None:
    assert repo.get_state("queue_index") is None
    repo.set_state("queue_index", "42")
    repo.set_state("queue_index", "43")  # upsert
    assert repo.get_state("queue_index") == "43"
    assert repo.get_state("missing", "fallback") == "fallback"


def test_work_survives_process_death(tmp_path: Path) -> None:
    """Simulate a kill: nothing is flushed on exit, yet everything is present."""
    path = tmp_path / "annotations.db"
    first = AnnotationRepository(Database(path))
    file_id = _add(first, tmp_path, "one.fit.gz")
    first.set_verdict(file_id, VERDICT_BURST)
    first.add_box(file_id, 10, 50, 100, 260, "Type III")
    first.set_state("queue_index", "7")
    # No close(), no commit() - exactly what happens when the app is killed.

    revived = AnnotationRepository(Database(path))
    record = revived.get_file(file_id)
    assert record.verdict == VERDICT_BURST
    assert len(revived.boxes_for_file(file_id)) == 1
    assert revived.get_state("queue_index") == "7"


def test_concurrent_readers_during_writes(tmp_path: Path) -> None:
    """WAL must let prefetch workers read while the UI thread writes."""
    database = Database(tmp_path / "annotations.db")
    repo = AnnotationRepository(database)
    file_ids = [_add(repo, tmp_path, f"file_{i:03d}.fit.gz") for i in range(40)]

    errors: list[Exception] = []

    def reader() -> None:
        try:
            worker_repo = AnnotationRepository(database)
            for _ in range(60):
                worker_repo.files(QueueFilter())
                worker_repo.status_counts()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for file_id in file_ids:
        repo.set_verdict(file_id, VERDICT_BURST)
        repo.add_box(file_id, 0, 20, 0, 100, "Type III")
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent access failed: {errors}"
    assert repo.total_boxes() == 40
