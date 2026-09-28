"""Deleting snapshots: the guards, and the tab wiring around them.

A snapshot is the provenance record for every model trained from it and holds
thousands of files, so the delete path is written to refuse anything it cannot
positively identify rather than to trust its caller.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.store.export import (  # noqa: E402
    delete_snapshot,
    list_snapshots,
    snapshot_size_bytes,
)


def _make_snapshot(datasets: Path, kind: str, name: str, files: int = 3) -> Path:
    directory = datasets / kind / name
    (directory / "npz").mkdir(parents=True)
    (directory / "manifest.csv").write_text("file_path\n", encoding="utf-8")
    (directory / "snapshot.json").write_text('{"kind": "%s"}' % kind, encoding="utf-8")
    for index in range(files):
        (directory / "npz" / f"s{index}.npz").write_bytes(b"x" * 1024)
    return directory


# -- the happy path --------------------------------------------------------


def test_deleting_a_snapshot_removes_it_and_reports_the_space(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    directory = _make_snapshot(datasets, "unified", "20260101_000000", files=4)
    expected = snapshot_size_bytes(directory)

    freed = delete_snapshot(datasets, directory)

    assert not directory.exists()
    assert freed == expected >= 4 * 1024
    assert list_snapshots(datasets, "unified") == []


def test_deleting_one_snapshot_leaves_the_others(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    doomed = _make_snapshot(datasets, "unified", "20260101_000000")
    keep = _make_snapshot(datasets, "unified", "20260102_000000")
    other_kind = _make_snapshot(datasets, "binary", "20260101_000000")

    delete_snapshot(datasets, doomed)

    assert keep.exists() and other_kind.exists()
    assert [p.name for p in list_snapshots(datasets, "unified")] == ["20260102_000000"]


# -- the guards ------------------------------------------------------------


def test_refuses_a_path_outside_the_datasets_directory(tmp_path: Path) -> None:
    """The guard that stops a stale path from becoming a recursive delete."""
    datasets = tmp_path / "datasets"
    _make_snapshot(datasets, "unified", "20260101_000000")

    outsider = tmp_path / "important_work"
    outsider.mkdir()
    (outsider / "manifest.csv").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the datasets directory"):
        delete_snapshot(datasets, outsider)
    assert outsider.exists(), "a path outside the datasets tree must survive"


def test_refuses_to_delete_a_whole_kind_directory(tmp_path: Path) -> None:
    """datasets/unified holds every unified snapshot; it is not itself one."""
    datasets = tmp_path / "datasets"
    kept = _make_snapshot(datasets, "unified", "20260101_000000")
    kind_dir = datasets / "unified"
    (kind_dir / "manifest.csv").write_text("x", encoding="utf-8")  # make it look plausible

    with pytest.raises(ValueError, match="datasets/<kind>/<run>"):
        delete_snapshot(datasets, kind_dir)
    assert kept.exists(), "deleting the kind directory would have taken every snapshot"


def test_refuses_a_directory_that_is_not_a_snapshot(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    stray = datasets / "unified" / "notes"
    stray.mkdir(parents=True)
    (stray / "todo.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="does not look like a snapshot"):
        delete_snapshot(datasets, stray)
    assert (stray / "todo.txt").exists()


def test_missing_snapshot_raises_rather_than_passing_silently(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    with pytest.raises(FileNotFoundError):
        delete_snapshot(datasets, datasets / "unified" / "gone")


def test_refuses_a_file(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    directory = _make_snapshot(datasets, "unified", "20260101_000000")
    with pytest.raises(ValueError, match="Not a directory"):
        delete_snapshot(datasets, directory / "manifest.csv")


# -- the Dataset tab -------------------------------------------------------


@pytest.fixture
def tab(tmp_path: Path):
    from PySide6.QtWidgets import QApplication

    from callisto_trainer.settings import AppSettings
    from callisto_trainer.store.db import Database
    from callisto_trainer.store.repository import AnnotationRepository
    from callisto_trainer.ui.dataset_tab import DatasetTab

    QApplication.instance() or QApplication([])
    datasets = tmp_path / "datasets"
    _make_snapshot(datasets, "unified", "20260101_000000")
    _make_snapshot(datasets, "binary", "20260102_000000")

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "annotations.db",
        datasets_dir=datasets,
        outputs_dir=tmp_path / "outputs",
    )
    repository = AnnotationRepository(Database(settings.database_path))
    return DatasetTab(repository, settings), datasets


def test_table_lists_snapshots_with_their_size(tab) -> None:
    widget, _ = tab
    assert widget.table.rowCount() == 2
    assert "on disk" in widget.disk_label.text()


def test_delete_button_is_disabled_until_a_row_is_selected(tab) -> None:
    widget, _ = tab
    assert not widget.delete_selected_button.isEnabled()

    widget.table.selectRow(0)
    assert widget.delete_selected_button.isEnabled()
    assert len(widget._selected_snapshots()) == 1


def test_selected_paths_come_from_the_row_the_operator_saw(tab) -> None:
    widget, datasets = tab
    widget.table.selectRow(0)
    selected = widget._selected_snapshots()[0]

    assert selected.exists()
    assert selected.parent.parent == datasets


def test_all_snapshots_covers_every_kind(tab) -> None:
    widget, _ = tab
    kinds = {path.parent.name for path in widget._all_snapshots()}
    assert kinds == {"unified", "binary"}


def test_a_snapshot_in_use_is_refused(tab, monkeypatch) -> None:
    """Deleting under a running job would crash it minutes later, not now."""
    from PySide6.QtWidgets import QMessageBox

    widget, _ = tab
    busy = widget._all_snapshots()[0]
    widget.snapshot_in_use = lambda: [busy]

    asked = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: asked.append(a))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: pytest.fail("must not reach confirmation")
    )

    widget._confirm_and_delete([busy], "it")

    assert asked, "the operator must be told why the delete was refused"
    assert busy.exists()


def test_cancelling_the_confirmation_deletes_nothing(tab, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    widget, _ = tab
    targets = widget._all_snapshots()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )

    widget._confirm_and_delete(targets, "everything")

    assert all(path.exists() for path in targets)


def test_confirmed_delete_removes_the_snapshots_and_signals(tab, monkeypatch, qtbot=None) -> None:
    from PySide6.QtWidgets import QApplication, QMessageBox

    widget, datasets = tab
    targets = widget._all_snapshots()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    signalled: list[int] = []
    widget.snapshots_changed.connect(lambda: signalled.append(1))

    widget._confirm_and_delete(targets, "everything")
    # The delete runs on a worker thread; wait for it the way Qt intends.
    assert widget._delete_worker is not None
    widget._delete_worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()

    assert all(not path.exists() for path in targets)
    assert widget.table.rowCount() == 0
    assert signalled, "Train and Evaluate tabs must be told the list changed"
    assert list_snapshots(datasets, "unified") == []
