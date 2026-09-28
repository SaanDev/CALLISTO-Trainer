"""Raw-data inspection and the reset actions."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.fits_reader import read_header_text  # noqa: E402
from callisto_trainer.services.cache import load_bundle  # noqa: E402
from callisto_trainer.services.importer import import_files  # noqa: E402
from callisto_trainer.settings import AppSettings  # noqa: E402
from callisto_trainer.store.db import Database  # noqa: E402
from callisto_trainer.store.repository import (  # noqa: E402
    STATUS_PENDING,
    VERDICT_BURST,
    AnnotationRepository,
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config() -> dict:
    return load_config()


def pump(qapp, seconds: float = 1.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(qapp, tmp_path: Path, axes_files):
    from callisto_trainer.ui.main_window import MainWindow

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "data" / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    main = MainWindow(settings)
    import_files(main.repository, [Path(p) for p in axes_files[:3]])
    main.label_tab.refresh_queue(keep_selection=False)
    main.label_tab.queue.select_row(0)
    pump(qapp)
    assert main.label_tab._current_bundle is not None
    yield main
    main.label_tab.shutdown()
    main.close()


# -- raw data --------------------------------------------------------------


def test_bundle_carries_raw_and_normalized(any_real_file, config: dict) -> None:
    bundle = load_bundle(1, any_real_file, config)

    assert bundle.raw is not None
    assert bundle.raw.shape == bundle.normalized.shape
    # Raw values are uncalibrated receiver digits, not the 0-1 model input.
    # nanmax, because real files do contain invalid samples (see below).
    assert float(np.nanmax(bundle.raw)) > 1.0
    assert 0.0 <= float(bundle.normalized.min())
    assert float(bundle.normalized.max()) <= 1.0
    assert bundle.quiet is not None and bundle.quiet.shape == bundle.normalized.shape
    assert bundle.nbytes == bundle.raw.nbytes + bundle.normalized.nbytes + bundle.quiet.nbytes


def test_raw_keeps_invalid_samples_that_preprocessing_removes(
    any_real_file, config: dict
) -> None:
    """The raw view must show the file as it is, NaNs included.

    ``clean_invalid_values`` replaces them with the finite median on the way to
    the model, so the two views genuinely differ there -- which is exactly the
    kind of thing the raw view exists to reveal.
    """
    bundle = load_bundle(1, any_real_file, config)
    if not bundle.raw_has_invalid:
        pytest.skip("this file has no invalid samples")

    assert not np.isfinite(bundle.raw).all()
    assert np.isfinite(bundle.normalized).all(), "the model must never receive NaN"


def test_raw_levels_ignore_invalid_samples(any_real_file, config: dict) -> None:
    bundle = load_bundle(1, any_real_file, config)
    low, high = bundle.raw_levels()

    assert np.isfinite(low) and np.isfinite(high)
    assert high > low
    assert low >= float(np.nanmin(bundle.raw)) - 1e-6
    assert high <= float(np.nanmax(bundle.raw)) + 1e-6


def test_nan_aware_decimation_keeps_good_samples() -> None:
    """One invalid sample must not blank a whole pooled block."""
    from callisto_trainer.services.cache import decimate_for_display

    array = np.ones((4, 1000), dtype=np.float32)
    array[:, 500] = np.nan  # a single bad sample among good ones

    naive, _ = decimate_for_display(array, max_cols=100, nan_aware=False)
    aware, _ = decimate_for_display(array, max_cols=100, nan_aware=True)

    assert np.isnan(naive).any(), "plain max propagates the NaN"
    assert np.isfinite(aware).all(), "nan-aware pooling recovers the good samples"


def test_fully_invalid_block_stays_invalid() -> None:
    """Genuinely missing data must render as a gap, not as fabricated signal."""
    from callisto_trainer.services.cache import decimate_for_display

    array = np.ones((4, 1000), dtype=np.float32)
    array[:, 0:10] = np.nan  # an entire pooled block is invalid

    pooled, _ = decimate_for_display(array, max_cols=100, nan_aware=True)
    assert np.isnan(pooled[:, 0]).all()


def test_header_text_reports_the_axis_source(axes_files, no_axis_table_file) -> None:
    with_axes = read_header_text(axes_files[0])
    assert "PRIMARY HEADER" in with_axes
    assert "INSTRUME" in with_axes
    assert "Source: AXES table" in with_axes

    without = read_header_text(no_axis_table_file)
    assert "CRVAL2/CDELT2 header fallback" in without
    assert "approximate" in without


def test_bundle_exposes_header_text(any_real_file, config: dict) -> None:
    bundle = load_bundle(1, any_real_file, config)
    assert "PRIMARY HEADER" in bundle.header_text


def test_canvas_switches_view_without_moving_boxes(window) -> None:
    tab = window.label_tab
    tab._on_box_created(20, 70, 300, 520)
    before = tab.canvas._rois[next(iter(tab.canvas._rois))].pixel_box()

    tab.view_mode.setCurrentIndex(1)
    assert tab.canvas.view_mode == "raw"

    after = tab.canvas._rois[next(iter(tab.canvas._rois))].pixel_box()
    assert after == before, "switching views must not move an annotation"


def test_raw_view_renders_the_raw_array(window) -> None:
    tab = window.label_tab
    tab.view_mode.setCurrentIndex(1)
    displayed = tab.canvas.image.image

    assert displayed is not None
    # The raw array's dynamic range is far wider than the normalized 0-1 one.
    assert float(np.nanmax(displayed)) > 1.0


def test_switching_back_restores_the_model_view(window) -> None:
    tab = window.label_tab
    tab.view_mode.setCurrentIndex(1)
    tab.view_mode.setCurrentIndex(0)

    assert tab.canvas.view_mode == "normalized"
    assert float(np.nanmax(tab.canvas.image.image)) <= 1.0


def test_contrast_controls_disabled_in_raw_view(window) -> None:
    """The dB window is meaningless for uncalibrated digits."""
    tab = window.label_tab
    tab.view_mode.setCurrentIndex(1)

    assert not tab.level_low.isEnabled()
    assert "raw data" in tab.level_warning.text()

    tab.view_mode.setCurrentIndex(0)
    assert tab.level_low.isEnabled()
    assert tab.level_warning.text() == ""


def test_view_mode_persists_across_files(window, qapp) -> None:
    tab = window.label_tab
    tab.view_mode.setCurrentIndex(1)
    tab.queue.step(1)
    pump(qapp, 0.9)
    assert tab.canvas.view_mode == "raw"


def test_pixel_inspector_reports_all_three_stages(window) -> None:
    tab = window.label_tab
    sample = tab.canvas.sample_at(40, 200)

    assert sample["raw"] is not None
    assert 0.0 <= sample["normalized"] <= 1.0
    # -1..8 dB window: the reported dB must invert the normalization exactly.
    assert sample["db"] == pytest.approx(sample["normalized"] * 9.0 - 1.0)
    assert sample["mhz"] > 0
    assert sample["time_label"]

    tab._on_cursor_moved(sample)
    assert tab.inspector._values["raw"].text() != "-"
    assert "MHz" in tab.inspector._values["mhz"].text()


def test_inspector_flags_saturated_samples(window) -> None:
    tab = window.label_tab
    tab._on_cursor_moved(
        {
            "row": 0, "column": 0, "raw": 200.0, "normalized": 1.0, "db": 8.0,
            "mhz": 60.0, "seconds": 0.0, "time_label": "12:00:00", "clipped": True,
        }
    )
    assert "Saturated" in tab.inspector.note.text()


def test_header_panel_is_populated_on_load(window) -> None:
    assert "PRIMARY HEADER" in window.label_tab.header_panel.text.toPlainText()
    assert window.label_tab.header_panel.path_label.text().endswith(".fit.gz")


# -- reset: view and layout ------------------------------------------------


def test_reset_view_restores_defaults_without_touching_labels(window) -> None:
    tab = window.label_tab
    tab._on_box_created(20, 70, 300, 520)
    tab.level_high.setValue(0.4)
    tab.view_mode.setCurrentIndex(1)
    tab.colormap.setCurrentIndex(2)
    tab.queue.search.setText("ALASKA")

    window._reset_view_and_layout()

    assert tab.canvas.levels == (0.0, 1.0)
    assert tab.canvas.view_mode == "normalized"
    assert tab.colormap.currentIndex() == 0
    assert tab.queue.search.text() == ""
    # Labels survive: this reset is about display only.
    assert len(window.repository.boxes_for_file(tab._current_file_id)) == 1


# -- reset: one file -------------------------------------------------------


def test_clear_current_file_labels(window) -> None:
    tab = window.label_tab
    file_id = tab._current_file_id
    tab._on_box_created(20, 70, 300, 520)
    tab._on_box_created(90, 140, 700, 860)
    window.repository.set_notes(file_id, "a note")

    assert tab.clear_current_file_labels()

    record = window.repository.get_file(file_id)
    assert record.verdict is None
    assert record.status == STATUS_PENDING
    assert record.notes is None
    assert window.repository.boxes_for_file(file_id) == []
    assert tab.canvas._rois == {}
    # The file itself stays imported.
    assert window.repository.total_files() == 3


def test_clearing_one_file_leaves_others_alone(window, qapp) -> None:
    tab = window.label_tab
    first = tab._current_file_id
    tab._on_box_created(20, 70, 300, 520)

    tab.queue.step(1)
    pump(qapp, 0.9)
    second = tab._current_file_id
    tab._on_box_created(10, 60, 100, 300)

    tab.clear_current_file_labels()

    assert window.repository.boxes_for_file(second) == []
    assert len(window.repository.boxes_for_file(first)) == 1


# -- reset: display cache --------------------------------------------------


def test_clear_display_cache_empties_memory_and_disk(window, qapp) -> None:
    tab = window.label_tab
    pump(qapp, 1.5)  # let prefetch populate both caches
    assert len(tab.loader.cache) > 0

    window._clear_display_cache()

    assert len(tab.loader.cache) == 0
    assert tab.loader.disk_cache is not None
    assert list(Path(tab.loader.disk_cache.directory).glob("*.npy")) == []


# -- reset: whole dataset --------------------------------------------------


def test_reset_dataset_deletes_everything(tmp_path: Path, axes_files) -> None:
    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(p) for p in axes_files[:2]])
    for record in repo.files():
        repo.set_verdict(record.id, VERDICT_BURST)
        repo.add_box(record.id, 0, 20, 0, 100, "Type III")
    repo.set_state("queue.current_file_id", "1")

    repo.reset_dataset()

    assert repo.total_files() == 0
    assert repo.total_boxes(confirmed_only=False) == 0
    assert repo.get_state("queue.current_file_id") is None


def test_reset_dataset_writes_a_recoverable_backup(tmp_path: Path, axes_files) -> None:
    database_path = tmp_path / "annotations.db"
    repo = AnnotationRepository(Database(database_path))
    import_files(repo, [Path(p) for p in axes_files[:2]])
    record_id = repo.files()[0].id
    repo.set_verdict(record_id, VERDICT_BURST)
    repo.add_box(record_id, 0, 20, 0, 100, "Type II")

    backup_path = tmp_path / "backup.db"
    returned = repo.reset_dataset(backup_path)

    assert returned == backup_path and backup_path.exists()
    assert repo.total_files() == 0

    # The backup must be a working database with the work still in it.
    restored = AnnotationRepository(Database(backup_path))
    assert restored.total_files() == 2
    assert restored.total_boxes() == 1
    assert restored.boxes_for_file(record_id)[0].burst_type == "Type II"


def test_ids_restart_after_reset(tmp_path: Path, axes_files) -> None:
    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(p) for p in axes_files[:2]])
    repo.reset_dataset()
    import_files(repo, [Path(p) for p in axes_files[:1]])

    assert repo.files()[0].id == 1


def test_reset_dataset_returns_the_ui_to_an_empty_state(window) -> None:
    tab = window.label_tab
    tab._on_box_created(20, 70, 300, 520)

    window.repository.reset_dataset()
    tab.reload_after_reset()

    assert tab._current_file_id is None
    assert tab._current_bundle is None
    assert tab.canvas._rois == {}
    assert tab.queue.model.rowCount() == 0
    assert tab.header_panel.text.toPlainText() == ""
    assert window.repository.total_files() == 0
