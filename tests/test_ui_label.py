"""Labelling workspace behaviour, driven headless through the real widgets.

These run against actual FITS files so the canvas, coordinate mapping and crop
preview are exercised on real data rather than mocks.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.services.importer import import_files  # noqa: E402
from callisto_trainer.settings import AppSettings  # noqa: E402
from callisto_trainer.store.repository import (  # noqa: E402
    VERDICT_BURST,
    VERDICT_NO_BURST,
)


@pytest.fixture(scope="session")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


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
    yield main
    main.label_tab.shutdown()
    main.close()


def pump(qapp, seconds: float = 1.2) -> None:
    """Spin the event loop until background decodes land."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def loaded(window, qapp):
    window.label_tab.queue.select_row(0)
    pump(qapp)
    assert window.label_tab._current_bundle is not None, "spectrum never finished loading"
    return window


# -- loading ---------------------------------------------------------------


def test_spectrum_loads_with_axes(loaded, qapp) -> None:
    tab = loaded.label_tab
    bundle = tab._current_bundle

    assert bundle.normalized.ndim == 2
    assert bundle.axes.n_freq == bundle.normalized.shape[0]
    assert bundle.axes.n_time == bundle.normalized.shape[1]
    assert bundle.axes.freq_descending
    assert tab.title.text().startswith(bundle.path.rsplit("\\", 1)[-1][:6])


def test_navigation_moves_through_the_queue(loaded, qapp) -> None:
    tab = loaded.label_tab
    first = tab._current_file_id
    tab.queue.step(1)
    pump(qapp, 0.8)
    assert tab._current_file_id != first

    tab.queue.step(-1)
    pump(qapp, 0.8)
    assert tab._current_file_id == first


def test_prefetch_warms_neighbouring_files(loaded, qapp) -> None:
    """Next must be instant, which means the next file is already decoded."""
    tab = loaded.label_tab
    pump(qapp, 2.0)

    next_id = tab.queue.model.id_at(1)
    assert next_id is not None
    assert tab.loader.cached(next_id) is not None, "prefetch did not warm the next file"


# -- annotation ------------------------------------------------------------


def test_drawing_a_box_stores_pixels_and_physical_units(loaded) -> None:
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)

    boxes = loaded.repository.boxes_for_file(tab._current_file_id)
    assert len(boxes) == 1
    box = boxes[0]

    assert (box.row0, box.row1, box.col0, box.col1) == (20, 70, 300, 520)
    assert box.freq_lo_mhz < box.freq_hi_mhz
    assert box.t_start_s < box.t_end_s
    # Descending frequency axis: the top row of the box is the higher frequency.
    axes = tab._current_bundle.axes
    assert box.freq_hi_mhz == pytest.approx(float(axes.freq_mhz[20]), abs=0.01)


def test_drawing_a_box_implies_the_file_has_a_burst(loaded) -> None:
    tab = loaded.label_tab
    assert loaded.repository.get_file(tab._current_file_id).verdict is None

    tab._on_box_created(20, 70, 300, 520)
    assert loaded.repository.get_file(tab._current_file_id).verdict == VERDICT_BURST


def test_multiple_boxes_can_carry_different_types(loaded) -> None:
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    tab._on_box_created(90, 140, 700, 860)

    boxes = loaded.repository.boxes_for_file(tab._current_file_id)
    assert len(boxes) == 2

    tab.panel.select_box(boxes[0].id)
    tab._on_type_assigned("Type II")
    tab.panel.select_box(boxes[1].id)
    tab._on_type_assigned("Other")

    types = {b.id: b.burst_type for b in loaded.repository.boxes_for_file(tab._current_file_id)}
    assert types[boxes[0].id] == "Type II"
    assert types[boxes[1].id] == "Other"
    assert loaded.repository.box_type_counts() == {"Type II": 1, "Other": 1}


def test_new_box_inherits_the_previous_type(loaded) -> None:
    """Marking several bursts of one type in a file should not need re-picking."""
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    boxes = loaded.repository.boxes_for_file(tab._current_file_id)
    tab.panel.select_box(boxes[0].id)
    tab._on_type_assigned("Type II")

    tab._on_box_created(90, 140, 700, 860)
    newest = loaded.repository.boxes_for_file(tab._current_file_id)[-1]
    assert newest.burst_type == "Type II"


def test_editing_a_box_updates_physical_coordinates(loaded) -> None:
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = loaded.repository.boxes_for_file(tab._current_file_id)[0]
    before = box.freq_lo_mhz

    tab._on_box_edited(box.id, 20, 120, 300, 520)
    after = loaded.repository.boxes_for_file(tab._current_file_id)[0]

    assert after.row1 == 120
    assert after.freq_lo_mhz < before, "extending downward should reach lower frequencies"


def test_deleting_a_box_removes_it_from_store_and_canvas(loaded) -> None:
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box_id = loaded.repository.boxes_for_file(tab._current_file_id)[0].id

    tab._on_box_deleted(box_id)

    assert loaded.repository.boxes_for_file(tab._current_file_id) == []
    assert box_id not in tab.canvas._rois


def test_canvas_shows_a_roi_per_stored_box(loaded, qapp) -> None:
    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    tab._on_box_created(90, 140, 700, 860)

    file_id = tab._current_file_id
    tab.queue.step(1)
    pump(qapp, 0.8)
    tab.queue.step(-1)
    pump(qapp, 0.8)

    assert tab._current_file_id == file_id
    assert len(tab.canvas._rois) == 2, "stored boxes must be redrawn when revisiting a file"


# -- crop preview ----------------------------------------------------------


def test_crop_preview_matches_the_exported_tensor(loaded) -> None:
    """What the operator previews must be exactly what training will receive."""
    from callisto_trainer.core.crops import PixelBox, crop_from_normalized

    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = loaded.repository.boxes_for_file(tab._current_file_id)[0]
    tab.panel.select_box(box.id)

    expected = crop_from_normalized(
        tab._current_bundle.normalized,
        PixelBox(box.row0, box.row1, box.col0, box.col1),
        tab.crop_config,
    )
    shown = tab.panel.preview.image.image

    assert shown is not None
    assert np.array_equal(shown, expected[0])
    assert "224x224" in tab.panel.preview.caption.text()


def test_crop_preview_shows_the_drawn_region_and_nothing_more(loaded) -> None:
    """The preview is only trustworthy if it stops at the edges of the box."""
    from callisto_trainer.core.preprocess import resize_spectrum

    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = loaded.repository.boxes_for_file(tab._current_file_id)[0]
    tab.panel.select_box(box.id)

    patch = tab._current_bundle.normalized[box.row0 : box.row1, box.col0 : box.col1]
    exact = resize_spectrum(patch, target_shape=tab.crop_config.target_shape)

    assert np.array_equal(tab.panel.preview.image.image, exact)
    assert "50 x 220 px exactly" in tab.panel.preview.caption.text()


# -- verdicts --------------------------------------------------------------


def test_no_burst_verdict_with_boxes_asks_before_discarding(loaded, monkeypatch) -> None:
    """Boxes on a no_burst file would never be exported; do not drop them silently."""
    from PySide6.QtWidgets import QMessageBox

    tab = loaded.label_tab
    tab._on_box_created(20, 70, 300, 520)

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel)
    )
    tab._on_verdict_changed(VERDICT_NO_BURST)

    record = loaded.repository.get_file(tab._current_file_id)
    assert record.verdict == VERDICT_BURST, "cancelling must leave the file unchanged"
    assert len(loaded.repository.boxes_for_file(tab._current_file_id)) == 1

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    tab._on_verdict_changed(VERDICT_NO_BURST)

    record = loaded.repository.get_file(tab._current_file_id)
    assert record.verdict == VERDICT_NO_BURST
    assert loaded.repository.boxes_for_file(tab._current_file_id) == []


def test_no_burst_on_a_clean_file_needs_no_confirmation(loaded) -> None:
    tab = loaded.label_tab
    tab._on_verdict_changed(VERDICT_NO_BURST)
    assert loaded.repository.get_file(tab._current_file_id).verdict == VERDICT_NO_BURST


# -- display contract ------------------------------------------------------


def test_default_contrast_is_exactly_the_model_view(loaded) -> None:
    tab = loaded.label_tab
    assert tab.canvas.levels == (0.0, 1.0)
    assert tab.canvas.levels_match_model
    assert tab.level_warning.text() == ""


def test_changing_contrast_raises_a_warning_badge(loaded) -> None:
    tab = loaded.label_tab
    tab.level_high.setValue(0.4)

    assert not tab.canvas.levels_match_model
    assert tab.level_warning.text() != ""

    tab._reset_levels()
    assert tab.canvas.levels_match_model
    assert tab.level_warning.text() == ""


def test_rendered_window_covers_the_visible_range(loaded) -> None:
    """The level-of-detail renderer must place the image in original columns."""
    tab = loaded.label_tab
    n_freq, n_time = tab._current_bundle.shape

    rect = tab.canvas.image.boundingRect()
    mapped = tab.canvas.image.mapRectToView(rect)
    assert mapped.width() == pytest.approx(n_time, rel=0.02)
    assert mapped.height() == pytest.approx(n_freq, rel=0.02)


# -- persistence -----------------------------------------------------------


def test_session_resumes_on_the_same_file(window, qapp, tmp_path: Path) -> None:
    from callisto_trainer.ui.main_window import MainWindow

    window.label_tab.queue.select_row(1)
    pump(qapp, 0.8)
    expected_id = window.label_tab._current_file_id
    window.label_tab._on_box_created(30, 80, 200, 400)
    settings = window.settings
    window.label_tab.shutdown()
    window.close()

    revived = MainWindow(settings)
    pump(qapp, 1.2)
    try:
        assert revived.label_tab._current_file_id == expected_id
        assert len(revived.repository.boxes_for_file(expected_id)) == 1
    finally:
        revived.label_tab.shutdown()
        revived.close()
