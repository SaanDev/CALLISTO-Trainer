"""Pre-labelling assistance: candidate regions, scoring and UI integration."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.services.assist import (  # noqa: E402
    DEFAULT_MIN_AREA,
    Proposal,
    find_candidate_regions,
    find_latest_checkpoint,
    propose_from_normalized,
)
from callisto_trainer.services.importer import import_files  # noqa: E402
from callisto_trainer.settings import AppSettings  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config() -> dict:
    return load_config()


# -- candidate regions -----------------------------------------------------


def test_finds_a_single_bright_blob() -> None:
    array = np.zeros((200, 800), dtype=np.float32)
    array[50:90, 300:360] = 0.9

    proposals = find_candidate_regions(array)

    assert len(proposals) == 1
    found = proposals[0]
    assert (found.row0, found.row1) == (50, 90)
    assert (found.col0, found.col1) == (300, 360)
    assert found.area == 40 * 60
    assert found.peak == pytest.approx(0.9)


def test_separates_disconnected_blobs() -> None:
    array = np.zeros((200, 800), dtype=np.float32)
    array[20:60, 100:160] = 0.8
    array[120:170, 500:570] = 0.7

    proposals = find_candidate_regions(array)

    assert len(proposals) == 2
    # Sorted by area, largest first.
    assert proposals[0].area >= proposals[1].area


def test_specks_below_the_minimum_area_are_ignored() -> None:
    array = np.zeros((100, 100), dtype=np.float32)
    array[10, 10] = 1.0
    array[50:52, 50:52] = 1.0

    assert find_candidate_regions(array, min_area=DEFAULT_MIN_AREA) == []


def test_dim_features_below_the_threshold_are_ignored() -> None:
    array = np.full((100, 400), 0.2, dtype=np.float32)
    assert find_candidate_regions(array, threshold=0.45) == []


def test_candidate_count_is_capped() -> None:
    array = np.zeros((300, 900), dtype=np.float32)
    for index in range(10):
        array[index * 30 : index * 30 + 20, 100:200] = 0.9

    proposals = find_candidate_regions(array, max_candidates=4)
    assert len(proposals) == 4


def test_diagonal_pixels_count_as_connected() -> None:
    """Burst lanes drift, so 8-connectivity is required or they fragment."""
    array = np.zeros((60, 60), dtype=np.float32)
    for index in range(40):
        array[index : index + 3, index : index + 3] = 0.9

    proposals = find_candidate_regions(array, min_area=20)
    assert len(proposals) == 1, "a drifting lane must stay one region"


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        find_candidate_regions(np.zeros((2, 10, 10), dtype=np.float32))


def test_fallback_labeller_agrees_with_scipy() -> None:
    from callisto_trainer.services.assist import _label_connected, _label_connected_fallback

    rng = np.random.RandomState(3)
    mask = rng.rand(60, 90) > 0.7

    _, scipy_count = _label_connected(mask)
    _, fallback_count = _label_connected_fallback(mask)
    assert scipy_count == fallback_count


def test_proposal_converts_to_a_crop_box() -> None:
    box = Proposal(10, 40, 100, 220, area=3600, peak=0.9).as_box()
    assert box.as_tuple() == (10, 40, 100, 220)


def test_proposals_on_a_real_spectrum(any_real_file, config: dict) -> None:
    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum

    spectrum, _ = read_fits_spectrum(any_real_file)
    normalized = normalize_full_spectrum(spectrum, config)
    proposals = propose_from_normalized(normalized, config)

    for proposal in proposals:
        assert 0 <= proposal.row0 < proposal.row1 <= normalized.shape[0]
        assert 0 <= proposal.col0 < proposal.col1 <= normalized.shape[1]
        assert proposal.burst_type is None, "no checkpoint given, so no type is claimed"


# -- checkpoint discovery --------------------------------------------------


def test_latest_checkpoint_is_none_when_untrained(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "outputs", "type") is None


def test_latest_checkpoint_prefers_the_newest_run(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    for name in ("type_20260101_000000", "type_20260727_120000"):
        (outputs / name / "checkpoints").mkdir(parents=True)
        (outputs / name / "checkpoints" / "best.pt").write_bytes(b"x")
    (outputs / "binary_20260727_130000" / "checkpoints").mkdir(parents=True)
    (outputs / "binary_20260727_130000" / "checkpoints" / "best.pt").write_bytes(b"x")

    found = find_latest_checkpoint(outputs, "type")
    assert found is not None and "type_20260727_120000" in str(found)
    assert "binary" in str(find_latest_checkpoint(outputs, "binary"))


def test_run_without_a_checkpoint_is_skipped(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "type_20260727_120000" / "checkpoints").mkdir(parents=True)  # no best.pt
    assert find_latest_checkpoint(outputs, "type") is None


# -- label tab integration -------------------------------------------------


@pytest.fixture
def loaded(qapp, tmp_path: Path, axes_files):
    from callisto_trainer.ui.main_window import MainWindow

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "data" / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    window = MainWindow(settings)
    import_files(window.repository, [Path(p) for p in axes_files[:2]])
    window.label_tab.refresh_queue(keep_selection=False)
    window.label_tab.queue.select_row(0)

    deadline = time.monotonic() + 5
    while window.label_tab._current_bundle is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert window.label_tab._current_bundle is not None
    yield window
    window.label_tab.shutdown()
    window.close()


def test_suggestions_are_stored_unconfirmed(loaded) -> None:
    tab = loaded.label_tab
    tab.propose_boxes()

    boxes = loaded.repository.boxes_for_file(tab._current_file_id)
    if not boxes:
        pytest.skip("this file has no region above the brightness threshold")

    assert all(box.source == "assisted" for box in boxes)
    assert all(not box.confirmed for box in boxes)


def test_unconfirmed_suggestions_are_never_exported(loaded, config: dict, tmp_path: Path) -> None:
    """A suggestion must not become training data just by being suggested."""
    from callisto_trainer.store.export import export_type_dataset
    from callisto_trainer.store.repository import VERDICT_BURST

    tab = loaded.label_tab
    tab.propose_boxes()
    loaded.repository.set_verdict(tab._current_file_id, VERDICT_BURST)
    if not loaded.repository.boxes_for_file(tab._current_file_id):
        pytest.skip("no suggestions produced for this file")

    result = export_type_dataset(
        loaded.repository, tmp_path / "datasets", config, tmp_path / "outputs"
    )
    assert result.written == 0


def test_giving_a_suggestion_a_type_confirms_it(loaded) -> None:
    tab = loaded.label_tab
    tab.propose_boxes()
    boxes = loaded.repository.boxes_for_file(tab._current_file_id)
    if not boxes:
        pytest.skip("no suggestions produced for this file")

    tab.panel.select_box(boxes[0].id)
    tab._on_type_assigned("Type II")

    updated = loaded.repository.boxes_for_file(tab._current_file_id)[0]
    assert updated.confirmed
    assert updated.burst_type == "Type II"


def test_suggestions_do_not_duplicate_existing_boxes(loaded) -> None:
    tab = loaded.label_tab
    tab.propose_boxes()
    first_count = len(loaded.repository.boxes_for_file(tab._current_file_id))
    if first_count == 0:
        pytest.skip("no suggestions produced for this file")

    tab.propose_boxes()
    assert len(loaded.repository.boxes_for_file(tab._current_file_id)) == first_count


def test_suggesting_without_a_trained_model_claims_no_type(loaded) -> None:
    tab = loaded.label_tab
    assert tab._scorer is None
    tab.propose_boxes()
    assert "no model trained yet" in tab.readout.text() or "No candidate" in tab.readout.text()
