"""Burst parameters of a drawn box: from its geometry, for Type II and III only.

The box's height is the frequency range and its width the duration; a Type II or
Type III drifts from high to low frequency, so the burst starts at the top of the
box and df/dt = (f_start - f_end) / (t_start - t_end) is negative.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.burst_physics import (  # noqa: E402
    BOX_CONFIDENCE,
    NO_BOX_PARAMETERS,
    box_parameters,
    box_parameters_current,
    physics_from_row,
)
from callisto_trainer.core.coords import SpectrumAxes, box_to_physical  # noqa: E402
from callisto_trainer.core.taxonomy import OTHER, TYPE_II, TYPE_III, TYPE_IIIG, TYPE_IV  # noqa: E402

SHAPE = (200, 3600)
AXES = SpectrumAxes(time_s=np.arange(SHAPE[1]) * 0.25, freq_mhz=np.linspace(80.0, 20.0, SHAPE[0]))


def _expected(row0, row1, col0, col1):
    bounds = box_to_physical(AXES, row0, row1, col0, col1)
    f_start, f_end = bounds["freq_hi_mhz"], bounds["freq_lo_mhz"]
    t_start, t_end = bounds["t_start_s"], bounds["t_end_s"]
    return f_start, f_end, t_start, t_end, (f_start - f_end) / (t_start - t_end)


@pytest.mark.parametrize("burst_type", [TYPE_II, TYPE_III])
def test_the_drift_is_the_box_diagonal(burst_type: str) -> None:
    box = (10, 150, 400, 480)
    f_start, f_end, t_start, t_end, drift = _expected(*box)
    physics = box_parameters(AXES, *box, burst_type)

    assert physics.freq_start_mhz == pytest.approx(f_start) and f_start > f_end
    assert physics.freq_end_mhz == pytest.approx(f_end)
    assert physics.time_start_s == pytest.approx(t_start)
    assert physics.time_end_s == pytest.approx(t_end)
    assert physics.duration_s == pytest.approx(t_end - t_start)
    assert physics.bandwidth_mhz == pytest.approx(f_start - f_end)
    assert physics.drift_mhz_per_s == pytest.approx(drift)
    assert physics.drift_mhz_per_s < 0, "high to low frequency"
    assert physics.relative_drift_per_s == pytest.approx(drift / (0.5 * (f_start + f_end)))
    assert physics.measured and physics.from_box and physics.confidence == BOX_CONFIDENCE


def test_a_type_iii_box_drifts_faster_than_a_type_ii_box_of_the_same_band() -> None:
    iii = box_parameters(AXES, 10, 150, 400, 440, TYPE_III)     # 10 s
    ii = box_parameters(AXES, 10, 150, 400, 1600, TYPE_II)      # 5 minutes
    assert abs(iii.drift_mhz_per_s) > 10 * abs(ii.drift_mhz_per_s)


@pytest.mark.parametrize("burst_type", [TYPE_IIIG, TYPE_IV, OTHER, None])
def test_other_types_get_no_parameters(burst_type) -> None:
    physics = box_parameters(AXES, 10, 150, 400, 480, burst_type)
    assert not physics.measured
    assert physics.freq_start_mhz is None and physics.duration_s is None
    assert physics.note == NO_BOX_PARAMETERS


def test_a_group_box_still_counts_its_bursts() -> None:
    """The count drives the Type III / IIIG hint, so both types keep it."""
    rng = np.random.default_rng(0)
    array = np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)
    for start in (400, 440, 490, 530):
        for row in range(10, 150):
            col = int(start + (row - 10) * 0.08)
            array[row, col:col + 6] = 0.8
    group = box_parameters(AXES, 5, 160, 390, 560, TYPE_IIIG, normalized=array)
    single = box_parameters(AXES, 5, 160, 390, 560, TYPE_III, normalized=array)
    assert group.burst_count == single.burst_count >= 3
    assert box_parameters(AXES, 5, 160, 390, 560, TYPE_IV, normalized=array).burst_count == 0


def test_a_box_one_sample_wide_has_no_drift() -> None:
    physics = box_parameters(AXES, 10, 150, 400, 401, TYPE_III)
    assert not physics.measured and "more than one sample" in physics.note


def test_stored_parameters_from_an_earlier_method_are_recalculated() -> None:
    fitted = {"physics_confidence": "good", "drift_mhz_per_s": -1.2}
    assert not box_parameters_current(fitted, TYPE_III)
    assert box_parameters_current({"physics_confidence": BOX_CONFIDENCE}, TYPE_III)
    assert not box_parameters_current(fitted, TYPE_IV), "a drift a Type IV must not keep"
    assert box_parameters_current({"physics_confidence": "none", "drift_mhz_per_s": None}, OTHER)
    assert not box_parameters_current({}, TYPE_II), "never measured"


def test_parameters_survive_the_database_columns() -> None:
    from callisto_trainer.services.physics_service import physics_to_columns

    physics = box_parameters(AXES, 10, 150, 400, 480, TYPE_III)
    restored = physics_from_row(physics_to_columns(physics))
    assert restored.from_box
    assert restored.drift_mhz_per_s == pytest.approx(physics.drift_mhz_per_s)
    assert restored.freq_start_mhz == pytest.approx(physics.freq_start_mhz)


# -- the Label tab ---------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def label_tab(qapp, tmp_path: Path, axes_files):
    from callisto_trainer.services.importer import import_files
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.main_window import MainWindow

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "data" / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    window = MainWindow(settings)
    import_files(window.repository, [Path(p) for p in axes_files[:1]])
    tab = window.label_tab
    tab.refresh_queue(keep_selection=False)
    tab.queue.select_row(0)
    deadline = time.monotonic() + 5
    while tab._current_bundle is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert tab._current_bundle is not None
    yield window
    tab.shutdown()
    window.close()


def _only_box(window):
    return window.repository.boxes_for_file(window.label_tab._current_file_id)[0]


def test_drawing_resizing_and_retyping_recalculate(label_tab) -> None:
    window, tab = label_tab, label_tab.label_tab
    axes = tab._current_bundle.axes
    tab._sticky_type = TYPE_III
    tab._on_box_created(20, 70, 300, 520)
    box = _only_box(window)
    assert box.burst_type == TYPE_III
    expected = box_parameters(axes, 20, 70, 300, 520, TYPE_III)
    assert box.physics["drift_mhz_per_s"] == pytest.approx(expected.drift_mhz_per_s)
    assert "From the box" in tab.panel.physics_note.text()

    tab._on_box_edited(box.id, 20, 70, 300, 900)
    wider = box_parameters(axes, 20, 70, 300, 900, TYPE_III)
    assert _only_box(window).physics["drift_mhz_per_s"] == pytest.approx(wider.drift_mhz_per_s)
    assert abs(wider.drift_mhz_per_s) < abs(expected.drift_mhz_per_s), "wider box, slower drift"

    tab.panel.select_box(box.id)
    tab._on_type_assigned(TYPE_IV)
    assert _only_box(window).physics["drift_mhz_per_s"] is None
    assert tab.panel.physics_labels["drift"].text() == "-"


def test_opening_a_file_recalculates_old_fitted_boxes(label_tab) -> None:
    window, tab = label_tab, label_tab.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = _only_box(window)
    tab.panel.select_box(box.id)
    tab._on_type_assigned(TYPE_II)
    window.repository.set_box_physics(
        box.id, {"drift_mhz_per_s": -9.9, "physics_confidence": "good", "track_axis": "time"}
    )
    tab._display(tab._current_bundle)
    stored = _only_box(window).physics
    assert stored["physics_confidence"] == BOX_CONFIDENCE
    assert stored["drift_mhz_per_s"] == pytest.approx(
        box_parameters(tab._current_bundle.axes, 20, 70, 300, 520, TYPE_II).drift_mhz_per_s
    )


# -- predictions -----------------------------------------------------------------


def test_a_predicted_type_iii_region_reports_the_box_drift(tmp_path: Path, monkeypatch) -> None:
    import torch

    from callisto_trainer.core.config import load_config
    from callisto_trainer.core.inference import CascadePredictor, FileResult
    from callisto_trainer.core.models.model_factory import create_model
    from callisto_trainer.core.region_features import feature_count
    from callisto_trainer.store.export import UNIFIED_CLASSES

    config = load_config()
    config["data"]["classes"] = dict(UNIFIED_CLASSES)
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(UNIFIED_CLASSES),
         "use_metadata": False, "use_physics": True,
         "views": ["crop", "context"], "feature_set": "region_v2"}
    )
    config["inference"] = {"burst_threshold": 0.5}
    model = create_model("simple_cnn", in_channels=1, num_classes=len(UNIFIED_CLASSES),
                         use_physics=True, num_physics=feature_count("region_v2"), num_views=2)
    path = tmp_path / "unified.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    predictor = CascadePredictor(load_config(), unified_checkpoint=path)

    original = predictor._unified_probabilities
    vector = np.array([0.9 if name == TYPE_III else 0.1 / (len(UNIFIED_CLASSES) - 1)
                       for name in predictor.unified_class_names])

    def fake(normalized, axes, boxes, rfi_channels=None, quiet=None, file_meta=None):
        _, encoded = original(
            normalized, axes, boxes, rfi_channels, quiet=quiet, file_meta=file_meta
        )
        return np.tile(vector, (len(boxes), 1)), encoded

    monkeypatch.setattr(predictor, "_unified_probabilities", fake)
    rng = np.random.default_rng(0)
    array = np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)
    for row in range(10, 150):
        col = int(400 + (row - 10) * 0.08)
        array[row, col:col + 6] = 0.8
    result = predictor.predict_normalized(array, AXES, FileResult("x.fit.gz", "x.fit.gz"))
    region = result.regions[0]
    assert region.burst_type == TYPE_III
    expected = box_parameters(AXES, region.row0, region.row1, region.col0, region.col1, TYPE_III)
    assert region.drift_mhz_per_s == pytest.approx(expected.drift_mhz_per_s)
    assert region.physics_confidence == BOX_CONFIDENCE
