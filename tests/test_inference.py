"""Region-based cascade inference and the Predict tab."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.crops import (  # noqa: E402
    CropConfig,
    PixelBox,
    crop_from_normalized,
    normalize_full_spectrum,
    whole_file_box,
)
from callisto_trainer.core.fits_reader import read_fits_spectrum  # noqa: E402
from callisto_trainer.core.inference import (  # noqa: E402
    DEFAULT_REGION_THRESHOLD,
    CascadePredictor,
    FileResult,
    RegionResult,
    _dominant_type,
    predict_paths,
    write_csv,
    write_json,
)
from callisto_trainer.core.models.model_factory import create_model  # noqa: E402
from callisto_trainer.settings import AppSettings  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config() -> dict:
    return load_config()


def _checkpoint(tmp_path: Path, task: str) -> Path:
    """A real, untrained model saved as a checkpoint with a full config."""
    config = load_config()
    is_type = task == "type"
    config["data"]["classes"] = (
        {"Type II": 0, "Type III": 1, "Other": 2} if is_type else {"No_Burst": 0, "Burst": 1}
    )
    config["model"].update(
        {
            "name": "simple_cnn",
            "in_channels": 1,
            "num_classes": 3 if is_type else 1,
            "use_metadata": False,
        }
    )
    config["training"]["threshold"] = 0.5

    model = create_model("simple_cnn", in_channels=1, num_classes=3 if is_type else 1)
    path = tmp_path / f"{task}.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


# -- construction ----------------------------------------------------------


def test_predictor_requires_at_least_one_model(config: dict) -> None:
    with pytest.raises(ValueError, match="At least one checkpoint"):
        CascadePredictor(config)


def test_binary_only_predicts_without_regions(tmp_path: Path, config: dict, any_real_file) -> None:
    predictor = CascadePredictor(config, binary_checkpoint=_checkpoint(tmp_path, "binary"))
    result = predictor.predict_file(any_real_file)

    assert result.predicted_label in ("Burst", "No_Burst")
    assert 0.0 <= result.burst_probability <= 1.0
    assert result.alert_level
    assert result.regions == [], "no type model, so nothing should be typed"


def test_type_only_produces_typed_regions(tmp_path: Path, config: dict, any_real_file) -> None:
    predictor = CascadePredictor(config, type_checkpoint=_checkpoint(tmp_path, "type"))
    result = predictor.predict_file(any_real_file)

    assert result.predicted_label is None, "no binary model, so no burst verdict"
    for region in result.regions:
        assert region.burst_type in ("Type II", "Type III", "Other")
        assert 0.0 <= region.type_confidence <= 1.0
        assert pytest.approx(sum(region.type_probabilities.values()), abs=1e-4) == 1.0


# -- the crop contract -----------------------------------------------------


def test_type_model_receives_a_crop_not_the_whole_file(
    tmp_path: Path, config: dict, any_real_file, monkeypatch
) -> None:
    """The heart of this module: stage 2 must see a training-shaped crop.

    Feeding it the whole file would be the train/serve mismatch this design
    exists to avoid, so the tensor handed to the type model is captured and
    compared against a crop of the region it reported.
    """
    predictor = CascadePredictor(config, type_checkpoint=_checkpoint(tmp_path, "type"))
    captured: list[np.ndarray] = []
    original = predictor._classify_crop

    def spy(tensor, unified=False, physics=None):
        captured.append(np.array(tensor, copy=True))
        return original(tensor, unified=unified, physics=physics)

    monkeypatch.setattr(predictor, "_classify_crop", spy)
    result = predictor.predict_file(any_real_file)
    if not result.regions:
        pytest.skip("no region located in this file")

    spectrum, _ = read_fits_spectrum(any_real_file)
    normalized = normalize_full_spectrum(spectrum, config)
    region = result.regions[0]
    expected = crop_from_normalized(
        normalized,
        PixelBox(region.row0, region.row1, region.col0, region.col1),
        CropConfig.from_config(config),
    )

    assert np.array_equal(captured[0], expected)
    whole = crop_from_normalized(
        normalized, whole_file_box(normalized.shape), CropConfig.from_config(config),
        apply_margin=False,
    )
    assert not np.array_equal(captured[0], whole), "stage 2 must not see the whole file"


def test_binary_model_receives_the_whole_file(
    tmp_path: Path, config: dict, any_real_file
) -> None:
    """Stage 1 was trained on whole files, so it must be given one."""
    predictor = CascadePredictor(config, binary_checkpoint=_checkpoint(tmp_path, "binary"))
    spectrum, _ = read_fits_spectrum(any_real_file)
    normalized = normalize_full_spectrum(spectrum, config)

    captured: list[np.ndarray] = []
    real_model = predictor.binary_model

    def spy(*inputs):
        captured.append(inputs[0].cpu().numpy())
        return real_model(*inputs)

    predictor.binary_model = spy
    predictor._score_binary(normalized, {})

    expected = crop_from_normalized(
        normalized, whole_file_box(normalized.shape), CropConfig.from_config(config),
        apply_margin=False,
    )
    assert np.array_equal(captured[0][0], expected)


def test_regions_carry_physical_coordinates(
    tmp_path: Path, config: dict, any_real_file
) -> None:
    predictor = CascadePredictor(config, type_checkpoint=_checkpoint(tmp_path, "type"))
    result = predictor.predict_file(any_real_file)
    if not result.regions:
        pytest.skip("no region located in this file")

    for region in result.regions:
        # Equal for a one-channel region -- a narrowband carrier, which is
        # exactly the kind of candidate the finder does propose.
        assert region.freq_lo_mhz <= region.freq_hi_mhz
        assert region.t_start_s <= region.t_end_s
        assert region.row0 < region.row1 and region.col0 < region.col1


# -- gating ----------------------------------------------------------------


def test_no_burst_files_skip_region_typing(tmp_path: Path, config: dict, any_real_file) -> None:
    """When the binary model says no burst, stage 2 must not run at all."""
    predictor = CascadePredictor(
        config,
        binary_checkpoint=_checkpoint(tmp_path, "binary"),
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    predictor.decision_threshold = 1.1  # force every file to No_Burst
    result = predictor.predict_file(any_real_file)

    assert result.predicted_label == "No_Burst"
    assert result.regions == []


def test_burst_files_run_region_typing(tmp_path: Path, config: dict, any_real_file) -> None:
    predictor = CascadePredictor(
        config,
        binary_checkpoint=_checkpoint(tmp_path, "binary"),
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    predictor.decision_threshold = -0.1  # force every file to Burst
    result = predictor.predict_file(any_real_file)

    assert result.predicted_label == "Burst"
    assert result.confidence == pytest.approx(result.burst_probability)


# -- robustness ------------------------------------------------------------


def test_unreadable_file_is_reported_not_raised(tmp_path: Path, config: dict) -> None:
    broken = tmp_path / "broken.fit.gz"
    broken.write_bytes(b"not FITS at all")

    predictor = CascadePredictor(config, type_checkpoint=_checkpoint(tmp_path, "type"))
    result = predictor.predict_file(broken)

    assert result.error
    assert result.predicted_label is None
    assert result.region_summary == "error"


def test_batch_continues_past_a_bad_file(tmp_path: Path, config: dict, axes_files) -> None:
    broken = tmp_path / "broken.fit.gz"
    broken.write_bytes(b"nope")
    paths = [axes_files[0], broken, axes_files[1]]

    results = predict_paths(
        paths, config, binary_checkpoint=_checkpoint(tmp_path, "binary")
    )
    assert len(results) == 3
    assert results[1].error and not results[0].error and not results[2].error


def test_progress_callback_can_cancel(tmp_path: Path, config: dict, axes_files) -> None:
    seen: list[str] = []

    def progress(index, total, name):
        seen.append(name)
        return index < 1

    results = predict_paths(
        [Path(p) for p in axes_files[:4]],
        config,
        binary_checkpoint=_checkpoint(tmp_path, "binary"),
        progress=progress,
    )
    assert len(results) == 1
    assert len(seen) == 2


# -- summarising -----------------------------------------------------------


def test_dominant_type_prefers_the_largest_region() -> None:
    regions = [
        RegionResult(0, 10, 0, 10, area=100, peak=0.9, burst_type="Other", type_confidence=0.99),
        RegionResult(0, 50, 0, 50, area=2500, peak=0.7, burst_type="Type III", type_confidence=0.51),
    ]
    assert _dominant_type(regions) == "Type III"


def test_dominant_type_is_none_without_typed_regions() -> None:
    assert _dominant_type([]) is None
    assert _dominant_type([RegionResult(0, 5, 0, 5, area=25, peak=0.5)]) is None


def test_region_summary_wording() -> None:
    result = FileResult(file_path="/x/a.fit.gz", file_name="a.fit.gz", predicted_label="Burst")
    assert result.region_summary == "no region located"

    result.regions = [
        RegionResult(0, 5, 0, 5, 25, 0.9, burst_type="Type III"),
        RegionResult(0, 5, 0, 5, 25, 0.9, burst_type="Type III"),
        RegionResult(0, 5, 0, 5, 25, 0.9, burst_type="Type II"),
    ]
    assert "Type III x2" in result.region_summary
    assert "Type II" in result.region_summary

    quiet = FileResult(file_path="/x/b.fit.gz", file_name="b.fit.gz", predicted_label="No_Burst")
    assert quiet.region_summary == "-"


# -- reports ---------------------------------------------------------------


def test_csv_has_one_row_per_file_with_regions_as_json(
    tmp_path: Path, config: dict, axes_files
) -> None:
    results = predict_paths(
        [Path(p) for p in axes_files[:2]],
        config,
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    path = write_csv(results, tmp_path / "out.csv")

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    for row, result in zip(rows, results):
        assert row["file_name"] == result.file_name
        assert int(row["region_count"]) == len(result.regions)
        parsed = json.loads(row["regions"])
        assert len(parsed) == len(result.regions)
        for entry in parsed:
            assert len(entry["pixel_box"]) == 4


def test_json_report_records_the_heuristic_caveat(
    tmp_path: Path, config: dict, axes_files
) -> None:
    results = predict_paths(
        [Path(p) for p in axes_files[:1]],
        config,
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    payload = json.loads(write_json(results, tmp_path / "out.json").read_text(encoding="utf-8"))

    assert "not by a trained detector" in payload["region_finder"]["note"]
    assert len(payload["files"]) == 1
    assert "regions" in payload["files"][0]


# -- calibration -----------------------------------------------------------


def test_adaptive_threshold_only_relaxes_never_tightens() -> None:
    from callisto_trainer.core.inference import ADAPTIVE_FLOOR, resolve_threshold

    faint = np.zeros((100, 1000), dtype=np.float32)
    faint[40:50, 400:460] = 0.26  # whole burst sits below the absolute setting
    bright = np.zeros((100, 1000), dtype=np.float32)
    bright[40:50, 400:460] = 0.95

    assert resolve_threshold(faint, absolute=0.35) < 0.35, "faint files must relax"
    assert resolve_threshold(bright, absolute=0.35) == pytest.approx(0.35), (
        "bright files must not be made more permissive"
    )
    assert resolve_threshold(faint, absolute=0.35) >= ADAPTIVE_FLOOR


def test_adaptive_can_be_switched_off() -> None:
    from callisto_trainer.core.inference import resolve_threshold

    faint = np.zeros((100, 1000), dtype=np.float32)
    faint[40:50, 400:460] = 0.26
    assert resolve_threshold(faint, absolute=0.35, adaptive=False) == pytest.approx(0.35)


def test_adaptive_threshold_finds_a_faint_burst_a_fixed_one_misses() -> None:
    from callisto_trainer.core.inference import resolve_threshold
    from callisto_trainer.services.assist import find_candidate_regions

    spectrum = np.zeros((100, 1000), dtype=np.float32)
    spectrum[40:52, 400:470] = 0.28  # clearly a feature, but below 0.35

    assert find_candidate_regions(spectrum, threshold=0.35) == []
    adaptive = resolve_threshold(spectrum, absolute=0.35)
    assert len(find_candidate_regions(spectrum, threshold=adaptive)) == 1


def test_predictor_honours_the_adaptive_flag(
    tmp_path: Path, config: dict, any_real_file
) -> None:
    predictor = CascadePredictor(
        config, type_checkpoint=_checkpoint(tmp_path, "type"), adaptive_threshold=False
    )
    assert predictor.adaptive_threshold is False
    predictor.predict_file(any_real_file)  # must still run


def test_default_threshold_locates_regions_in_known_bursts(axes_files, config: dict) -> None:
    """The default must actually find the bursts it will be pointed at.

    0.45 missed most files in a quieter station subset, which is why the default
    is 0.35. This pins that decision to observable behaviour.
    """
    from callisto_trainer.services.assist import find_candidate_regions

    found = 0
    for path in axes_files:
        spectrum, _ = read_fits_spectrum(path)
        normalized = normalize_full_spectrum(spectrum, config)
        if find_candidate_regions(normalized, threshold=DEFAULT_REGION_THRESHOLD):
            found += 1

    assert found >= len(axes_files) * 0.7, (
        f"the default threshold located regions in only {found}/{len(axes_files)} "
        "known burst files"
    )


# -- Predict tab -----------------------------------------------------------


@pytest.fixture
def predict_tab(qapp, tmp_path: Path):
    from callisto_trainer.ui.predict_tab import PredictTab

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    settings.ensure_directories()
    tab = PredictTab(settings)
    yield tab
    tab.shutdown()


def test_tab_warns_when_the_binary_gate_is_missing(predict_tab, tmp_path: Path) -> None:
    """Quiet files produce as many regions as bursts; the UI must say so."""
    predict_tab.type_model.addItem("type", _checkpoint(tmp_path, "type"))
    predict_tab.type_model.setCurrentIndex(predict_tab.type_model.count() - 1)

    assert "⚠" in predict_tab.gate_warning.text()
    assert "quiet ones" in predict_tab.gate_warning.text()


def test_tab_notes_when_only_the_binary_model_is_chosen(predict_tab, tmp_path: Path) -> None:
    predict_tab.binary_model.addItem("binary", _checkpoint(tmp_path, "binary"))
    predict_tab.binary_model.setCurrentIndex(predict_tab.binary_model.count() - 1)

    assert "no regions located" in predict_tab.gate_warning.text()


def test_tab_has_no_warning_with_both_models(predict_tab, tmp_path: Path) -> None:
    predict_tab.binary_model.addItem("binary", _checkpoint(tmp_path, "binary"))
    predict_tab.binary_model.setCurrentIndex(predict_tab.binary_model.count() - 1)
    predict_tab.type_model.addItem("type", _checkpoint(tmp_path, "type"))
    predict_tab.type_model.setCurrentIndex(predict_tab.type_model.count() - 1)

    assert predict_tab.gate_warning.text() == ""


def test_run_is_blocked_without_files_or_models(predict_tab) -> None:
    assert not predict_tab.run_button.isEnabled()


def test_results_table_populates(predict_tab, tmp_path: Path, axes_files, qapp) -> None:
    results = predict_paths(
        [Path(p) for p in axes_files[:2]],
        predict_tab.settings.pipeline,
        binary_checkpoint=_checkpoint(tmp_path, "binary"),
    )
    predict_tab._on_finished(results)

    assert predict_tab.table.rowCount() == 2
    assert predict_tab.export_csv.isEnabled()
    assert "file(s)" in predict_tab.summary.text()


def test_worker_is_released_only_after_its_thread_stops(
    predict_tab, tmp_path: Path, axes_files, qapp
) -> None:
    """Regression: clearing the worker inside a result handler crashed Qt.

    Dropping the last reference from ``_on_finished`` destroyed the QThread while
    ``run()`` was still returning -- "QThread: Destroyed while thread is still
    running". The reference must survive until ``QThread.finished``.
    """
    predict_tab.unified_model.addItem("unified", _checkpoint(tmp_path, "type"))
    predict_tab._add_paths([Path(axes_files[0])])

    # The handlers must not clear the reference themselves.
    predict_tab._worker = object()
    predict_tab._on_finished([])
    assert predict_tab._worker is not None, "results arriving must not destroy the thread"

    predict_tab._worker = object()
    predict_tab._on_failed("boom")
    assert predict_tab._worker is not None, "a failure must not destroy the thread either"

    predict_tab._worker = None
    predict_tab._release_worker()
    assert predict_tab._worker is None


def test_release_worker_is_connected_to_thread_finished(predict_tab) -> None:
    """The only safe clearing point is QThread.finished; make sure it is wired."""
    import inspect

    source = inspect.getsource(type(predict_tab)._run)
    assert "finished.connect(self._release_worker)" in source


def test_clear_results_empties_the_view(predict_tab, tmp_path: Path, axes_files, qapp) -> None:
    results = predict_paths(
        [Path(p) for p in axes_files[:2]],
        predict_tab.settings.pipeline,
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    predict_tab._on_finished(results)
    predict_tab.table.setCurrentCell(0, 0)
    qapp.processEvents()
    assert predict_tab.table.rowCount() == 2
    assert predict_tab.clear_results_button.isEnabled()

    predict_tab.clear_results()

    assert predict_tab.table.rowCount() == 0
    assert predict_tab._results == []
    assert predict_tab.canvas._rois == {}
    assert predict_tab.summary.text() == ""
    assert predict_tab.region_detail.text() == ""
    assert "Select a result" in predict_tab.preview_title.text()
    for button in (
        predict_tab.clear_results_button,
        predict_tab.export_csv,
        predict_tab.export_json,
    ):
        assert not button.isEnabled()


def test_clear_results_keeps_models_and_settings(
    predict_tab, tmp_path: Path, axes_files, qapp
) -> None:
    """Clearing is for loading another batch, so the configuration must survive."""
    checkpoint = _checkpoint(tmp_path, "type")
    predict_tab.type_model.addItem("type", checkpoint)
    predict_tab.type_model.setCurrentIndex(predict_tab.type_model.count() - 1)
    predict_tab.sensitivity.setValue(0.28)
    predict_tab.min_area.setValue(120)
    predict_tab._add_paths([Path(p) for p in axes_files[:2]])

    predict_tab._on_finished(
        predict_paths(
            [Path(p) for p in axes_files[:1]],
            predict_tab.settings.pipeline,
            type_checkpoint=checkpoint,
        )
    )
    predict_tab.clear_results()

    assert predict_tab.type_model.currentData() == checkpoint
    assert predict_tab.sensitivity.value() == pytest.approx(0.28)
    assert predict_tab.min_area.value() == 120
    assert len(predict_tab._paths) == 2, "the selected files must remain queued"


def test_clearing_an_empty_view_is_harmless(predict_tab) -> None:
    predict_tab.clear_results()
    predict_tab.clear_results()
    assert predict_tab._results == []
    assert predict_tab.table.rowCount() == 0


def test_selecting_a_result_draws_its_regions(
    predict_tab, tmp_path: Path, axes_files, qapp
) -> None:
    results = predict_paths(
        [Path(p) for p in axes_files[:1]],
        predict_tab.settings.pipeline,
        type_checkpoint=_checkpoint(tmp_path, "type"),
    )
    predict_tab._on_finished(results)
    predict_tab.table.setCurrentCell(0, 0)
    qapp.processEvents()

    assert len(predict_tab.canvas._rois) == len(results[0].regions)
    # Candidates, not labels: they must render dashed.
    assert all(not roi.confirmed for roi in predict_tab.canvas._rois.values())
