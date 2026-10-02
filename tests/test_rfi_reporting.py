"""RFI is reported with No_Burst as one outcome, and detected separately.

The operator's rules: a file with a burst is Burst whatever interference it also
holds, and lists that interference; a file with only interference is No_Burst.
RFI and No_Burst are one "not a burst" in everything reported, although the
model is trained with RFI as its own rejection class.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.coords import SpectrumAxes  # noqa: E402
from callisto_trainer.core.region_features import feature_count  # noqa: E402
from callisto_trainer.core.taxonomy import NO_BURST, RFI, TYPE_II, TYPE_III  # noqa: E402

SHAPE = (200, 3600)
AXES = SpectrumAxes(time_s=np.arange(SHAPE[1]) * 0.25, freq_mhz=np.linspace(80.0, 20.0, SHAPE[0]))


def _quiet(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)


def _with_burst(array: np.ndarray) -> np.ndarray:
    for row in range(10, 150):
        col = int(400 + (row - 10) * 0.08)
        array[row, col:col + 6] = 0.8
    return array


def _with_carrier(array: np.ndarray) -> np.ndarray:
    array[170:173, 100:3500] = 0.9       # a long thin line: a carrier signature
    return array


def _checkpoint(tmp_path: Path, classes: list[str]) -> Path:
    from callisto_trainer.core.models.model_factory import create_model

    config = load_config()
    config["data"]["classes"] = {name: index for index, name in enumerate(classes)}
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(classes),
         "use_metadata": False, "use_physics": True,
         "views": ["crop", "context"], "feature_set": "region_v2"}
    )
    config["inference"] = {"burst_threshold": 0.5}
    model = create_model("simple_cnn", in_channels=1, num_classes=len(classes), use_physics=True,
                         num_physics=feature_count("region_v2"), num_views=2)
    path = tmp_path / f"unified_{len(classes)}.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def _predictor(tmp_path, monkeypatch, classes, burst_row, background_row):
    """A predictor whose model calls tall regions ``burst_row``, others ``background_row``."""
    from callisto_trainer.core.inference import CascadePredictor

    predictor = CascadePredictor(load_config(), unified_checkpoint=_checkpoint(tmp_path, classes))
    original = predictor._unified_probabilities
    names = predictor.unified_class_names

    def fake(normalized, axes, boxes, rfi_channels=None, quiet=None, file_meta=None):
        _, encoded = original(
            normalized, axes, boxes, rfi_channels, quiet=quiet, file_meta=file_meta
        )
        rows = [burst_row if box.n_rows > 50 else background_row for box in boxes]
        return np.array([[row.get(name, 0.0) for name in names] for row in rows]), encoded

    monkeypatch.setattr(predictor, "_unified_probabilities", fake)
    return predictor


def _predict(predictor, array):
    from callisto_trainer.core.inference import FileResult

    return predictor.predict_normalized(array, AXES, FileResult("x.fit.gz", "x.fit.gz"))


CLASSES = [NO_BURST, RFI, TYPE_II, TYPE_III]
BURST = {NO_BURST: 0.05, RFI: 0.05, TYPE_III: 0.9}
BACKGROUND = {NO_BURST: 0.9, RFI: 0.05, TYPE_III: 0.05}


def test_a_burst_with_rfi_is_a_burst_and_lists_the_rfi(tmp_path, monkeypatch) -> None:
    predictor = _predictor(tmp_path, monkeypatch, CLASSES, BURST, BACKGROUND)
    result = _predict(predictor, _with_carrier(_with_burst(_quiet())))

    assert result.predicted_label == "Burst"
    assert [region.burst_type for region in result.regions] == [TYPE_III]
    assert result.rfi_regions and all(r.burst_type == RFI for r in result.rfi_regions)
    assert result.rfi_regions[0].rfi_kind == "carrier", "found by its signature"
    assert result.region_summary == "Type III  ·  RFI x1"


def test_only_rfi_is_no_burst_and_lists_the_rfi(tmp_path, monkeypatch) -> None:
    predictor = _predictor(tmp_path, monkeypatch, CLASSES, BURST, BACKGROUND)
    result = _predict(predictor, _with_carrier(_quiet()))

    assert result.predicted_label == "No_Burst"
    assert result.regions == []
    assert len(result.rfi_regions) == 1
    assert result.region_summary == "RFI x1"


def test_rfi_is_detected_without_an_rfi_class(tmp_path, monkeypatch) -> None:
    classes = [NO_BURST, TYPE_II, TYPE_III]
    predictor = _predictor(tmp_path, monkeypatch, classes, BURST, BACKGROUND)
    result = _predict(predictor, _with_carrier(_quiet()))
    assert result.predicted_label == "No_Burst" and len(result.rfi_regions) == 1


def test_the_models_own_rfi_output_also_counts(tmp_path, monkeypatch) -> None:
    """No signature, but the model puts more on RFI than on background."""
    rfi_heavy = {NO_BURST: 0.2, RFI: 0.75, TYPE_III: 0.05}
    predictor = _predictor(tmp_path, monkeypatch, CLASSES, rfi_heavy, rfi_heavy)
    result = _predict(predictor, _with_burst(_quiet()))
    assert result.predicted_label == "No_Burst"
    assert result.rfi_regions and result.rfi_regions[0].rfi_kind == "interference"


def test_reported_probabilities_merge_rfi_into_no_burst(tmp_path, monkeypatch) -> None:
    predictor = _predictor(tmp_path, monkeypatch, CLASSES, BURST, BACKGROUND)
    result = _predict(predictor, _with_carrier(_with_burst(_quiet())))
    for region in result.regions + result.rfi_regions:
        assert RFI not in region.type_probabilities
        assert sum(region.type_probabilities.values()) == pytest.approx(1.0)
    rfi = result.rfi_regions[0]
    assert rfi.type_probabilities[NO_BURST] == pytest.approx(0.95)


def test_rfi_never_suppresses_a_burst_region(tmp_path, monkeypatch) -> None:
    """A burst region with an interference signature stays a burst."""
    predictor = _predictor(tmp_path, monkeypatch, CLASSES, BURST, BURST)
    result = _predict(predictor, _with_carrier(_with_burst(_quiet())))
    assert result.predicted_label == "Burst"
    assert result.rfi_regions == [], "every region was called a burst"


def test_the_csv_lists_rfi_separately(tmp_path, monkeypatch) -> None:
    import csv
    import json

    from callisto_trainer.core.inference import write_csv

    predictor = _predictor(tmp_path, monkeypatch, CLASSES, BURST, BACKGROUND)
    result = _predict(predictor, _with_carrier(_with_burst(_quiet())))
    path = write_csv([result], tmp_path / "out.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["predicted_label"] == "Burst"
    assert [item["kind"] for item in json.loads(row["rfi_regions"])] == ["carrier"]


# -- reports ---------------------------------------------------------------------


def test_merge_rejections_folds_rfi_into_no_burst() -> None:
    from callisto_trainer.core.unified_metrics import merge_rejections

    labels, names = merge_rejections([0, 1, 2, 3, 1], CLASSES)
    assert names == [NO_BURST, TYPE_II, TYPE_III]
    assert labels.tolist() == [0, 0, 1, 2, 0]
    unchanged, same = merge_rejections([0, 1], [TYPE_II, TYPE_III])
    assert same == [TYPE_II, TYPE_III] and unchanged.tolist() == [0, 1]


def test_snapshot_counts_show_rfi_inside_no_burst() -> None:
    from callisto_trainer.store.export import describe_class_counts, shown_class_counts

    counts = {NO_BURST: 8413, RFI: 10256, TYPE_III: 2632}
    assert shown_class_counts(counts) == {NO_BURST: 18669, TYPE_III: 2632}
    assert describe_class_counts(counts) == "No_Burst: 18,669 (RFI 10,256)  Type III: 2,632"


# -- the Dataset tab -------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_deleting_more_than_2_gib_of_snapshots_reports_it(qapp, tmp_path: Path) -> None:
    """10,025,980,020 bytes overflowed a C++ int in the signal."""
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.dataset_tab import DatasetTab, _DeleteWorker
    from callisto_trainer.store.db import Database
    from callisto_trainer.store.repository import AnnotationRepository

    settings = AppSettings(project_root=tmp_path, database_path=tmp_path / "a.db",
                           display_cache_dir=tmp_path / "c", datasets_dir=tmp_path / "d",
                           outputs_dir=tmp_path / "o")
    tab = DatasetTab(AnnotationRepository(Database(tmp_path / "a.db")), settings)
    worker = _DeleteWorker(settings.datasets_dir, [])
    worker.finished_with.connect(tab._on_deleted)
    worker.finished_with.emit(10_025_980_020, [])
    qapp.processEvents()
    assert "10.03 GB" in tab.status.text()


def test_a_training_event_before_start_does_not_crash(qapp, tmp_path: Path) -> None:
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.train_tab import TrainTab

    settings = AppSettings(project_root=tmp_path, database_path=tmp_path / "a.db",
                           display_cache_dir=tmp_path / "c", datasets_dir=tmp_path / "d",
                           outputs_dir=tmp_path / "o")
    tab = TrainTab(settings)
    tab._on_progress({"event": "epoch", "epoch": 1, "total_epochs": 2, "train_loss": 0.5,
                      "val_loss": 0.4, "val_accuracy": 0.9, "val_macro_f1": 0.8,
                      "score": 0.8, "is_best": True, "monitor": "unified_score"})
    tab.shutdown()


def test_segments_on_the_same_channels_are_one_interference_source() -> None:
    """The lowest-channel calibration block comes out as ~14 segments per file."""
    from callisto_trainer.core.inference import RegionResult, group_interference

    def region(row0, row1, col0, col1, kind):
        return RegionResult(row0, row1, col0, col1, area=1, peak=1.0, rfi_kind=kind)

    band = [region(194, 200, start, start + 126, "periodic") for start in range(0, 3400, 255)]
    band[0].rfi_kind = "carrier"
    impulse = region(0, 200, 1000, 1002, "impulse")
    carrier = region(100, 103, 0, 3000, "carrier")
    sources = group_interference(band + [impulse, carrier], AXES)

    assert len(sources) == 3
    calibration = sources[0]
    assert calibration.kind == "periodic" and calibration.segments == len(band)
    assert (calibration.row0, calibration.row1) == (194, 200)
    assert calibration.col1 == band[-1].col1
    assert calibration.freq_lo_mhz < calibration.freq_hi_mhz
    assert {s.kind for s in sources[1:]} == {"impulse", "carrier"}, "a crossing impulse stays apart"
