"""Dataset export: tensor contract, manifest shape and split integrity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from callisto_trainer.core.config import load_config
from callisto_trainer.core.crops import CropConfig, normalize_full_spectrum, whole_file_box
from callisto_trainer.core.crops import crop_from_normalized
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.manifest import event_key_for_path
from callisto_trainer.core.preprocess import preprocess_array
from callisto_trainer.services.importer import import_files
from callisto_trainer.store.db import Database
from callisto_trainer.store.export import (
    MANIFEST_COLUMNS,
    export_binary_dataset,
    export_type_dataset,
    list_snapshots,
    read_snapshot_info,
)
from callisto_trainer.store.repository import (
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
)


@pytest.fixture
def config() -> dict:
    return load_config()


@pytest.fixture
def populated(tmp_path: Path, axes_files, config: dict):
    """A store with several labelled files, spanning all three burst types."""
    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(p) for p in axes_files])

    records = repo.files()
    assert len(records) >= 6, "need enough files to fill three splits"

    types = ["Type II", "Type III", "Other"]
    for index, record in enumerate(records):
        if index % 4 == 3:
            repo.set_verdict(record.id, VERDICT_NO_BURST)
            continue
        repo.set_verdict(record.id, VERDICT_BURST)
        n_freq, n_time = record.n_freq, record.n_time
        # Two boxes per burst file, one of each of two types.
        for offset in range(2):
            burst_type = types[(index + offset) % 3]
            repo.add_box(
                record.id,
                row0=10 + offset * 40,
                row1=min(60 + offset * 40, n_freq),
                col0=100 + offset * 300,
                col1=min(400 + offset * 300, n_time),
                burst_type=burst_type,
                physical={
                    "freq_lo_mhz": 20.0,
                    "freq_hi_mhz": 60.0,
                    "t_start_s": 10.0,
                    "t_end_s": 90.0,
                },
            )
    return repo, tmp_path


# -- type export -----------------------------------------------------------


def test_type_export_writes_one_sample_per_box(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert result.failed == 0
    assert result.written == repo.total_boxes()
    assert result.rows == result.written

    npz_files = list((result.directory / "npz").glob("*.npz"))
    assert len(npz_files) == result.written


def test_exported_type_tensor_contract(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    for path in list((result.directory / "npz").glob("*.npz"))[:5]:
        with np.load(path, allow_pickle=False) as loaded:
            spectrum = loaded["spectrum"]
            assert spectrum.shape == (1, 224, 224)
            assert spectrum.dtype == np.float32
            assert 0.0 <= float(spectrum.min()) <= float(spectrum.max()) <= 1.0
            assert 0 <= int(loaded["label_id"]) <= 2
            metadata = json.loads(str(loaded["metadata_json"]))
            assert metadata["burst_type"] in ("Type II", "Type III", "Other")
            assert len(metadata["pixel_box"]) == 4


def test_exported_crop_matches_recomputing_it(populated, config: dict) -> None:
    """The stored tensor must equal a fresh crop from the same box, exactly."""
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    crop_config = CropConfig.from_config(config)

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    spectrum, _ = read_fits_spectrum(row["file_path"])
    normalized = normalize_full_spectrum(spectrum, config)
    from callisto_trainer.core.crops import PixelBox

    expected = crop_from_normalized(
        normalized,
        PixelBox(int(row["row0"]), int(row["row1"]), int(row["col0"]), int(row["col1"])),
        crop_config,
    )
    with np.load(row["processed_path"], allow_pickle=False) as loaded:
        assert np.array_equal(loaded["spectrum"], expected)


def test_manifest_has_the_columns_the_trainer_reads(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MANIFEST_COLUMNS
        rows = list(reader)

    for row in rows:
        assert Path(row["processed_path"]).exists()
        assert row["split"] in ("train", "val", "test")
        assert row["label"] in ("Type II", "Type III", "Other")
        assert row["box_id"]


def test_no_source_file_spans_two_splits(populated, config: dict) -> None:
    """Two crops from one spectrum in different splits would leak the answer."""
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    splits_by_file: dict[str, set[str]] = {}
    for row in rows:
        splits_by_file.setdefault(row["file_path"], set()).add(row["split"])

    offenders = {path: splits for path, splits in splits_by_file.items() if len(splits) > 1}
    assert not offenders, f"source files split across train/test: {offenders}"


def test_no_solar_event_spans_two_splits(populated, config: dict) -> None:
    """Sibling recordings of one event, from different stations, must stay together."""
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    splits_by_event: dict[tuple, set[str]] = {}
    for row in rows:
        splits_by_event.setdefault(event_key_for_path(row["file_path"]), set()).add(row["split"])

    assert all(len(splits) == 1 for splits in splits_by_event.values())
    assert result.event_leakage == 0


def test_only_confirmed_boxes_on_burst_files_are_exported(populated, config: dict) -> None:
    repo, tmp_path = populated
    burst_file = repo.files_with_verdict([VERDICT_BURST])[0]
    repo.add_box(burst_file.id, 0, 20, 0, 100, "Type II", source="assisted", confirmed=False)

    quiet = repo.files_with_verdict([VERDICT_NO_BURST])
    if quiet:
        repo.add_box(quiet[0].id, 0, 20, 0, 100, "Other")

    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    assert result.written == repo.total_boxes(confirmed_only=True) - (1 if quiet else 0)


# -- binary export ---------------------------------------------------------


def test_binary_export_uses_file_verdicts(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_binary_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert result.failed == 0
    assert set(result.class_counts) <= {"Burst", "No_Burst"}
    assert result.written == len(repo.files_with_verdict([VERDICT_BURST, VERDICT_NO_BURST]))


def test_binary_tensor_is_identical_to_upstream_preprocessing(populated, config: dict) -> None:
    """The binary track must stay byte-compatible with the original pipeline."""
    repo, tmp_path = populated
    result = export_binary_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    spectrum, _ = read_fits_spectrum(row["file_path"])
    expected = preprocess_array(spectrum, config)
    with np.load(row["processed_path"], allow_pickle=False) as loaded:
        assert np.array_equal(loaded["spectrum"], expected)


# -- snapshot metadata -----------------------------------------------------


def test_snapshot_records_provenance(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    info = read_snapshot_info(result.directory)

    assert info["kind"] == "types"
    assert info["classes"] == {"Type II": 0, "Type III": 1, "Other": 2}
    assert info["samples"] == result.written
    assert info["preprocessing"]["db_vmin"] == -1.0
    assert info["preprocessing"]["db_vmax"] == 8.0
    assert info["crops"]["context_margin"] == pytest.approx(0.0)
    assert info["event_leakage"] == 0
    assert info["tool_version"]


def test_generated_training_config_is_runnable(populated, config: dict) -> None:
    import yaml

    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    config_path = result.directory / "config.yaml"
    assert config_path.exists()
    with config_path.open(encoding="utf-8") as handle:
        generated = yaml.safe_load(handle)

    assert generated["data"]["classes"] == {"Type II": 0, "Type III": 1, "Other": 2}
    assert generated["model"]["num_classes"] == 3
    assert generated["model"]["use_metadata"] is False, "type model is image-only"
    assert generated["training"]["monitor"] == "macro_f1"
    assert Path(generated["paths"]["manifest_path"]) == result.manifest_path

    # And it must load through the vendored loader without losing the class set.
    loaded = load_config(config_path)
    assert loaded["data"]["classes"] == {"Type II": 0, "Type III": 1, "Other": 2}


def test_binary_config_keeps_metadata_branch_and_threshold(populated, config: dict) -> None:
    import yaml

    repo, tmp_path = populated
    result = export_binary_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    with (result.directory / "config.yaml").open(encoding="utf-8") as handle:
        generated = yaml.safe_load(handle)

    assert generated["model"]["use_metadata"] is True
    assert generated["model"]["num_classes"] == 1
    assert generated["training"]["auto_threshold"] is True
    assert generated["training"]["class_balance"]["strategy"] == "auto_pos_weight"


def test_snapshots_are_listed_newest_first(populated, config: dict) -> None:
    repo, tmp_path = populated
    first = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    import time

    time.sleep(1.1)  # run ids have second resolution
    second = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    snapshots = list_snapshots(tmp_path / "datasets", "types")
    assert snapshots[0] == second.directory
    assert first.directory in snapshots


def test_earlier_snapshot_is_untouched_by_a_later_one(populated, config: dict) -> None:
    """Snapshots are immutable so a checkpoint always traces to fixed data."""
    repo, tmp_path = populated
    first = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    original = first.manifest_path.read_bytes()

    burst_file = repo.files_with_verdict([VERDICT_BURST])[0]
    repo.add_box(burst_file.id, 5, 30, 5, 200, "Other")
    import time

    time.sleep(1.1)
    export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert first.manifest_path.read_bytes() == original


# -- guardrails ------------------------------------------------------------


def test_empty_export_reports_a_blocking_problem(tmp_path: Path, config: dict) -> None:
    repo = AnnotationRepository(Database(tmp_path / "empty.db"))
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert result.written == 0
    problems = result.blocking_problems()
    assert problems and "empty" in problems[0].lower()


def test_thin_class_is_reported_before_training(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    problems = result.blocking_problems(minimum_per_split=1000)
    assert problems, "an impossible minimum must be reported, not silently accepted"
    assert any("split" in problem for problem in problems)


def test_unreadable_source_is_counted_not_fatal(populated, config: dict, tmp_path: Path) -> None:
    repo, _ = populated
    broken = tmp_path / "broken.fit.gz"
    broken.write_bytes(b"definitely not FITS")
    file_id = repo.add_file(broken, {"station": "X", "date": "2023-01-01", "n_freq": 1, "n_time": 1})
    repo.set_verdict(file_id, VERDICT_BURST)
    repo.add_box(file_id, 0, 10, 0, 10, "Other")

    result = export_type_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert result.failed >= 1
    assert result.written > 0, "one bad file must not abort the whole export"
    assert result.errors
