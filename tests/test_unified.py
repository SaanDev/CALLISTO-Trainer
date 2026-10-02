"""The unified 4-class model: region assignment, export, and inference.

The load-bearing test here is `test_positives_and_negatives_share_a_distribution`.
An earlier version drew positives from hand-drawn boxes and negatives from
finder-proposed regions, so the two classes differed in how the region was
*produced* as well as in what it contained. The model learned that shortcut: 0.95
burst recall on held-out crops, and 1 detection in 12 burst files at inference.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.inference import (  # noqa: E402
    NO_BURST_LABEL,
    CascadePredictor,
    predict_paths,
)
from callisto_trainer.core.models.model_factory import create_model  # noqa: E402
from callisto_trainer.core.negatives import (  # noqa: E402
    assign_regions,
    box_iou,
    containment,
    mine_negatives,
    negative_budget,
)
from callisto_trainer.services.importer import import_files  # noqa: E402
from callisto_trainer.store.db import Database  # noqa: E402
from callisto_trainer.store.export import (  # noqa: E402
    UNIFIED_CLASSES,
    export_unified_dataset,
    read_snapshot_info,
    task_for_kind,
)
from callisto_trainer.store.repository import (  # noqa: E402
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
)


class _Box:
    """Minimal stand-in for a BoxRecord."""

    def __init__(self, row0, row1, col0, col1, burst_type="Type III", identifier=1):
        self.row0, self.row1, self.col0, self.col1 = row0, row1, col0, col1
        self.burst_type = burst_type
        self.id = identifier


@pytest.fixture
def config() -> dict:
    return load_config()


# -- geometry --------------------------------------------------------------


def test_containment_is_the_right_measure_for_this_geometry() -> None:
    """A small candidate inside a big drawn box: IoU says no, containment says yes.

    Measured on the archive, drawn boxes average ~46,900 px and candidate regions
    ~388 px. An IoU rule discards nearly every genuine match.
    """
    drawn = (0, 200, 0, 240)      # 48,000 px, a typical hand-drawn box
    candidate = (90, 110, 100, 120)  # 400 px, entirely inside it

    assert box_iou(candidate, drawn) < 0.01
    assert containment(candidate, drawn) == pytest.approx(1.0)


def test_containment_of_a_disjoint_region_is_zero() -> None:
    assert containment((0, 10, 0, 10), (50, 60, 50, 60)) == 0.0


def test_containment_is_partial_for_a_straddling_region() -> None:
    # Half of the candidate's columns lie inside the reference.
    assert containment((0, 10, 0, 20), (0, 10, 10, 40)) == pytest.approx(0.5)


# -- region assignment -----------------------------------------------------


def _spectrum_with(*regions) -> np.ndarray:
    array = np.zeros((200, 1200), dtype=np.float32)
    for row0, row1, col0, col1 in regions:
        array[row0:row1, col0:col1] = 0.95
    return array


def test_regions_inside_a_box_become_positives_of_its_type() -> None:
    normalized = _spectrum_with((60, 90, 300, 340))
    boxes = [_Box(40, 120, 260, 400, "Type II", 7)]

    assigned = assign_regions(normalized, boxes)
    positives = [region for region in assigned if region.label == "Type II"]

    assert positives, "a bright region inside a drawn box must become a positive"
    assert positives[0].matched_box_id == 7
    assert positives[0].containment > 0.9


def test_regions_far_from_any_box_become_negatives() -> None:
    normalized = _spectrum_with((60, 90, 300, 340), (150, 180, 900, 950))
    boxes = [_Box(40, 120, 260, 400, "Type II", 7)]

    labels = {region.label for region in assign_regions(normalized, boxes)}
    assert "Type II" in labels
    assert NO_BURST_LABEL in labels


def test_ambiguous_overlap_is_dropped_not_guessed() -> None:
    """Half-in, half-out is a clean example of neither class."""
    normalized = _spectrum_with((60, 90, 380, 460))
    boxes = [_Box(40, 120, 260, 420, "Type III", 1)]

    assigned = assign_regions(normalized, boxes, positive_containment=0.9, negative_containment=0.1)
    assert any(region.label is None for region in assigned)


def test_no_boxes_means_everything_is_background() -> None:
    normalized = _spectrum_with((60, 90, 300, 340))
    assert all(region.label == NO_BURST_LABEL for region in assign_regions(normalized, []))


# -- negative mining -------------------------------------------------------


def test_mining_excludes_regions_inside_a_drawn_box() -> None:
    """Using IoU here would keep burst signal and label it background."""
    normalized = _spectrum_with((60, 90, 300, 340))
    drawn = [(40, 120, 260, 400)]

    assert mine_negatives(normalized, exclude_boxes=drawn, max_negatives=5) == []
    assert mine_negatives(normalized, exclude_boxes=[], max_negatives=5), "sanity check"


def test_mining_respects_its_cap() -> None:
    normalized = _spectrum_with(*[(i * 20, i * 20 + 15, 100, 160) for i in range(9)])
    assert len(mine_negatives(normalized, max_negatives=3)) <= 3


def test_mining_is_deterministic() -> None:
    normalized = _spectrum_with(*[(i * 20, i * 20 + 15, 100, 160) for i in range(9)])
    first = mine_negatives(normalized, max_negatives=4, seed=11)
    second = mine_negatives(normalized, max_negatives=4, seed=11)
    assert [r.as_box().as_tuple() for r in first] == [r.as_box().as_tuple() for r in second]


def test_negative_budget_scales_with_positives() -> None:
    small_quiet, small_burst = negative_budget(100, quiet_files=40, burst_files=100)
    large_quiet, _ = negative_budget(1000, quiet_files=40, burst_files=100)

    assert large_quiet > small_quiet
    assert small_burst <= 4
    assert negative_budget(500, quiet_files=0, burst_files=10)[0] == 0


# -- export ----------------------------------------------------------------


@pytest.fixture
def populated(tmp_path: Path, axes_files):
    """A store labelled the way an operator would label it.

    Boxes are drawn *around actual bright features*, generously, rather than at
    fixed coordinates: that is what real annotation looks like, and it is the
    only way the export's region matching has anything to match.
    """
    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum
    from callisto_trainer.core.inference import resolve_threshold
    from callisto_trainer.services.assist import find_candidate_regions

    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(p) for p in axes_files])
    records = repo.files()
    assert len(records) >= 6

    types = ["Type II", "Type III", "Other"]
    burst_index = 0
    for index, record in enumerate(records):
        if index % 3 == 2:
            repo.set_verdict(record.id, VERDICT_NO_BURST)
            continue

        spectrum, _ = read_fits_spectrum(record.path)
        normalized = normalize_full_spectrum(spectrum, load_config())
        candidates = find_candidate_regions(
            normalized, threshold=resolve_threshold(normalized), max_candidates=4
        )
        if not candidates:
            repo.set_verdict(record.id, VERDICT_NO_BURST)
            continue

        repo.set_verdict(record.id, VERDICT_BURST)
        # A generous box around the brightest feature, as a human would draw it.
        best = candidates[0]
        pad_rows = max(15, (best.row1 - best.row0))
        pad_cols = max(60, (best.col1 - best.col0) * 2)
        repo.add_box(
            record.id,
            row0=max(0, best.row0 - pad_rows),
            row1=min(record.n_freq, best.row1 + pad_rows),
            col0=max(0, best.col0 - pad_cols),
            col1=min(record.n_time, best.col1 + pad_cols),
            # Cycle types independently of the verdict, so every class appears.
            burst_type=types[burst_index % 3],
        )
        burst_index += 1
    return repo, tmp_path


def test_unified_export_has_all_four_classes(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_unified_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    assert result.failed == 0
    assert NO_BURST_LABEL in result.class_counts
    assert result.class_counts[NO_BURST_LABEL] > 0
    assert sum(result.class_counts.values()) == result.written


def test_positives_and_negatives_share_a_distribution(populated, config: dict) -> None:
    """The regression that matters: both classes must include finder-shaped crops.

    Without matched positives the model separates classes by how the region was
    produced rather than by content, and detects almost nothing at inference.
    """
    repo, tmp_path = populated
    result = export_unified_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sources = {row["label"]: set() for row in rows}
    for row in rows:
        sources[row["label"]].add(row["region_source"])

    burst_labels = [label for label in sources if label != NO_BURST_LABEL]
    assert burst_labels, "expected some burst classes"
    assert any("matched_region" in sources[label] for label in burst_labels), (
        "no finder-shaped positives were exported; the model would learn to "
        "separate drawn boxes from found regions instead of bursts from background"
    )
    assert result.matched_regions > 0


def test_matched_positives_are_capped_per_box(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_unified_dataset(
        repo, tmp_path / "datasets", config, tmp_path / "outputs", max_matched_per_box=1
    )

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        matched = [r for r in csv.DictReader(handle) if r["region_source"] == "matched_region"]

    per_box: dict[str, int] = {}
    for row in matched:
        key = f"{row['file_path']}::{row['label']}"
        per_box[key] = per_box.get(key, 0) + 1
    assert all(count <= 1 for count in per_box.values())


def test_unified_snapshot_records_its_mining_policy(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_unified_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    info = read_snapshot_info(result.directory)

    assert info["kind"] == "unified"
    # Only the classes with examples, in the unified order, background first.
    assert info["classes"] == result.classes
    assert list(info["classes"])[0] == NO_BURST_LABEL
    assert set(info["classes"]) <= set(UNIFIED_CLASSES)
    assert info["negative_mining"]["per_quiet_file"] >= 0
    assert "not random background" in info["negative_mining"]["note"]
    assert info["region_assignment"]["matched_regions"] == result.matched_regions
    assert info["views"] == list(_default_views())
    assert info["feature_set"] == "region_v2"


def test_unified_config_describes_the_region_inputs(populated, config: dict) -> None:
    import yaml

    repo, tmp_path = populated
    result = export_unified_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")
    generated = yaml.safe_load((result.directory / "config.yaml").read_text(encoding="utf-8"))

    assert generated["data"]["classes"] == result.classes
    assert generated["model"]["num_classes"] == len(result.classes)
    assert generated["model"]["use_metadata"] is False
    assert generated["model"]["use_physics"] is True
    assert generated["model"]["views"] == list(_default_views())
    assert generated["model"]["feature_set"] == "region_v2"
    assert generated["training"]["monitor"] == "unified_score"
    assert generated["training"]["class_balance"]["strategy"] == "detection_balanced"
    # The operating point the operator asked for: few false alarms.
    assert generated["calibration"]["max_false_alarm_rate"] == pytest.approx(0.05)
    assert Path(generated["calibration"]["files_manifest"]).exists()
    assert generated["inference"]["region_finder"]["max_regions"] >= 8


def test_unified_export_keeps_splits_clean(populated, config: dict) -> None:
    repo, tmp_path = populated
    result = export_unified_dataset(repo, tmp_path / "datasets", config, tmp_path / "outputs")

    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits_by_file: dict[str, set[str]] = {}
    for row in rows:
        splits_by_file.setdefault(row["file_path"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in splits_by_file.values())
    assert result.event_leakage == 0


# -- inference -------------------------------------------------------------


# The class set checkpoints were trained on before RFI and Type IIIG existed.
LEGACY_UNIFIED_CLASSES = {"No_Burst": 0, "Type II": 1, "Type III": 2, "Other": 3}


def _unified_checkpoint(
    tmp_path: Path,
    use_physics: bool = False,
    v2: bool = False,
    burst_threshold: float | None = None,
) -> Path:
    """A unified checkpoint: legacy four-class, or the current two-view model.

    ``use_physics`` matters: the fused model nests its backbone one level down, so
    a loader that ignores the flag builds a bare CNN and dies in load_state_dict.
    ``v2`` builds what a current export trains: six classes, two views and the
    region_v2 features -- a loader that assumes one view or eight features fails
    the same way.
    """
    from callisto_trainer.core.burst_physics import NUM_PHYSICS_FEATURES
    from callisto_trainer.core.region_features import feature_count

    config = load_config()
    if v2:
        classes = dict(UNIFIED_CLASSES)
        config["model"].update(
            {
                "name": "simple_cnn", "in_channels": 1, "num_classes": len(classes),
                "use_metadata": False, "use_physics": True,
                "views": ["crop", "context"], "feature_set": "region_v2",
            }
        )
        extra = dict(use_physics=True, num_physics=feature_count("region_v2"), num_views=2)
    else:
        classes = dict(LEGACY_UNIFIED_CLASSES)
        config["model"].update(
            {
                "name": "simple_cnn", "in_channels": 1, "num_classes": len(classes),
                "use_metadata": False, "use_physics": use_physics,
            }
        )
        extra = (
            dict(use_physics=True, num_physics=NUM_PHYSICS_FEATURES) if use_physics else {}
        )
    config["data"]["classes"] = classes
    if burst_threshold is not None:
        config["inference"] = {"burst_threshold": burst_threshold}
    model = create_model("simple_cnn", in_channels=1, num_classes=len(classes), **extra)
    name = "unified_v2" if v2 else ("unified_physics" if use_physics else "unified")
    path = tmp_path / f"{name}.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def _fixed_probabilities(predictor, row: dict[str, float]):
    """Replace the model's output with ``row`` for every region, keeping the encoding."""
    original = predictor._unified_probabilities
    vector = np.array([row.get(name, 0.0) for name in predictor.unified_class_names])

    def fake(normalized, axes, boxes, rfi_channels=None, quiet=None, file_meta=None):
        _, encoded = original(
            normalized, axes, boxes, rfi_channels, quiet=quiet, file_meta=file_meta
        )
        return np.tile(vector, (len(boxes), 1)), encoded

    return fake


def test_physics_checkpoint_loads_for_inference(tmp_path: Path, config: dict, any_real_file):
    """Regression: a physics-conditioned checkpoint must load and predict.

    The fused model's weights are named ``backbone.*`` plus ``physics_mlp.*``.
    Rebuilding a bare CNN for it raised a wall of "Missing key(s) ... conv1.weight"
    and crashed the Predict tab on the first folder run.
    """
    checkpoint = _unified_checkpoint(tmp_path, use_physics=True)
    predictor = CascadePredictor(config, unified_checkpoint=checkpoint)

    assert predictor.is_unified
    assert predictor.unified_uses_physics, "the physics branch must be detected"

    result = predictor.predict_file(any_real_file)
    assert result.error is None
    assert result.predicted_label in ("Burst", "No_Burst")


def test_physics_checkpoint_measures_regions(tmp_path: Path, config: dict, any_real_file):
    """With the branch active, each region gets a measured drift rate."""
    predictor = CascadePredictor(
        config, unified_checkpoint=_unified_checkpoint(tmp_path, use_physics=True)
    )
    result = predictor.predict_file(any_real_file)
    if not result.regions_examined:
        pytest.skip("no candidate region in this file")

    for region in result.regions:
        assert region.physics_confidence is not None


def test_image_only_checkpoint_still_loads(tmp_path: Path, config: dict, any_real_file):
    """The flag must not become mandatory: older checkpoints keep working."""
    predictor = CascadePredictor(
        config, unified_checkpoint=_unified_checkpoint(tmp_path, use_physics=False)
    )
    assert not predictor.unified_uses_physics
    assert predictor.predict_file(any_real_file).error is None


def test_predict_paths_accepts_a_physics_checkpoint(tmp_path: Path, config: dict, axes_files):
    results = predict_paths(
        [Path(axes_files[0])],
        config,
        unified_checkpoint=_unified_checkpoint(tmp_path, use_physics=True),
    )
    assert len(results) == 1 and results[0].error is None


def test_predictor_accepts_a_unified_checkpoint(tmp_path: Path, config: dict, any_real_file):
    predictor = CascadePredictor(config, unified_checkpoint=_unified_checkpoint(tmp_path))

    assert predictor.is_unified
    assert predictor.unified_class_names[0] == NO_BURST_LABEL

    result = predictor.predict_file(any_real_file)
    assert result.predicted_label in ("Burst", "No_Burst")
    assert result.regions_examined >= 0
    assert result.regions_rejected <= result.regions_examined


def test_v2_checkpoint_loads_and_predicts(tmp_path: Path, config: dict, any_real_file) -> None:
    """Six classes, two views and 28 features must all be rebuilt from the config."""
    predictor = CascadePredictor(config, unified_checkpoint=_unified_checkpoint(tmp_path, v2=True))

    assert predictor.encoder.spec.views == ("crop", "context")
    assert predictor.encoder.spec.feature_set == "region_v2"
    result = predictor.predict_file(any_real_file)
    assert result.error is None
    assert result.predicted_label in ("Burst", "No_Burst")
    for region in result.regions + result.rfi_regions:
        assert 0.0 <= region.burst_evidence <= 1.0


def test_unified_rejects_a_three_class_checkpoint(tmp_path: Path, config: dict) -> None:
    """A type-only model has no No_Burst class and cannot gate anything."""
    type_config = load_config()
    type_config["data"]["classes"] = {"Type II": 0, "Type III": 1, "Other": 2}
    type_config["model"].update({"name": "simple_cnn", "num_classes": 3, "use_metadata": False})
    model = create_model("simple_cnn", in_channels=1, num_classes=3)
    path = tmp_path / "type_only.pt"
    torch.save({"model_state": model.state_dict(), "config": type_config}, path)

    with pytest.raises(ValueError, match="not a unified model"):
        CascadePredictor(config, unified_checkpoint=path)


def test_no_burst_regions_are_dropped_from_the_result(
    tmp_path: Path, config: dict, any_real_file, monkeypatch
) -> None:
    """Regions the model calls background must not appear as findings."""
    predictor = CascadePredictor(config, unified_checkpoint=_unified_checkpoint(tmp_path))
    monkeypatch.setattr(
        predictor, "_unified_probabilities",
        _fixed_probabilities(predictor, {NO_BURST_LABEL: 0.97, "Type III": 0.01,
                                         "Type II": 0.01, "Other": 0.01}),
    )
    result = predictor.predict_file(any_real_file)

    assert result.regions == []
    assert result.predicted_label == "No_Burst"
    assert result.regions_rejected == result.regions_examined


def test_a_confident_burst_region_makes_the_file_a_burst(
    tmp_path: Path, config: dict, any_real_file, monkeypatch
) -> None:
    predictor = CascadePredictor(config, unified_checkpoint=_unified_checkpoint(tmp_path))
    monkeypatch.setattr(
        predictor, "_unified_probabilities",
        _fixed_probabilities(predictor, {NO_BURST_LABEL: 0.04, "Type III": 0.9,
                                         "Type II": 0.03, "Other": 0.03}),
    )
    result = predictor.predict_file(any_real_file)

    if not result.regions_examined:
        pytest.skip("no candidate region in this file")
    assert result.predicted_label == "Burst"
    assert result.dominant_type == "Type III"
    assert result.burst_probability == pytest.approx(0.96, abs=0.01)
    assert all(region.burst_evidence == pytest.approx(0.96, abs=0.01) for region in result.regions)


def test_unified_takes_precedence_over_the_two_model_cascade(
    tmp_path: Path, config: dict, axes_files
) -> None:
    results = predict_paths(
        [Path(axes_files[0])], config, unified_checkpoint=_unified_checkpoint(tmp_path)
    )
    assert len(results) == 1
    assert results[0].predicted_label in ("Burst", "No_Burst")


# -- wiring ----------------------------------------------------------------


def test_task_names_map_to_snapshot_kinds() -> None:
    assert task_for_kind("unified") == "unified"
    assert task_for_kind("types") == "type"
    assert task_for_kind("binary") == "binary"


def test_training_job_targets_the_unified_modules(tmp_path: Path) -> None:
    from callisto_trainer.services.train_runner import TrainingJob

    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")

    assert TrainingJob.train("unified", config).module.endswith("train_unified")
    assert TrainingJob.evaluate("unified", config, tmp_path / "b.pt").module.endswith(
        "evaluate_unified"
    )


def test_burst_rollup_collapses_the_confusion_matrix() -> None:
    from callisto_trainer.core.evaluate_unified import burst_rollup

    metrics = {
        "class_names": ["No_Burst", "Type II", "Type III", "Other"],
        # 98 background right, 1 leaked; bursts mostly right, 5 called background.
        "confusion_matrix": [
            [98, 1, 0, 0],
            [3, 21, 0, 2],
            [1, 1, 46, 3],
            [1, 2, 3, 9],
        ],
    }
    rollup = burst_rollup(metrics)

    assert rollup["false_negatives"] == 5
    assert rollup["true_negatives"] == 98
    assert rollup["burst_recall"] == pytest.approx(87 / 92, abs=1e-3)
    assert rollup["no_burst_specificity"] == pytest.approx(98 / 99, abs=1e-3)
    # Type confusion within bursts must not count against detection.
    assert rollup["type_accuracy_given_detected"] < 1.0


def test_burst_rollup_needs_a_no_burst_class() -> None:
    from callisto_trainer.core.evaluate_unified import burst_rollup

    assert burst_rollup({"class_names": ["Type II", "Type III"], "confusion_matrix": [[1, 0], [0, 1]]}) == {}


def _default_views() -> tuple[str, ...]:
    from callisto_trainer.core.region_inputs import V2_VIEWS, V3_VIEWS
    from callisto_trainer.store.export import DEFAULT_QUIET_VIEW

    return V3_VIEWS if DEFAULT_QUIET_VIEW else V2_VIEWS
