"""The v2 pipeline: Type IIIG, Type IV and automatic RFI labels, interference
features, context views, file-level calibration, and the labelling rules that go
with them.

The motivation, in one line: crop-level metrics reported 98.8% of background
rejected while real use flagged far too many quiet files. Most tests here pin one
piece of the fix to observable behaviour -- a carrier must *look* like a carrier
to the features, a calibrated threshold must actually hold its false-alarm
budget, interference must be named RFI without anyone drawing it.
"""

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

from callisto_trainer.core.burst_physics import (  # noqa: E402
    BurstPhysics,
    count_bursts,
    group_hint,
    measure_burst,
)
from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.coords import SpectrumAxes  # noqa: E402
from callisto_trainer.core.crops import (  # noqa: E402
    CropConfig,
    PixelBox,
    crop_context,
    crop_from_normalized,
    region_views,
)
from callisto_trainer.core.region_features import (  # noqa: E402
    REGION_FEATURES,
    _periodicity,
    feature_count,
    feature_vector,
    file_context,
    measure_region,
)
from callisto_trainer.core.taxonomy import (  # noqa: E402
    BOX_TYPES,
    NO_BURST,
    RFI,
    TYPE_III,
    TYPE_IIIG,
    TYPE_IV,
    fold_rare_subclasses,
    ordered_classes,
)

# -- synthetic spectra -----------------------------------------------------

SHAPE = (200, 3600)


def _quiet(seed: int = 0, shape=SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.12, 0.03, shape), 0.0, 1.0).astype(np.float32)


def _axes(shape=SHAPE, cadence: float = 0.25) -> SpectrumAxes:
    return SpectrumAxes(
        time_s=np.arange(shape[1]) * cadence, freq_mhz=np.linspace(80.0, 20.0, shape[0])
    )


def _type_iii(array: np.ndarray, start: int, width: int = 6, rows=(10, 150), slope=0.08) -> None:
    """A drifting lane: later at lower frequency, several samples wide."""
    for row in range(*rows):
        col = int(start + (row - rows[0]) * slope)
        array[row, col:col + width] = np.maximum(array[row, col:col + width], 0.8)


def _features(array: np.ndarray, box: tuple[int, int, int, int], rfi=None) -> dict:
    return measure_region(file_context(array, _axes(array.shape), rfi), *box)


# -- taxonomy --------------------------------------------------------------


def test_box_types_follow_the_key_layout() -> None:
    """Keys 1-5: Type II, Type III, Type IIIG, Type IV, Other. RFI is never drawn."""
    assert BOX_TYPES == ("Type II", "Type III", "Type IIIG", "Type IV", "Other")
    assert RFI not in BOX_TYPES


def test_rare_classes_fold_into_their_fallback() -> None:
    mapping, folded = fold_rare_subclasses(
        {TYPE_IIIG: 3, TYPE_IV: 40, TYPE_III: 400, RFI: 50}, minimum=20
    )
    assert mapping[TYPE_IIIG] == TYPE_III, "a handful of groups trains as Type III"
    assert mapping[TYPE_IV] == TYPE_IV, "enough Type IV boxes keeps Type IV its own class"
    assert mapping[RFI] == RFI
    assert folded == {TYPE_IIIG: TYPE_III}

    mapping, folded = fold_rare_subclasses({TYPE_IV: 5, TYPE_IIIG: 30}, minimum=20)
    assert mapping[TYPE_IV] == "Other", "Type IV was labelled Other before it had a class"
    assert RFI not in folded, "RFI is only folded when its (automatic) count is given"


def test_classes_keep_the_unified_order_with_background_first() -> None:
    assert ordered_classes({"Other", NO_BURST, RFI}) == {NO_BURST: 0, RFI: 1, "Other": 2}


# -- burst counting (Type IIIG) --------------------------------------------


def test_a_group_of_bursts_is_counted_as_a_group() -> None:
    group = _quiet()
    for start in (400, 440, 490, 530):
        _type_iii(group, start)
    single = _quiet()
    _type_iii(single, 400)

    assert count_bursts(group[:, 380:600], 0.35, 0.25) == 4
    assert count_bursts(single[:, 380:600], 0.35, 0.25) == 1


def test_single_sample_spikes_are_not_bursts() -> None:
    spikes = _quiet()
    for col in range(420, 580, 20):
        spikes[:, col] = 0.9
    assert count_bursts(spikes[:, 400:600], 0.35, 0.25) == 0


def test_a_short_gap_does_not_split_one_burst_in_two() -> None:
    lane = _quiet()
    lane[20:150, 400:412] = 0.8
    lane[20:150, 405] = 0.1  # one dark sample inside the lane
    assert count_bursts(lane[:, 380:450], 0.35, 0.25) == 1


def test_measure_burst_records_the_count() -> None:
    group = _quiet()
    for start in (400, 440, 490, 530):
        _type_iii(group, start)
    physics = measure_burst(group, _axes(), 5, 160, 390, 570)
    assert physics.burst_count == 4


def test_group_hint_suggests_iiig_and_questions_a_lonely_one() -> None:
    assert "Type IIIG" in group_hint(BurstPhysics(burst_count=4), TYPE_III)
    assert "Only 1" in group_hint(BurstPhysics(burst_count=1), TYPE_IIIG)
    assert group_hint(BurstPhysics(burst_count=4), "Type II") is None
    assert group_hint(BurstPhysics(burst_count=0), TYPE_IIIG) is None, (
        "nothing measurable is not evidence against the label"
    )


# -- interference features ---------------------------------------------------


def test_a_carrier_is_persistent_outside_its_region() -> None:
    carrier = _quiet()
    carrier[100:102, :] = 0.9
    burst = _quiet()
    _type_iii(burst, 400)

    assert _features(carrier, (95, 107, 1000, 1400))["outside_persistence"] > 0.9
    assert _features(burst, (5, 160, 390, 430))["outside_persistence"] < 0.05


def test_an_impulse_lights_the_band_at_once_and_a_burst_drifts() -> None:
    spike = _quiet()
    spike[:, 1500] = 0.9
    burst = _quiet()
    _type_iii(burst, 400)

    impulse = _features(spike, (0, 200, 1490, 1512))
    drifting = _features(burst, (5, 160, 390, 430))
    assert impulse["simultaneity"] > 0.9 and drifting["simultaneity"] < 0.5
    assert impulse["log_onset_spread"] < 0.05 < drifting["log_onset_spread"]


def test_periodic_interference_scores_high_and_noise_does_not() -> None:
    periodic = _quiet()
    for col in range(200, 3400, 40):
        periodic[:, col] = 0.9
    assert _features(periodic, (0, 200, 1400, 1700))["periodicity"] > 0.8

    rng = np.random.default_rng(1)
    noise = [_periodicity(rng.normal(0, 1, 2000), 0.25)[0] for _ in range(10)]
    assert max(noise) < 0.15, "white noise must not look periodic"


def test_a_gain_step_is_a_flat_full_band_plateau() -> None:
    step = _quiet()
    step[:, 1000:2200] = np.clip(step[:, 1000:2200] + 0.35, 0, 1)
    values = _features(step, (0, 200, 1000, 2200))
    assert values["band_fraction"] > 0.9
    assert values["fill_fraction"] > 0.9
    assert values["peakiness"] < 0.1


def test_a_sweep_is_thinner_than_a_burst() -> None:
    sweep = _quiet()
    for row in range(SHAPE[0]):
        sweep[row, 500 + row * 2] = 0.9
    burst = _quiet()
    _type_iii(burst, 400)
    assert (
        _features(sweep, (0, 200, 495, 905))["log_thickness_time"]
        < _features(burst, (5, 160, 390, 430))["log_thickness_time"]
    )


def test_the_station_rfi_table_is_used() -> None:
    carrier = _quiet()
    carrier[100:102, :] = 0.9
    flagged = float(_axes().freq_mhz[100])
    assert _features(carrier, (95, 107, 1000, 1400), rfi=[flagged])["rfi_flag_fraction"] > 0.4
    assert _features(carrier, (95, 107, 1000, 1400))["rfi_flag_fraction"] == 0.0


def test_a_faint_region_is_flagged_not_zeroed() -> None:
    faint = _quiet()
    faint[50:60, 500:540] = 0.2  # below the file's bright level
    values = _features(faint, (45, 65, 490, 550))
    assert values["faint"] == 1.0
    assert values["log_rows"] > 0 and values["log_cols"] > 0


def test_feature_vector_is_fixed_length_and_finite() -> None:
    burst = _quiet()
    _type_iii(burst, 400)
    physics = measure_burst(burst, _axes(), 5, 160, 390, 430)
    region = _features(burst, (5, 160, 390, 430))
    vector = feature_vector("region_v2", physics, region)

    assert vector.shape == (feature_count("region_v2"),) == (8 + len(REGION_FEATURES),)
    assert np.isfinite(vector).all()
    assert np.abs(vector).max() < 10, "features must stay order-1 for the linear layer"


# -- context view ------------------------------------------------------------


def test_the_first_view_is_exactly_the_crop() -> None:
    array = _quiet()
    _type_iii(array, 400)
    box = PixelBox(5, 160, 390, 430)
    views = region_views(array, box, CropConfig(), ("crop", "context"))

    assert views.shape == (2, 224, 224)
    assert np.array_equal(views[0], crop_from_normalized(array, box, CropConfig())[0])
    assert np.array_equal(views[1], crop_context(array, box, CropConfig())[0])


def test_the_context_view_keeps_a_one_sample_spike() -> None:
    """Linear downsampling can step over a one-column spike; max-pooling cannot."""
    from callisto_trainer.core.preprocess import resize_spectrum

    array = np.zeros((200, 20000), dtype=np.float32)
    array[:, 10001] = 1.0
    box = PixelBox(0, 200, 9000, 11000)
    pooled = crop_context(array, box, CropConfig())
    around = array[:, 7000:13000]
    assert pooled.max() > 0.4
    assert pooled.max() > resize_spectrum(around, (224, 224)).max()


def test_unknown_views_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown view"):
        region_views(_quiet(), PixelBox(0, 10, 0, 10), CropConfig(), ("crop", "spectrum"))


# -- region finder -----------------------------------------------------------


def test_the_fast_finder_matches_a_direct_implementation() -> None:
    from callisto_trainer.core.region_finder import _label_connected, find_candidate_regions

    rng = np.random.RandomState(7)
    for trial in range(5):
        array = (rng.rand(120, 400) > 0.72).astype(np.float32)
        labels, count = _label_connected(array >= 0.5)
        expected = []
        for component in range(1, count + 1):
            rows, cols = np.nonzero(labels == component)
            if rows.size >= 4:
                expected.append((rows.min(), rows.max() + 1, cols.min(), cols.max() + 1,
                                 rows.size))
        expected.sort(key=lambda item: item[4], reverse=True)
        found = find_candidate_regions(
            array, threshold=0.5, min_area=4, max_candidates=10_000, hysteresis=None
        )
        assert [(p.row0, p.row1, p.col0, p.col1, p.area) for p in found] == expected


def _faint_lane(seed_patches: int = 6) -> np.ndarray:
    """A faint drifting burst (~+1.6 dB) that crosses 0.35 only in small patches."""
    array = _quiet(3)
    for row in range(20, 170):
        col = 400 + int((row - 20) * 0.1)
        array[row, col:col + 6] = np.maximum(array[row, col:col + 6], 0.29)
    for index in range(seed_patches):
        row = 30 + index * 22
        col = 400 + int((row - 20) * 0.1)
        array[row:row + 3, col + 1:col + 4] = 0.45   # 9 px each: far below 60
    return array


def test_hysteresis_joins_a_faint_burst_into_one_region() -> None:
    from callisto_trainer.core.region_finder import find_candidate_regions

    lane = _faint_lane()
    assert find_candidate_regions(lane, threshold=0.35, hysteresis=None) == [], (
        "sanity: plain thresholding sees only sub-60 px fragments"
    )
    found = find_candidate_regions(lane, threshold=0.35)
    on_lane = [p for p in found if p.col0 <= 405 and p.col1 >= 415 and p.row1 - p.row0 > 100]
    assert len(on_lane) == 1, "the fragments of one faint burst must become one region"


def test_hysteresis_needs_a_bright_core() -> None:
    from callisto_trainer.core.region_finder import find_candidate_regions

    lane = _faint_lane(seed_patches=0)
    lane[60, 402:405] = 0.45                          # 3 seed pixels: fewer than 5
    assert find_candidate_regions(lane, threshold=0.35) == [], (
        "faint structure with no bright core must never become a candidate"
    )


def test_hysteresis_does_not_widen_a_sharp_region() -> None:
    from callisto_trainer.core.region_finder import find_candidate_regions

    array = np.zeros((200, 800), dtype=np.float32)
    array[50:90, 300:360] = 0.9
    (found,) = find_candidate_regions(array, threshold=0.35)
    assert (found.row0, found.row1, found.col0, found.col1) == (50, 90, 300, 360)
    assert found.area == 40 * 60


def test_the_predictor_follows_its_checkpoints_finder(tmp_path: Path) -> None:
    from callisto_trainer.core.inference import CascadePredictor
    from callisto_trainer.core.region_finder import DEFAULT_HYSTERESIS

    plain = _v2_checkpoint(tmp_path, 0.5)            # records "hysteresis": None
    checkpoint = torch.load(plain, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["inference"]["region_finder"]["hysteresis"] is None
    assert CascadePredictor(load_config(), unified_checkpoint=plain).hysteresis is None

    del checkpoint["config"]["inference"]["region_finder"]["hysteresis"]
    older = tmp_path / "older.pt"
    torch.save(checkpoint, older)
    assert CascadePredictor(load_config(), unified_checkpoint=older).hysteresis == DEFAULT_HYSTERESIS
    assert CascadePredictor(
        load_config(), unified_checkpoint=plain, hysteresis=0.6
    ).hysteresis == 0.6, "an explicit setting wins"


# -- region assignment -------------------------------------------------------


class _Box:
    def __init__(self, row0, row1, col0, col1, burst_type, identifier):
        self.row0, self.row1, self.col0, self.col1 = row0, row1, col0, col1
        self.burst_type, self.id = burst_type, identifier


def test_a_carrier_fragment_inside_a_burst_box_is_not_a_burst() -> None:
    """Boxes average ~1,000 columns; a keyed carrier crossing one used to be labelled a burst."""
    from callisto_trainer.core.negatives import assign_regions

    array = _quiet()
    # An intermittent band, on 200 of every 300 samples; taller than a line, so
    # this exercises the persistence rule on its own.
    for start in range(0, SHAPE[1], 300):
        array[100:130, start:start + 200] = 0.9
    boxes = [_Box(80, 140, 280, 560, "Type II", 1)]  # holds one fragment entirely
    context = file_context(array, _axes())

    with_context = assign_regions(array, boxes, context=context)
    without = assign_regions(array, boxes)
    assert any(r.label == "Type II" for r in without), "sanity: containment alone says burst"
    assert not any(r.label == "Type II" for r in with_context)
    assert any(r.reason.startswith("carrier") for r in with_context)


def test_a_line_inside_a_burst_box_is_never_a_burst_sample() -> None:
    """Regression: 36% of 'Type II' finder samples were lines, a third of them carriers."""
    from callisto_trainer.core.negatives import assign_regions

    array = _quiet()
    array[60:63, 300:500] = 0.9          # a 3-channel segment crossing a generous box
    array[100:140, 320:360] = 0.9        # the compact burst itself
    boxes = [_Box(40, 160, 250, 600, "Type II", 1)]
    by_shape = {(r.row1 - r.row0 <= 22): r for r in assign_regions(array, boxes)}

    assert by_shape[True].label is None and by_shape[True].reason.startswith("line")
    assert by_shape[False].label == "Type II", "the burst itself is still a positive"

    kept = {(r.row1 - r.row0 <= 22): r for r in assign_regions(array, boxes, drop_lines=False)}
    assert kept[True].label == "Type II", "with the option off, lines are positives again"


def test_the_dataset_tab_can_keep_line_positives(label_window) -> None:
    tab = label_window.dataset_tab
    assert tab.drop_lines.isChecked(), "on by default"
    assert "drop_line_positives" not in tab._unified_options()
    tab.drop_lines.setChecked(False)
    assert tab._unified_options()["drop_line_positives"] is False


def test_lines_outside_the_boxes_still_teach_rejection() -> None:
    from callisto_trainer.core.negatives import assign_regions

    array = _quiet()
    array[60:63, 300:500] = 0.9
    array[150:153, 1500:1700] = 0.9
    labels = {
        (r.row0, r.col0): r.label
        for r in assign_regions(array, [_Box(100, 140, 2500, 2700, TYPE_III, 1)])
    }
    assert labels[(60, 300)] == NO_BURST and labels[(150, 1500)] == NO_BURST, (
        "a line outside every burst box is a rejection, named RFI or not at export"
    )


def test_negatives_take_the_inference_visible_regions_first() -> None:
    from callisto_trainer.core.negatives import select_negatives
    from callisto_trainer.core.region_finder import DEFAULT_MAX_REGIONS

    candidates = list(range(30))                  # already largest-first
    hardness = [float(i) for i in range(30)]      # later ones look "harder"
    chosen = select_negatives(candidates, hardness, limit=DEFAULT_MAX_REGIONS)
    assert sorted(chosen) == list(range(DEFAULT_MAX_REGIONS)), (
        "regions inference never examines must not crowd out the ones it does"
    )
    assert chosen[0] == DEFAULT_MAX_REGIONS - 1, "hardest visible region first"


def test_a_scorer_drives_hard_negative_mining() -> None:
    from callisto_trainer.core.negatives import mine_negatives

    array = _quiet()
    for index in range(6):
        array[10 + index * 30: 20 + index * 30, 100:200] = 0.9
    target = (70, 80, 100, 200)
    chosen = mine_negatives(
        array, max_negatives=1,
        scorer=lambda proposals: [
            1.0 if (p.row0, p.row1, p.col0, p.col1) == target else 0.0 for p in proposals
        ],
    )
    assert [(c.row0, c.row1, c.col0, c.col1) for c in chosen] == [target]


# -- store -------------------------------------------------------------------


@pytest.fixture
def repository(tmp_path: Path):
    from callisto_trainer.store.db import Database
    from callisto_trainer.store.repository import AnnotationRepository

    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    ids = [repo.add_file(tmp_path / f"STATION_20260101_000{i}_000{i + 1}.fit.gz", {})
           for i in range(3)]
    return repo, ids


def test_only_burst_boxes_on_burst_files_reach_training(repository) -> None:
    from callisto_trainer.store.repository import VERDICT_BURST, VERDICT_NO_BURST, VERDICT_UNSURE

    repo, (burst, quiet, unsure) = repository
    repo.set_verdict(burst, VERDICT_BURST)
    repo.set_verdict(quiet, VERDICT_NO_BURST)
    repo.set_verdict(unsure, VERDICT_UNSURE)
    repo.add_box(burst, 0, 10, 0, 10, TYPE_III)
    repo.add_box(burst, 40, 50, 0, 10, TYPE_IV)
    repo.add_box(burst, 20, 30, 0, 10, RFI)        # left over from an earlier version
    repo.add_box(quiet, 0, 10, 0, 10, RFI)
    repo.add_box(quiet, 20, 30, 0, 10, TYPE_III)   # contradicts the verdict
    repo.add_box(unsure, 0, 10, 0, 10, TYPE_III)

    pairs = {(record.id, box.burst_type) for record, box in repo.iter_training_boxes()}
    assert pairs == {(burst, TYPE_III), (burst, TYPE_IV)}


def test_marking_no_burst_drops_only_burst_boxes(repository) -> None:
    repo, (file_id, _, _) = repository
    repo.add_box(file_id, 0, 10, 0, 10, TYPE_III)
    repo.add_box(file_id, 20, 30, 0, 10, RFI)
    repo.delete_burst_boxes_for_file(file_id)
    assert [box.burst_type for box in repo.boxes_for_file(file_id)] == [RFI]


# -- training pieces ---------------------------------------------------------


def test_detection_balanced_keeps_background_at_full_weight() -> None:
    from collections import Counter

    from callisto_trainer.core.train_type import _detection_balanced_weights

    names = [NO_BURST, RFI, "Type II", TYPE_III]
    weights = _detection_balanced_weights(
        Counter({0: 3000, 1: 400, 2: 60, 3: 600}), names, exponent=0.5, background_weight=1.0
    )
    assert weights[0] == weights[1] == 1.0, "more background must not make it cheaper"
    assert np.mean(weights[2:]) == pytest.approx(1.0)
    assert weights[2] > weights[3], "the rarer burst type is upweighted"


def test_unified_metrics_separate_ranking_from_typing() -> None:
    from callisto_trainer.core.unified_metrics import unified_region_metrics

    names = [NO_BURST, RFI, TYPE_III, TYPE_IIIG]
    y_true = [0, 1, 2, 3, 3]
    probabilities = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.2, 0.7, 0.05, 0.05],
            [0.1, 0.0, 0.8, 0.1],
            [0.1, 0.0, 0.6, 0.3],   # a group typed as a single Type III
            [0.1, 0.0, 0.2, 0.7],
        ]
    )
    metrics = unified_region_metrics(y_true, probabilities, names)
    assert metrics["detection_ap"] == pytest.approx(1.0), "bursts all outrank non-bursts"
    assert metrics["type_macro_f1"] < 1.0
    assert metrics["family_type_macro_f1"] == pytest.approx(1.0), (
        "Type IIIG versus Type III is a subclass slip, not a typing failure"
    )
    assert metrics["rfi_rejected"] == 1.0
    assert metrics["unified_score"] == pytest.approx(
        (metrics["detection_ap"] + metrics["type_macro_f1"]) / 2
    )


def test_burst_rollup_counts_rfi_as_a_correct_rejection() -> None:
    from callisto_trainer.core.evaluate_unified import burst_rollup

    rollup = burst_rollup(
        {
            "class_names": [NO_BURST, RFI, TYPE_III, TYPE_IIIG],
            "confusion_matrix": [[50, 5, 1, 0], [4, 30, 1, 0], [2, 0, 40, 3], [0, 0, 5, 10]],
        }
    )
    assert rollup["true_negatives"] == 50 + 5 + 4 + 30
    assert rollup["false_positives"] == 2
    assert rollup["iii_iiig_confusions"] == 8
    assert rollup["type_accuracy_given_detected_family"] == pytest.approx(1.0)


def test_multi_view_augmentation_keeps_shape_and_range() -> None:
    from callisto_trainer.core.augmentations import SpectrumAugmenter

    config = load_config()
    config["augmentation"]["probability"] = 1.0
    augmenter = SpectrumAugmenter(config)
    tensor = torch.rand(2, 224, 224)
    out = augmenter(tensor)
    assert out.shape == (2, 224, 224)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# -- the two-view model ------------------------------------------------------


def test_the_two_view_model_shares_its_backbone_and_traces() -> None:
    from callisto_trainer.core.models.model_factory import create_model

    n = feature_count("region_v2")
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=6, use_physics=True, num_physics=n, num_views=2
    ).eval()
    single = create_model(
        "simple_cnn", in_channels=1, num_classes=6, use_physics=True, num_physics=n, num_views=1
    )
    backbone = sum(p.numel() for p in model.backbone.parameters())
    assert backbone == sum(p.numel() for p in single.backbone.parameters()), (
        "a second view must not double the backbone"
    )

    images, features = torch.rand(3, 2, 224, 224), torch.rand(3, n)
    traced = torch.jit.trace(model, (images[:1], features[:1]), strict=False)
    with torch.no_grad():
        assert torch.allclose(traced(images, features), model(images, features), atol=1e-5)


# -- file-level evaluation ---------------------------------------------------


def _scores(quiet: list[float], bursts: list[float]):
    from callisto_trainer.core.file_eval import FileScore

    return [
        FileScore(f"q{i}.fit.gz", f"q{i}.fit.gz", "Q", False, file_score=s)
        for i, s in enumerate(quiet)
    ] + [
        FileScore(f"b{i}.fit.gz", f"b{i}.fit.gz", "B", True, file_score=s, on_burst_score=s)
        for i, s in enumerate(bursts)
    ]


def test_calibration_holds_the_false_alarm_budget() -> None:
    from callisto_trainer.core.file_eval import choose_threshold

    quiet = [i / 20 for i in range(20)]            # 0.00 .. 0.95
    bursts = [0.5, 0.8, 0.9, 0.97, 0.99]
    choice = choose_threshold(_scores(quiet, bursts), max_false_alarm_rate=0.05)

    assert choice.met_budget
    assert choice.false_alarm_rate <= 0.05
    # Within budget (above the quiet 0.90) the most bursts found is 2 of 5, and
    # between the quiet 0.95 and the burst 0.97 that costs no false alarm at all:
    # the threshold goes to the middle of that range, not to the budget's edge.
    assert choice.burst_recall == pytest.approx(2 / 5)
    assert choice.false_alarm_rate == 0.0
    assert 0.95 < choice.threshold <= 0.97


def test_calibration_does_not_spend_the_budget_for_nothing() -> None:
    """Regression: the first rule took the lowest threshold within budget.

    Measured on real files, recall was flat over most of the range while false
    alarms fell steeply, so that rule flagged 7 held-out quiet files where 0-1
    would have found as many bursts.
    """
    from callisto_trainer.core.file_eval import choose_threshold

    quiet = [0.1] * 90 + [0.2, 0.25, 0.3, 0.35, 0.6] + [0.05] * 5   # 100 files
    bursts = [0.95, 0.9, 0.85, 0.8, 0.75]
    choice = choose_threshold(_scores(quiet, bursts), max_false_alarm_rate=0.05)
    assert choice.burst_recall == pytest.approx(1.0)
    assert choice.false_alarm_rate == 0.0, "the same recall is available with no false alarm"
    assert 0.6 < choice.threshold <= 0.75


def test_calibration_says_so_when_the_budget_cannot_be_met() -> None:
    from callisto_trainer.core.file_eval import choose_threshold

    choice = choose_threshold(_scores([1.0] * 10, [1.0]), max_false_alarm_rate=0.05)
    assert not choice.met_budget
    assert "unlabelled burst" in choice.note


def test_calibration_without_quiet_files_is_explicit() -> None:
    from callisto_trainer.core.file_eval import choose_threshold

    choice = choose_threshold(_scores([], [0.9]), max_false_alarm_rate=0.05)
    assert choice.threshold == 0.5 and not choice.met_budget and choice.note


def test_a_confident_model_gets_margin_on_both_sides() -> None:
    from callisto_trainer.core.file_eval import choose_threshold

    choice = choose_threshold(_scores([0.0] * 10, [0.9]), max_false_alarm_rate=0.05, floor=0.05)
    assert choice.threshold == pytest.approx((0.05 + 0.9) / 2)
    assert choice.false_alarm_rate == 0.0 and choice.burst_recall == 1.0


def test_file_report_counts_alarms_detections_and_strays() -> None:
    from callisto_trainer.core.file_eval import FileScore, file_level_report

    scores = [
        FileScore("q1", "q1", "A", False, file_score=0.9),
        FileScore("q2", "q2", "A", False, file_score=0.1),
        FileScore("q3", "q3", "B", False, file_score=0.2),
        FileScore("b1", "b1", "A", True, file_score=0.95, on_burst_score=0.95,
                  stray_scores=[0.8, 0.1]),
        FileScore("b2", "b2", "B", True, file_score=0.8, on_burst_score=0.3,
                  stray_scores=[0.8]),
    ]
    report = file_level_report(scores, threshold=0.5)
    assert report["false_alarms"] == 1 and report["quiet_files"] == 3
    assert report["bursts_detected"] == 1, "a hit on unrelated interference is not a detection"
    assert report["stray_detections"] == 2
    assert report["false_alarms_by_station"]["A"] == {"quiet_files": 2, "false_alarms": 1}
    assert [f["file_path"] for f in report["missed_burst_files"]] == ["b2"]


# -- inference thresholds ----------------------------------------------------


def _v2_checkpoint(tmp_path: Path, burst_threshold: float | None = None) -> Path:
    from callisto_trainer.core.models.model_factory import create_model
    from callisto_trainer.store.export import UNIFIED_CLASSES

    config = load_config()
    config["data"]["classes"] = dict(UNIFIED_CLASSES)
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(UNIFIED_CLASSES),
         "use_metadata": False, "use_physics": True,
         "views": ["crop", "context"], "feature_set": "region_v2"}
    )
    if burst_threshold is not None:
        config["inference"] = {
            "burst_threshold": burst_threshold,
            "region_finder": {"threshold": 0.35, "adaptive": True, "min_area": 60,
                              "max_regions": 12, "hysteresis": None},
            "calibration": {"quiet_files": 40, "false_alarm_rate": 0.05,
                            "max_false_alarm_rate": 0.05, "burst_recall": 0.9},
        }
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=len(UNIFIED_CLASSES), use_physics=True,
        num_physics=feature_count("region_v2"), num_views=2,
    )
    path = tmp_path / "unified_v2.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def _fixed(predictor, row: dict[str, float]):
    original = predictor._unified_probabilities
    vector = np.array([row.get(name, 0.0) for name in predictor.unified_class_names])

    def fake(normalized, axes, boxes, rfi_channels=None, quiet=None):
        _, encoded = original(normalized, axes, boxes, rfi_channels, quiet=quiet)
        return np.tile(vector, (len(boxes), 1)), encoded

    return fake


def _burst_spectrum() -> np.ndarray:
    array = _quiet()
    _type_iii(array, 400)
    return array


def _predict(predictor, array):
    from callisto_trainer.core.inference import FileResult

    return predictor.predict_normalized(array, _axes(), FileResult("x.fit.gz", "x.fit.gz"))


def test_the_calibrated_threshold_decides(tmp_path: Path, monkeypatch) -> None:
    from callisto_trainer.core.inference import CascadePredictor

    config = load_config()
    evidence_06 = {NO_BURST: 0.3, RFI: 0.1, TYPE_III: 0.6}

    strict = CascadePredictor(config, unified_checkpoint=_v2_checkpoint(tmp_path, 0.7))
    monkeypatch.setattr(strict, "_unified_probabilities", _fixed(strict, evidence_06))
    result = _predict(strict, _burst_spectrum())
    assert strict.burst_threshold == pytest.approx(0.7)
    assert result.threshold_calibrated
    assert result.predicted_label == "No_Burst", "evidence 0.6 is below the 0.7 threshold"

    lenient = CascadePredictor(
        config, unified_checkpoint=_v2_checkpoint(tmp_path, 0.7), burst_threshold=0.5
    )
    monkeypatch.setattr(lenient, "_unified_probabilities", _fixed(lenient, evidence_06))
    result = _predict(lenient, _burst_spectrum())
    assert result.predicted_label == "Burst"
    assert not result.threshold_calibrated, "an override is not the calibrated threshold"
    assert result.regions[0].burst_type == TYPE_III
    assert result.regions[0].type_confidence == pytest.approx(1.0), (
        "type confidence is conditional on the region being a burst"
    )


def test_rfi_regions_are_reported_never_detected(tmp_path: Path, monkeypatch) -> None:
    from callisto_trainer.core.inference import CascadePredictor

    predictor = CascadePredictor(load_config(), unified_checkpoint=_v2_checkpoint(tmp_path, 0.5))
    monkeypatch.setattr(
        predictor, "_unified_probabilities",
        _fixed(predictor, {NO_BURST: 0.05, RFI: 0.9, TYPE_III: 0.05}),
    )
    result = _predict(predictor, _burst_spectrum())
    assert result.predicted_label == "No_Burst"
    assert result.regions == []
    assert result.rfi_regions and all(region.is_rfi for region in result.rfi_regions)
    assert result.region_summary.startswith("RFI x")


def test_alert_wording_is_relative_to_the_threshold() -> None:
    from callisto_trainer.core.inference import _relative_alert

    assert _relative_alert(0.31, 0.3) != "No alert", "just above threshold is an alert"
    assert _relative_alert(0.29, 0.3) == "No alert"


def test_checkpoint_settings_are_readable(tmp_path: Path) -> None:
    from callisto_trainer.core.inference import checkpoint_inference_settings

    settings = checkpoint_inference_settings(_v2_checkpoint(tmp_path, 0.42))
    assert settings["burst_threshold"] == pytest.approx(0.42)
    assert settings["region_finder"]["max_regions"] == 12


def test_v2_bundle_describes_both_views_and_the_threshold(tmp_path: Path, any_real_file) -> None:
    import importlib.util
    import sys

    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum
    from callisto_trainer.services.model_export import export_model_bundle

    bundle = export_model_bundle(_v2_checkpoint(tmp_path, 0.61), "unified", tmp_path / "exports")
    card = json.loads((bundle.directory / "model_card.json").read_text(encoding="utf-8"))
    assert card["input"]["shape"] == [2, 224, 224]
    assert card["physics_input"]["feature_set"] == "region_v2"
    assert len(card["physics_input"]["feature_order"]) == feature_count("region_v2")
    assert card["output"]["burst_threshold"] == pytest.approx(0.61)
    assert card["output"]["non_burst_classes"] == [NO_BURST, RFI]

    assert bundle.torchscript_path is not None, bundle.torchscript_error
    scripted = torch.jit.load(str(bundle.torchscript_path), map_location="cpu")
    with torch.no_grad():
        assert scripted(torch.rand(1, 2, 224, 224), torch.zeros(1, 28)).shape == (1, 7)

    spec = importlib.util.spec_from_file_location("bundle_v2", bundle.directory / "predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = load_config()
    spectrum, _ = read_fits_spectrum(any_real_file)
    normalized = normalize_full_spectrum(spectrum, config)
    box = (10, 60, 200, 320)
    tensor = module.to_tensor(module.normalize(spectrum), box)
    assert tensor.shape == (1, 2, 224, 224)
    assert np.array_equal(
        tensor[0, 1], crop_context(normalized, PixelBox(*box), CropConfig.from_config(config))[0]
    ), "the standalone context view must be the one the model was trained on"


# -- export, training and calibration end to end ------------------------------


@pytest.fixture
def labelled(tmp_path: Path, axes_files):
    """Real files labelled with burst boxes; the quiet ones get none."""
    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum
    from callisto_trainer.core.region_finder import find_candidate_regions, resolve_threshold
    from callisto_trainer.services.importer import import_files
    from callisto_trainer.store.db import Database
    from callisto_trainer.store.repository import (
        VERDICT_BURST,
        VERDICT_NO_BURST,
        AnnotationRepository,
    )

    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(p) for p in axes_files])
    types = [TYPE_III, TYPE_IIIG, "Type II"]
    for index, record in enumerate(repo.files()):
        spectrum, _ = read_fits_spectrum(record.path)
        normalized = normalize_full_spectrum(spectrum, load_config())
        candidates = find_candidate_regions(
            normalized, threshold=resolve_threshold(normalized), max_candidates=3
        )
        if index % 3 == 2 or not candidates:
            repo.set_verdict(record.id, VERDICT_NO_BURST)
            continue
        repo.set_verdict(record.id, VERDICT_BURST)
        best = candidates[0]
        repo.add_box(
            record.id,
            max(0, best.row0 - 15), min(record.n_freq, best.row1 + 15),
            max(0, best.col0 - 60), min(record.n_time, best.col1 + 60),
            types[index % 3],
        )
    return repo, tmp_path


def _export(labelled, **kwargs):
    from callisto_trainer.store.export import export_unified_dataset

    repo, tmp_path = labelled
    return export_unified_dataset(
        repo, tmp_path / "datasets", load_config(), tmp_path / "outputs", **kwargs
    )


def test_rare_classes_fold_by_default(labelled) -> None:
    result = _export(labelled, synthetic_rfi_ratio=0.0)
    assert TYPE_IIIG not in result.classes
    assert result.folded[TYPE_IIIG] == TYPE_III
    info = json.loads((result.directory / "snapshot.json").read_text(encoding="utf-8"))
    assert info["folded"] == result.folded
    if RFI in result.classes:
        splits = result.class_split_counts[RFI]
        assert sum(splits.values()) >= 20 and all(splits.get(s) for s in ("train", "val", "test"))
    elif result.automatic_rfi:
        assert result.folded[RFI] == NO_BURST, "too little automatic RFI trains as No_Burst"


def test_export_writes_both_views_and_the_features(labelled) -> None:
    from callisto_trainer.core.dataset import CallistoBurstDataset

    result = _export(labelled, min_subclass_boxes=1, synthetic_rfi_ratio=0.0)
    assert TYPE_IIIG in result.classes
    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["rf_band_fraction"] != "" for row in rows)
    assert all(int(row["label_id"]) == result.classes[row["label"]] for row in rows), (
        "class ids are assigned once the final class set is known"
    )

    from callisto_trainer.core.region_inputs import V2_VIEWS, V3_VIEWS
    from callisto_trainer.store.export import DEFAULT_QUIET_VIEW

    views = len(V3_VIEWS if DEFAULT_QUIET_VIEW else V2_VIEWS)
    with np.load(rows[0]["processed_path"]) as sample:
        assert sample["spectrum"].shape == (views, 224, 224)
        assert sample["features"].shape == (feature_count("region_v2"),)
        assert "label_id" not in sample.files, "the manifest alone carries the class id"

    split = rows[0]["split"]
    dataset = CallistoBurstDataset(
        result.manifest_path, split, return_physics_features=True, feature_set="region_v2"
    )
    image, _, features = dataset[0]
    assert tuple(image.shape) == (views, 224, 224)
    assert tuple(features.shape) == (feature_count("region_v2"),)


def test_files_manifest_lists_every_file_with_its_split(labelled) -> None:
    result = _export(labelled, synthetic_rfi_ratio=0.0)
    with result.files_manifest_path.open(encoding="utf-8", newline="") as handle:
        files = list(csv.DictReader(handle))
    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    repo, _ = labelled
    assert len(files) == repo.total_files(), "quiet files with no candidates count too"
    split_of = {row["file_path"]: row["split"] for row in rows}
    for entry in files:
        assert entry["split"] in ("train", "val", "test")
        if entry["file_path"] in split_of:
            assert entry["split"] == split_of[entry["file_path"]]
        if entry["verdict"] == "no_burst":
            assert json.loads(entry["burst_boxes"]) == []


def test_splits_do_not_move_when_mining_changes(labelled) -> None:
    """The split is a property of the files, so two snapshots share their test files."""
    def splits(result):
        with result.files_manifest_path.open(encoding="utf-8", newline="") as handle:
            return {row["file_path"]: row["split"] for row in csv.DictReader(handle)}

    few = _export(labelled, synthetic_rfi_ratio=0.0, negative_ratio=0.5)
    many = _export(labelled, synthetic_rfi_ratio=1.0, negative_ratio=6.0)
    assert few.written != many.written, "sanity: the mining really differed"
    assert splits(few) == splits(many)


def test_the_config_records_the_finder_it_mined_with(labelled) -> None:
    import yaml

    from callisto_trainer.core.region_finder import DEFAULT_HYSTERESIS

    result = _export(labelled, synthetic_rfi_ratio=0.0)
    generated = yaml.safe_load((result.directory / "config.yaml").read_text(encoding="utf-8"))
    assert generated["inference"]["region_finder"]["hysteresis"] == DEFAULT_HYSTERESIS


def test_negatives_with_an_interference_signature_are_named_rfi(labelled) -> None:
    result = _export(labelled, min_subclass_boxes=1, synthetic_rfi_ratio=0.0)
    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    negatives = [r for r in rows if r["region_source"] in ("quiet_file", "burst_file_background")]
    assert negatives, "sanity: the finder proposed background"
    named = [r for r in negatives if r["rfi_kind"]]
    rfi_label = RFI if RFI in result.classes else NO_BURST
    assert all(r["label"] == rfi_label for r in named)
    assert all(r["label"] == NO_BURST for r in negatives if not r["rfi_kind"])
    assert sum(result.automatic_rfi.values()) == len(named)
    positives = [r for r in rows if r["region_source"] in ("manual", "matched_region")]
    assert not any(r["rfi_kind"] for r in positives), "a burst sample is never renamed RFI"


def test_synthetic_rfi_is_added_as_rfi(labelled) -> None:
    result = _export(labelled, min_subclass_boxes=1, synthetic_rfi_ratio=1.0)
    with result.manifest_path.open(encoding="utf-8", newline="") as handle:
        synthetic = [r for r in csv.DictReader(handle) if r["region_source"] == "synthetic_rfi"]
    assert result.synthetic_rfi == len(synthetic)
    assert synthetic, "every quiet file got a pattern, so some must be found"
    assert {row["label"] for row in synthetic} == {RFI}


def test_hard_negative_mining_uses_the_checkpoint(labelled, tmp_path: Path, monkeypatch) -> None:
    from callisto_trainer.core.inference import CascadePredictor

    calls: list[int] = []
    original = CascadePredictor.burst_evidence_for

    def spy(self, normalized, axes, boxes, rfi_channels=None, quiet=None):
        calls.append(len(boxes))
        return original(self, normalized, axes, boxes, rfi_channels, quiet=quiet)

    monkeypatch.setattr(CascadePredictor, "burst_evidence_for", spy)
    result = _export(
        labelled, synthetic_rfi_ratio=0.0, hard_negative_checkpoint=_v2_checkpoint(tmp_path)
    )
    assert calls, "background candidates must be ranked by the previous model"
    info = json.loads((result.directory / "snapshot.json").read_text(encoding="utf-8"))
    assert info["negative_mining"]["hard_negative_checkpoint"]


def test_train_then_calibrate_end_to_end(labelled) -> None:
    """A real (tiny) run: the calibrated threshold must land in the checkpoint."""
    from callisto_trainer.core.file_eval import calibrate_checkpoint
    from callisto_trainer.core.train_type import fit_type

    result = _export(labelled, synthetic_rfi_ratio=0.0)
    config = load_config(result.directory / "config.yaml")
    config["model"].update({"name": "simple_cnn", "pretrained": False})
    config["training"].update(
        {"epochs": 1, "batch_size": 8, "num_workers": 0, "persistent_workers": False,
         "class_balance": {"strategy": "none"}}
    )
    config["performance"].update(
        {"device": "cpu", "mixed_precision": False, "channels_last": False}
    )

    trained = fit_type(config)
    record = calibrate_checkpoint(config, trained["best_alias"], split="train")
    from callisto_trainer.core.file_eval import calibrate_type_priors

    priors = calibrate_type_priors(config, trained["best_alias"], split="train")

    checkpoint = torch.load(trained["best_alias"], map_location="cpu", weights_only=False)
    inference = checkpoint["config"]["inference"]
    from callisto_trainer.core.region_finder import DEFAULT_MAX_REGIONS

    assert inference["burst_threshold"] == pytest.approx(record["threshold"])
    assert inference["region_finder"]["max_regions"] == DEFAULT_MAX_REGIONS
    assert inference["type_priors"]["strength"] == pytest.approx(priors["strength"])
    assert set(inference["type_priors"]["adjustment"]) == {
        name for name in checkpoint["config"]["data"]["classes"] if name not in (NO_BURST, RFI)
    }
    reports = Path(config["paths"]["reports_dir"])
    assert (reports / "train_file_calibration.json").exists()
    assert (reports / "train_file_metrics.json").exists()


# -- labelling ---------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def label_window(qapp, tmp_path: Path, axes_files):
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


def _verdict(window):
    return window.repository.get_file(window.label_tab._current_file_id).verdict


def test_there_is_no_rfi_button(label_window) -> None:
    from PySide6.QtWidgets import QPushButton

    panel = label_window.label_tab.panel
    texts = [button.text() for button in panel.findChildren(QPushButton)]
    types = [text for text in texts if text.endswith(tuple(f"({n})" for n in range(1, 10)))]
    assert types == [f"{name}  ({index})" for index, name in enumerate(BOX_TYPES, start=1)]
    assert not any("RFI" in text for text in texts)


def test_drawing_on_a_quiet_file_marks_a_burst_and_deleting_it_undoes_that(label_window) -> None:
    from callisto_trainer.store.repository import VERDICT_BURST, VERDICT_NO_BURST

    tab = label_window.label_tab
    tab._on_verdict_changed(VERDICT_NO_BURST)
    tab._on_box_created(20, 70, 300, 520)
    box = label_window.repository.boxes_for_file(tab._current_file_id)[0]
    assert box.burst_type in BOX_TYPES, "every drawn box is a burst"
    assert _verdict(label_window) == VERDICT_BURST

    tab._on_box_deleted(box.id)
    assert _verdict(label_window) == VERDICT_NO_BURST, "the implied verdict is withdrawn"


def test_rfi_cannot_be_assigned_to_a_box(label_window) -> None:
    tab = label_window.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = label_window.repository.boxes_for_file(tab._current_file_id)[0]
    tab.panel.select_box(box.id)
    tab._on_type_assigned(RFI)
    assert label_window.repository.boxes_for_file(tab._current_file_id)[0].burst_type != RFI

    tab._on_type_assigned(TYPE_IV)
    assert label_window.repository.boxes_for_file(tab._current_file_id)[0].burst_type == TYPE_IV


def test_marking_no_burst_deletes_the_bursts(label_window, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from callisto_trainer.store.repository import VERDICT_NO_BURST

    tab = label_window.label_tab
    tab._on_box_created(20, 70, 300, 520)
    tab._on_box_created(90, 140, 700, 860)
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    tab._on_verdict_changed(VERDICT_NO_BURST)
    assert label_window.repository.boxes_for_file(tab._current_file_id) == []
    assert _verdict(label_window) == VERDICT_NO_BURST


def test_the_context_preview_is_the_context_tensor(label_window) -> None:
    tab = label_window.label_tab
    tab._on_box_created(20, 70, 300, 520)
    box = label_window.repository.boxes_for_file(tab._current_file_id)[0]
    tab.panel.select_box(box.id)

    expected = crop_context(
        tab._current_bundle.normalized,
        PixelBox(box.row0, box.row1, box.col0, box.col1),
        tab.crop_config,
    )
    assert np.array_equal(tab.panel.context_preview.image.image, expected[0])


def test_the_panel_hints_at_iiig(label_window) -> None:
    panel = label_window.label_tab.panel
    panel.show_physics(BurstPhysics(burst_count=5), TYPE_III)
    assert "Type IIIG" in panel.group_hint.text()
    assert panel.physics_labels["bursts"].text() == "5"


def test_the_dataset_tab_explains_automatic_rfi_and_folding(label_window) -> None:
    tab = label_window.label_tab
    tab._on_box_created(20, 70, 300, 520)

    dataset = label_window.dataset_tab
    dataset.refresh()
    assert "found automatically" in dataset.counts_label.text()
    warnings = dataset.balance_label.text()
    assert "Type IV: 0 box(es)" in warnings and "train as Other" in warnings
    assert "train as Type III" in warnings
    assert "RFI" not in warnings, "RFI is never drawn, so there is no RFI count to warn about"


def test_the_dataset_tab_passes_the_type_frequencies(label_window) -> None:
    dataset = label_window.dataset_tab
    assert dataset._unified_options()["type_frequencies"]["Type III"] == pytest.approx(86.0)
    dataset.type_frequency["Type II"].setValue(5.0)
    assert dataset._unified_options()["type_frequencies"]["Type II"] == pytest.approx(5.0)
