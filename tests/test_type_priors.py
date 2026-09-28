"""Automatic RFI labels, Type IV, and the type-frequency correction.

Two behaviours the operator relies on without ever seeing them:

* interference is named RFI by its measured signature, and a real burst is not;
* the burst type follows how often each type really occurs when the model is
  unsure, without ever changing *whether* a region is a burst.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from callisto_trainer.core.config import load_config
from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.region_features import feature_count, file_context, measure_region
from callisto_trainer.core.rfi_labels import (
    CARRIER,
    GAIN_STEP,
    IMPULSE,
    PERIODIC,
    SWEEP,
    interference_kind,
)
from callisto_trainer.core.taxonomy import (
    NO_BURST,
    OTHER,
    RFI,
    TYPE_II,
    TYPE_III,
    TYPE_IIIG,
    TYPE_IV,
)
from callisto_trainer.core import type_priors
from callisto_trainer.core.unified_metrics import burst_evidence

SHAPE = (200, 3600)


def _quiet(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)


def _axes() -> SpectrumAxes:
    return SpectrumAxes(
        time_s=np.arange(SHAPE[1]) * 0.25, freq_mhz=np.linspace(80.0, 20.0, SHAPE[0])
    )


def _kind(array: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    values = measure_region(file_context(array, _axes()), *box)
    return interference_kind(values, box)


# -- automatic RFI labels ------------------------------------------------------


def test_a_drifting_burst_is_not_interference() -> None:
    array = _quiet()
    for row in range(10, 150):
        col = int(400 + (row - 10) * 0.08)
        array[row, col:col + 6] = 0.8
    assert _kind(array, (10, 150, 395, 420)) is None


def test_a_persistent_carrier_is_a_carrier() -> None:
    array = _quiet()
    array[80:83, :] = 0.9
    assert _kind(array, (78, 85, 1000, 1100)) == CARRIER


def test_a_long_thin_line_is_a_carrier_even_spanning_the_file() -> None:
    array = _quiet()
    array[80:83, 100:3500] = 0.9
    assert _kind(array, (79, 84, 100, 3500)) == CARRIER


def test_a_broadband_spike_is_an_impulse() -> None:
    array = _quiet()
    array[5:195, 1000:1002] = 0.9
    assert _kind(array, (5, 195, 998, 1004)) == IMPULSE


def test_a_regular_pulse_train_is_periodic() -> None:
    array = _quiet()
    for col in range(200, 3400, 40):
        array[60:90, col:col + 2] = 0.9
    assert _kind(array, (60, 90, 1000, 1004)) == PERIODIC


def test_a_flat_full_band_plateau_is_a_gain_step() -> None:
    array = _quiet()
    array[10:190, 1000:1800] += 0.4
    assert _kind(array, (10, 190, 1000, 1800)) == GAIN_STEP


def test_a_thin_sweep_across_the_band_is_a_sweep() -> None:
    array = _quiet()
    rows = np.arange(10, 190)
    array[rows, 1000 + ((rows - 10) * 0.5).astype(int)] = 0.9
    assert _kind(array, (10, 190, 1000, 1091)) == SWEEP


def test_no_features_means_no_interference() -> None:
    assert interference_kind(None, (0, 1, 0, 1)) is None


def test_scarce_rfi_folds_into_background() -> None:
    from callisto_trainer.store.export import _fold_scarce_rfi

    splits = {"a": "train", "b": "val", "c": "test"}
    rows = [{"label": RFI, "file_path": path} for path in ("a", "b", "c") for _ in range(10)]
    assert not _fold_scarce_rfi(rows, splits, minimum=20), "30 samples in every split: kept"

    missing = [{"label": RFI, "file_path": "a"} for _ in range(40)]
    assert _fold_scarce_rfi(missing, splits, minimum=20), "a split without RFI would block"
    assert {row["label"] for row in missing} == {NO_BURST}


# -- observed shares -----------------------------------------------------------


def test_the_default_shares_are_the_observed_frequencies() -> None:
    shares = type_priors.normalise_shares(None)
    assert shares[TYPE_III] == pytest.approx(0.86)
    assert shares[TYPE_IV] == pytest.approx(0.105)
    assert shares[TYPE_II] == pytest.approx(0.025)
    assert shares[OTHER] == pytest.approx(0.01)
    assert type_priors.normalise_shares({TYPE_III: 50, TYPE_II: 50})[TYPE_II] == 0.5, (
        "percent or fractions, normalised either way"
    )


def test_type_iii_share_is_split_with_its_groups() -> None:
    classes = [NO_BURST, RFI, TYPE_II, TYPE_III, TYPE_IIIG, TYPE_IV, OTHER]
    shares = type_priors.class_shares(None, classes, {TYPE_III: 300, TYPE_IIIG: 100})
    assert shares[TYPE_III] == pytest.approx(0.86 * 0.75, rel=1e-3)
    assert shares[TYPE_IIIG] == pytest.approx(0.86 * 0.25, rel=1e-3)
    assert NO_BURST not in shares and RFI not in shares
    assert sum(shares.values()) == pytest.approx(1.0)


def test_a_folded_type_adds_its_share_to_its_fallback() -> None:
    shares = type_priors.class_shares(None, [NO_BURST, TYPE_II, TYPE_III, OTHER])
    assert shares[OTHER] == pytest.approx(0.105 + 0.01, rel=1e-3), "Type IV trains as Other"


def test_the_training_share_includes_the_loss_weight() -> None:
    shares = type_priors.training_shares(
        {TYPE_II: 100, TYPE_III: 400}, {TYPE_II: 2.0, TYPE_III: 1.0}, [NO_BURST, TYPE_II, TYPE_III]
    )
    assert shares[TYPE_II] == pytest.approx(200 / 600)


# -- the correction ------------------------------------------------------------

CLASSES = [NO_BURST, RFI, TYPE_II, TYPE_III]
ADJUSTMENT = type_priors.log_adjustment(
    {TYPE_II: 0.03, TYPE_III: 0.97}, {TYPE_II: 0.4, TYPE_III: 0.6}
)


def test_the_correction_never_changes_burst_evidence() -> None:
    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet(np.ones(len(CLASSES)), size=50)
    adjusted = type_priors.adjust_probabilities(probabilities, CLASSES, ADJUSTMENT, 1.0)
    assert np.allclose(
        burst_evidence(adjusted, CLASSES), burst_evidence(probabilities, CLASSES)
    )
    assert np.allclose(adjusted[:, :2], probabilities[:, :2]), "No_Burst and RFI untouched"
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_an_unsure_burst_becomes_the_commoner_type() -> None:
    unsure = np.array([[0.1, 0.0, 0.5, 0.4]])
    adjusted = type_priors.adjust_probabilities(unsure, CLASSES, ADJUSTMENT, 1.0)
    assert adjusted[0, 3] > adjusted[0, 2], "Type III wins when the model is torn"
    confident = np.array([[0.0, 0.0, 0.99, 0.01]])
    adjusted = type_priors.adjust_probabilities(confident, CLASSES, ADJUSTMENT, 1.0)
    assert adjusted[0, 2] > adjusted[0, 3], "a confident Type II survives"
    assert np.array_equal(
        type_priors.adjust_probabilities(unsure, CLASSES, ADJUSTMENT, 0.0), unsure
    ), "strength 0 is the model as trained"


def test_real_world_precision_weighs_each_type_by_how_common_it_is() -> None:
    # Truth: 10 Type II, 10 Type III. The model calls 2 of the Type III "Type II".
    y_true = np.array([2] * 10 + [3] * 10)
    probabilities = np.zeros((20, 4))
    probabilities[:10, 2] = 1.0
    probabilities[10:18, 3] = 1.0
    probabilities[18:, 2] = 1.0
    report = type_priors.real_world_type_report(
        y_true, probabilities, CLASSES, {TYPE_II: 0.03, TYPE_III: 0.97}
    )
    ii = report["per_type"][TYPE_II]
    assert ii["recall"] == 1.0
    # P(true II | called II) = 0.03 * 1.0 / (0.03 * 1.0 + 0.97 * 0.2)
    assert ii["real_world_precision"] == pytest.approx(0.03 / (0.03 + 0.97 * 0.2))
    assert report["real_world_accuracy"] == pytest.approx(0.03 * 1.0 + 0.97 * 0.8)


def test_a_model_that_over_calls_the_rare_type_gets_corrected() -> None:
    rng = np.random.default_rng(3)
    y_true = np.array([2] * 20 + [3] * 200)
    probabilities = np.zeros((220, 4))
    probabilities[:20, 2] = rng.uniform(0.6, 0.95, 20)       # real Type II, fairly sure
    probabilities[:20, 3] = 1.0 - probabilities[:20, 2]
    probabilities[20:, 2] = rng.uniform(0.3, 0.55, 200)      # Type III, often torn
    probabilities[20:, 3] = 1.0 - probabilities[20:, 2]
    observed = {TYPE_II: 0.03, TYPE_III: 0.97}
    trained = {TYPE_II: 0.5, TYPE_III: 0.5}
    strength, reports = type_priors.choose_strength(
        y_true, probabilities, CLASSES, observed, type_priors.log_adjustment(observed, trained)
    )
    assert strength > 0.0
    by_strength = {r["strength"]: r for r in reports}
    assert by_strength[strength]["real_world_macro_f1"] > by_strength[0.0]["real_world_macro_f1"]


def test_the_choice_is_the_middle_of_the_plateau() -> None:
    assert type_priors.plateau_middle([0.25, 0.5, 0.75, 1.0]) == 0.5, "gentler on a tie"
    assert type_priors.plateau_middle([1.0]) == 1.0
    assert type_priors.plateau_middle([0.0, 0.5, 1.0]) == 0.5


def test_a_correction_that_would_erase_a_rare_type_is_not_taken() -> None:
    """Every real Type II sits at 0.6: any real correction turns them all into Type III."""
    rng = np.random.default_rng(5)
    y_true = np.array([2] * 30 + [3] * 60)
    probabilities = np.zeros((90, 4))
    probabilities[:, 2] = np.where(y_true == 2, 0.6, rng.uniform(0.01, 0.1, 90))
    probabilities[:, 3] = 1.0 - probabilities[:, 2]
    observed = {TYPE_II: 0.03, TYPE_III: 0.97}
    adjustment = type_priors.log_adjustment(observed, {TYPE_II: 0.5, TYPE_III: 0.5})
    strength, reports = type_priors.choose_strength(
        y_true, probabilities, CLASSES, observed, adjustment, groups=list(range(90))
    )
    assert strength == 0.0
    assert reports[-1]["per_type"][TYPE_II]["recall"] == 0.0, "sanity: full strength erases it"
    assert all("shortfall_sd" in report for report in reports[1:])


# -- inference -----------------------------------------------------------------


def _checkpoint(tmp_path: Path, strength: float) -> Path:
    from callisto_trainer.core.models.model_factory import create_model
    from callisto_trainer.store.export import UNIFIED_CLASSES

    config = load_config()
    config["data"]["classes"] = dict(UNIFIED_CLASSES)
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(UNIFIED_CLASSES),
         "use_metadata": False, "use_physics": True,
         "views": ["crop", "context"], "feature_set": "region_v2"}
    )
    config["inference"] = {
        "burst_threshold": 0.5,
        "type_priors": {
            "adjustment": {TYPE_II: -1.5, TYPE_III: 1.0, TYPE_IIIG: 0.0, TYPE_IV: 0.3,
                           OTHER: 0.2},
            "strength": strength,
        },
    }
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=len(UNIFIED_CLASSES), use_physics=True,
        num_physics=feature_count("region_v2"), num_views=2,
    )
    path = tmp_path / "unified_priors.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def _burst() -> np.ndarray:
    array = _quiet()
    for row in range(10, 150):
        col = int(400 + (row - 10) * 0.08)
        array[row, col:col + 6] = 0.8
    return array


def _run(predictor, monkeypatch, row: dict[str, float]):
    from callisto_trainer.core.inference import FileResult

    original = predictor._unified_probabilities
    vector = np.array([row.get(name, 0.0) for name in predictor.unified_class_names])

    def fake(normalized, axes, boxes, rfi_channels=None, quiet=None):
        _, encoded = original(normalized, axes, boxes, rfi_channels, quiet=quiet)
        return np.tile(vector, (len(boxes), 1)), encoded

    monkeypatch.setattr(predictor, "_unified_probabilities", fake)
    return predictor.predict_normalized(_burst(), _axes(), FileResult("x.fit.gz", "x.fit.gz"))


def test_the_predictor_applies_the_calibrated_correction(tmp_path: Path, monkeypatch) -> None:
    from callisto_trainer.core.inference import CascadePredictor

    torn = {NO_BURST: 0.1, TYPE_II: 0.5, TYPE_III: 0.4}
    corrected = CascadePredictor(load_config(), unified_checkpoint=_checkpoint(tmp_path, 1.0))
    result = _run(corrected, monkeypatch, torn)
    assert corrected.type_strength == 1.0
    assert result.regions[0].burst_type == TYPE_III
    assert result.regions[0].burst_evidence == pytest.approx(0.9), "evidence unchanged"
    assert result.type_prior_strength == 1.0

    as_trained = CascadePredictor(
        load_config(), unified_checkpoint=_checkpoint(tmp_path, 1.0), type_prior_strength=0.0
    )
    assert _run(as_trained, monkeypatch, torn).regions[0].burst_type == TYPE_II


def test_the_export_records_the_observed_shares(tmp_path: Path) -> None:
    import yaml

    from callisto_trainer.store.export import ExportResult, write_training_config

    result = ExportResult("unified", tmp_path, tmp_path / "manifest.csv")
    path = write_training_config(
        result, load_config(), tmp_path / "outputs", {NO_BURST: 0, TYPE_III: 1}, task="unified",
        type_frequencies={TYPE_III: 80.0, TYPE_IV: 15.0, TYPE_II: 4.0, OTHER: 1.0},
    )
    priors = yaml.safe_load(path.read_text(encoding="utf-8"))["inference"]["type_priors"]
    assert priors["observed_shares"][TYPE_IV] == pytest.approx(0.15)
    assert priors["strength"] == "auto"
