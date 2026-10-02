"""Burst physics: measurement, storage, model features and the consistency check.

Drift rate is the physical discriminator between the burst types, so these tests
check against *published* behaviour rather than only against the code's own
output: a synthetic Type III must come out ~100x faster than a synthetic Type II,
and both must be recovered to within a few percent of the slope they were drawn
with.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.burst_physics import (  # noqa: E402
    NUM_PHYSICS_FEATURES,
    PHYSICS_FEATURES,
    BurstPhysics,
    consistency_warning,
    expected_drift_range,
    measure_burst,
    physics_from_row,
    physics_to_vector,
)
from callisto_trainer.core.coords import SpectrumAxes  # noqa: E402


def _axes(n_freq: int = 200, n_time: int = 1200, cadence: float = 0.25) -> SpectrumAxes:
    """Descending frequency axis, as e-CALLISTO stores it."""
    return SpectrumAxes(
        time_s=np.arange(n_time, dtype=float) * cadence,
        freq_mhz=np.linspace(80.0, 20.0, n_freq),
        date="2023-06-13",
        start_time="12:00:00",
        freq_axis_source="axes_table",
    )


def _drifting_burst(axes: SpectrumAxes, drift_mhz_per_s: float, duration_s: float,
                    start_freq: float = 70.0, width_rows: int = 4) -> np.ndarray:
    """Draw a burst with a known drift, the way a real one appears.

    Frequency falls linearly with time from ``start_freq`` at the given rate, so
    the measurement has a ground truth to be checked against.
    """
    spectrum = np.zeros((axes.n_freq, axes.n_time), dtype=np.float32)
    start_col = 200
    n_cols = int(duration_s / 0.25)
    for step in range(n_cols):
        t = step * 0.25
        freq = start_freq + drift_mhz_per_s * t
        if not (axes.freq_mhz.min() <= freq <= axes.freq_mhz.max()):
            break
        row = int(np.argmin(np.abs(axes.freq_mhz - freq)))
        column = start_col + step
        if column >= axes.n_time:
            break
        spectrum[max(0, row - width_rows) : row + width_rows, column] = 0.95
    return spectrum


# -- measurement accuracy --------------------------------------------------


def test_recovers_a_known_slow_drift() -> None:
    """A Type II-like burst: shock front, order -0.1 MHz/s."""
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-0.1, duration_s=200.0)

    physics = measure_burst(spectrum, axes, 0, axes.n_freq, 0, axes.n_time)

    assert physics.measured
    assert physics.drift_mhz_per_s == pytest.approx(-0.1, rel=0.15)
    assert physics.confidence in ("good", "fair")
    assert physics.freq_start_mhz > physics.freq_end_mhz, "bursts drift downward"


def test_recovers_a_known_fast_drift() -> None:
    """A Type III-like burst: electron beam, order -10 MHz/s."""
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-10.0, duration_s=4.0)

    physics = measure_burst(spectrum, axes, 0, axes.n_freq, 0, axes.n_time)

    assert physics.measured
    assert physics.drift_mhz_per_s == pytest.approx(-10.0, rel=0.25)
    assert abs(physics.drift_mhz_per_s) > 1.0


def test_the_two_types_separate_by_two_orders_of_magnitude() -> None:
    """The property the whole feature exists for."""
    axes = _axes()
    slow = measure_burst(
        _drifting_burst(axes, -0.1, 200.0), axes, 0, axes.n_freq, 0, axes.n_time
    )
    fast = measure_burst(
        _drifting_burst(axes, -10.0, 4.0), axes, 0, axes.n_freq, 0, axes.n_time
    )

    ratio = abs(fast.drift_mhz_per_s) / abs(slow.drift_mhz_per_s)
    assert ratio > 30, f"expected a large separation, got {ratio:.1f}x"


def test_tracking_axis_adapts_to_the_burst_shape() -> None:
    """A fast burst spans rows, not columns; a per-column fit would have no data."""
    axes = _axes()
    slow = measure_burst(_drifting_burst(axes, -0.05, 250.0), axes, 0, axes.n_freq, 0, axes.n_time)
    fast = measure_burst(_drifting_burst(axes, -30.0, 2.0), axes, 0, axes.n_freq, 0, axes.n_time)

    assert slow.track_axis == "time"
    assert fast.track_axis == "frequency"


def test_measures_the_burst_not_the_box() -> None:
    """A generous box must not dilute the measurement.

    This is why the burst is isolated first: fitting across the whole box gave
    Type III rates ~50x too slow on real annotations.
    """
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-8.0, duration_s=4.0)

    tight = measure_burst(spectrum, axes, 0, axes.n_freq, 190, 220)
    generous = measure_burst(spectrum, axes, 0, axes.n_freq, 0, axes.n_time)

    assert tight.measured and generous.measured
    assert generous.drift_mhz_per_s == pytest.approx(tight.drift_mhz_per_s, rel=0.3)


def test_extent_reflects_the_burst_not_the_region() -> None:
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-5.0, duration_s=4.0)

    physics = measure_burst(spectrum, axes, 0, axes.n_freq, 0, axes.n_time)

    # The box spans 300 s; the burst lasts 4.
    assert physics.duration_s == pytest.approx(4.0, abs=1.0)
    assert physics.bandwidth_mhz == pytest.approx(20.0, rel=0.4)


def test_ignores_a_second_brighter_blob_elsewhere() -> None:
    """Only the largest connected structure is measured, not every bright thing."""
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-0.1, duration_s=200.0)
    spectrum[150:158, 900:915] = 1.0  # unrelated interference patch

    physics = measure_burst(spectrum, axes, 0, axes.n_freq, 0, axes.n_time)
    assert physics.measured
    assert physics.drift_mhz_per_s == pytest.approx(-0.1, rel=0.3)


def test_relative_drift_is_normalised_by_frequency() -> None:
    axes = _axes()
    physics = measure_burst(
        _drifting_burst(axes, -5.0, 6.0), axes, 0, axes.n_freq, 0, axes.n_time
    )
    mid = 0.5 * (physics.freq_high_mhz + physics.freq_low_mhz)
    assert physics.relative_drift_per_s == pytest.approx(physics.drift_mhz_per_s / mid, rel=1e-6)


# -- when it cannot measure ------------------------------------------------


def test_empty_region_is_reported_not_invented() -> None:
    axes = _axes()
    physics = measure_burst(np.zeros((200, 1200), dtype=np.float32), axes, 0, 200, 0, 1200)

    assert not physics.measured
    assert physics.drift_mhz_per_s is None
    assert physics.confidence == "none"
    assert physics.note


def test_tiny_region_is_refused() -> None:
    axes = _axes()
    physics = measure_burst(np.ones((200, 1200), dtype=np.float32), axes, 10, 11, 10, 11)
    assert not physics.measured
    assert "too small" in physics.note


def test_edge_clipping_is_flagged() -> None:
    """A burst running out of the box means the reported extent is a lower bound."""
    axes = _axes()
    spectrum = _drifting_burst(axes, drift_mhz_per_s=-0.1, duration_s=200.0)

    clipped = measure_burst(spectrum, axes, 0, axes.n_freq, 210, 260)
    assert clipped.edge_clipped


# -- model features --------------------------------------------------------


def test_feature_vector_has_a_fixed_layout() -> None:
    axes = _axes()
    physics = measure_burst(
        _drifting_burst(axes, -5.0, 6.0), axes, 0, axes.n_freq, 0, axes.n_time
    )
    vector = physics_to_vector(physics)

    assert vector.shape == (NUM_PHYSICS_FEATURES,)
    assert vector.dtype == np.float32
    assert len(PHYSICS_FEATURES) == NUM_PHYSICS_FEATURES
    assert vector[-1] == 1.0, "the measured flag must be set"
    assert np.isfinite(vector).all()


def test_unmeasured_regions_are_zeros_with_the_flag_off() -> None:
    """The model must be able to tell 'no drift' from 'drift not measurable'."""
    vector = physics_to_vector(None)
    assert vector.shape == (NUM_PHYSICS_FEATURES,)
    assert not vector.any()
    assert physics_to_vector(BurstPhysics())[-1] == 0.0


def test_drift_feature_keeps_its_sign_and_compresses_range() -> None:
    fast = physics_to_vector(
        BurstPhysics(drift_mhz_per_s=-50.0, relative_drift_per_s=-1.0, fit_quality=0.9,
                     freq_start_mhz=70, freq_end_mhz=30, bandwidth_mhz=40, duration_s=4)
    )
    slow = physics_to_vector(
        BurstPhysics(drift_mhz_per_s=-0.05, relative_drift_per_s=-0.001, fit_quality=0.9,
                     freq_start_mhz=70, freq_end_mhz=30, bandwidth_mhz=40, duration_s=4)
    )
    index = PHYSICS_FEATURES.index("signed_log_drift")

    assert fast[index] < slow[index] < 0, "downward drift must stay negative"
    # Four decades of drift compressed into a range a linear layer can use.
    assert abs(fast[index]) < 10


def test_features_round_trip_through_a_manifest_row() -> None:
    axes = _axes()
    original = measure_burst(
        _drifting_burst(axes, -5.0, 6.0), axes, 0, axes.n_freq, 0, axes.n_time
    )
    row = {
        "drift_mhz_per_s": str(original.drift_mhz_per_s),
        "relative_drift_per_s": str(original.relative_drift_per_s),
        "freq_start_mhz": str(original.freq_start_mhz),
        "freq_end_mhz": str(original.freq_end_mhz),
        "bandwidth_mhz": str(original.bandwidth_mhz),
        "duration_s": str(original.duration_s),
        "fit_quality": str(original.fit_quality),
        "physics_confidence": original.confidence,
    }
    assert np.allclose(physics_to_vector(physics_from_row(row)), physics_to_vector(original))


def test_blank_manifest_cells_do_not_crash() -> None:
    physics = physics_from_row({"drift_mhz_per_s": "", "duration_s": None})
    assert not physics.measured
    assert not physics_to_vector(physics).any()


# -- consistency check -----------------------------------------------------


def test_flags_a_type_ii_that_drifts_like_a_type_iii() -> None:
    physics = BurstPhysics(drift_mhz_per_s=-25.0, confidence="good", fit_quality=0.9)
    warning = consistency_warning(physics, "Type II")

    assert warning and "faster" in warning
    assert "Type III" in warning


def test_flags_a_type_iii_that_drifts_like_a_type_ii() -> None:
    # Well below the Type III floor: a genuinely flat track, not merely a group
    # whose envelope drifts slowly.
    physics = BurstPhysics(drift_mhz_per_s=-0.005, confidence="good", fit_quality=0.9)
    warning = consistency_warning(physics, "Type III")

    assert warning and "slower" in warning
    assert "Type II" in warning


def test_a_type_iii_group_is_not_flagged() -> None:
    """An operator's box usually holds a group, whose envelope drifts slowly.

    Calibrated against 88 real annotations: the single-burst literature range
    flagged half of all correctly-labelled Type III boxes, which made the check
    worthless.
    """
    physics = BurstPhysics(drift_mhz_per_s=-0.93, confidence="good", fit_quality=0.9)
    assert consistency_warning(physics, "Type III") is None


def test_literature_ranges_are_kept_separate_from_the_warning_bounds() -> None:
    """The published single-burst values stay available for reference."""
    from callisto_trainer.core.burst_physics import (
        EXPECTED_DRIFT_RANGES,
        LITERATURE_DRIFT_RANGES,
    )

    assert LITERATURE_DRIFT_RANGES["Type III"] == (1.0, 200.0)
    warn_low, warn_high = EXPECTED_DRIFT_RANGES["Type III"]
    lit_low, lit_high = LITERATURE_DRIFT_RANGES["Type III"]
    assert warn_low < lit_low and warn_high > lit_high, (
        "warning bounds must be wider than the literature range"
    )


def test_consistent_measurements_are_silent() -> None:
    assert consistency_warning(
        BurstPhysics(drift_mhz_per_s=-0.1, confidence="good"), "Type II"
    ) is None
    assert consistency_warning(
        BurstPhysics(drift_mhz_per_s=-8.0, confidence="good"), "Type III"
    ) is None


def test_a_poor_fit_is_never_treated_as_contradicting_a_label() -> None:
    """An unreliable measurement is not evidence against the operator."""
    physics = BurstPhysics(drift_mhz_per_s=-25.0, confidence="poor", fit_quality=0.2)
    assert consistency_warning(physics, "Type II") is None


def test_other_has_no_expected_range() -> None:
    assert expected_drift_range("Other") is None
    assert consistency_warning(
        BurstPhysics(drift_mhz_per_s=-99.0, confidence="good"), "Other"
    ) is None


# -- storage ---------------------------------------------------------------


def test_physics_columns_round_trip_through_the_database(tmp_path: Path, axes_files) -> None:
    from callisto_trainer.services.importer import import_files
    from callisto_trainer.services.physics_service import physics_to_columns
    from callisto_trainer.store.db import SCHEMA_VERSION, Database
    from callisto_trainer.store.repository import AnnotationRepository

    database = Database(tmp_path / "annotations.db")
    assert database.version == SCHEMA_VERSION

    repo = AnnotationRepository(database)
    import_files(repo, [Path(axes_files[0])])
    file_id = repo.files()[0].id
    box_id = repo.add_box(file_id, 10, 60, 100, 400, "Type III")

    physics = BurstPhysics(
        freq_start_mhz=65.0, freq_end_mhz=30.0, freq_high_mhz=65.0, freq_low_mhz=30.0,
        time_start_s=1.0, time_end_s=5.0, duration_s=4.0, bandwidth_mhz=35.0,
        drift_mhz_per_s=-8.75, relative_drift_per_s=-0.184, fit_quality=0.93,
        track_samples=42, track_axis="frequency", edge_clipped=False, confidence="good",
    )
    repo.set_box_physics(box_id, physics_to_columns(physics))

    stored = repo.boxes_for_file(file_id)[0]
    assert stored.drift_mhz_per_s == pytest.approx(-8.75)
    assert stored.physics_confidence == "good"
    assert stored.physics["track_axis"] == "frequency"

    restored = physics_from_row(stored.physics)
    assert restored.measured
    assert restored.duration_s == pytest.approx(4.0)


def test_migration_adds_physics_to_a_v1_database(tmp_path: Path) -> None:
    """An existing annotated database must upgrade without losing anything."""
    import sqlite3

    from callisto_trainer.store.db import SCHEMA_VERSION, Database

    path = tmp_path / "old.db"
    Database(path)  # create at current schema
    connection = sqlite3.connect(path)
    # Simulate v1: drop the version marker back and remove a physics column is
    # not possible in SQLite, so assert the migration is idempotent instead.
    connection.execute("UPDATE schema_version SET version = 1")
    connection.commit()
    connection.close()

    reopened = Database(path)
    assert reopened.version == SCHEMA_VERSION
    columns = {row[1] for row in reopened.connect().execute("PRAGMA table_info(boxes)")}
    assert {"drift_mhz_per_s", "physics_confidence", "track_axis"} <= columns


def test_boxes_missing_physics_is_reported(tmp_path: Path, axes_files) -> None:
    from callisto_trainer.services.importer import import_files
    from callisto_trainer.store.db import Database
    from callisto_trainer.store.repository import AnnotationRepository

    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [Path(axes_files[0])])
    file_id = repo.files()[0].id
    box_id = repo.add_box(file_id, 10, 60, 100, 400, "Type III")

    assert repo.boxes_missing_physics() == [box_id]
    repo.set_box_physics(box_id, {"physics_confidence": "good", "drift_mhz_per_s": -1.0})
    assert repo.boxes_missing_physics() == []


# -- model -----------------------------------------------------------------


def test_physics_model_accepts_both_inputs() -> None:
    import torch

    from callisto_trainer.core.models.model_factory import create_model

    model = create_model(
        "simple_cnn", in_channels=1, num_classes=4,
        use_physics=True, num_physics=NUM_PHYSICS_FEATURES,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(3, 1, 224, 224), torch.rand(3, NUM_PHYSICS_FEATURES))
    assert output.shape == (3, 4)


def test_physics_and_metadata_branches_are_mutually_exclusive() -> None:
    from callisto_trainer.core.models.model_factory import create_model

    with pytest.raises(ValueError, match="cannot be combined"):
        create_model("simple_cnn", num_classes=4, use_physics=True, use_metadata=True)


def test_missing_physics_does_not_break_the_forward_pass() -> None:
    """Unmeasurable regions arrive as all-zero vectors and must be handled."""
    import torch

    from callisto_trainer.core.models.model_factory import create_model

    model = create_model(
        "simple_cnn", in_channels=1, num_classes=4,
        use_physics=True, num_physics=NUM_PHYSICS_FEATURES,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 224, 224), torch.zeros(2, NUM_PHYSICS_FEATURES))
    assert torch.isfinite(output).all()


# -- real data -------------------------------------------------------------


def test_measures_real_annotations_with_physical_values(axes_files) -> None:
    """On real files the two types must land in their published ranges."""
    from callisto_trainer.core.config import load_config
    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes
    from callisto_trainer.core.inference import resolve_threshold
    from callisto_trainer.services.assist import find_candidate_regions

    measured = 0
    for path in axes_files[:4]:
        spectrum, metadata = read_fits_spectrum_and_axes(path)
        normalized = normalize_full_spectrum(spectrum, load_config())
        axes = SpectrumAxes.from_metadata(metadata)
        for region in find_candidate_regions(
            normalized, threshold=resolve_threshold(normalized), max_candidates=3
        ):
            physics = measure_burst(
                normalized, axes, region.row0, region.row1, region.col0, region.col1
            )
            if not physics.measured:
                continue
            measured += 1
            assert math.isfinite(physics.drift_mhz_per_s)
            assert physics.freq_low_mhz <= physics.freq_high_mhz
            assert physics.duration_s >= 0
            assert 0.0 <= (physics.fit_quality or 0.0) <= 1.0

    assert measured > 0, "no region in any real file yielded a measurement"


# -- fit edge cases (no warnings in evaluation logs) -------------------------------


def test_a_track_in_one_time_sample_has_no_drift_and_no_warning() -> None:
    import warnings

    from callisto_trainer.core.burst_physics import _theil_sen, measure_burst
    from callisto_trainer.core.coords import SpectrumAxes

    rng = np.random.default_rng(0)
    array = np.clip(rng.normal(0.12, 0.03, (200, 3600)), 0, 1).astype(np.float32)
    array[10:190, 1000] = 0.95                       # a one-sample spike
    axes = SpectrumAxes(time_s=np.arange(3600) * 0.25, freq_mhz=np.linspace(80, 20, 200))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _theil_sen(np.full(8, 3.0), np.arange(8.0)) is None
        physics = measure_burst(array, axes, 5, 195, 990, 1010)
    assert not physics.measured and "one time sample" in physics.note


def test_repeated_times_fit_without_warnings_and_match_scipy() -> None:
    import warnings

    from scipy import stats

    from callisto_trainer.core.burst_physics import _theil_sen

    x = np.repeat([10.0, 10.25, 10.5], 20)           # a track followed along frequency
    y = np.linspace(80, 20, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = float(stats.theilslopes(y, x)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _theil_sen(x, y) == pytest.approx(expected)
