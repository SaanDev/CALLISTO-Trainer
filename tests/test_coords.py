"""Pixel <-> physical coordinate mapping, including descending frequency axes."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from callisto_trainer.core.coords import (
    SpectrumAxes,
    box_to_physical,
    col_to_seconds,
    format_time_label,
    mhz_to_row,
    physical_to_box,
    row_to_mhz,
    seconds_to_col,
    seconds_to_utc,
)
from callisto_trainer.core.fits_reader import read_fits_axes


@pytest.fixture
def descending_axes() -> SpectrumAxes:
    """Mimics a real e-CALLISTO file: row 0 is the highest frequency."""
    return SpectrumAxes(
        time_s=np.arange(1200, dtype=np.float64) * 0.25,
        freq_mhz=np.linspace(65.875, 5.875, 181),
        date="2023-06-13",
        start_time="23:00:59",
        freq_axis_source="axes_table",
    )


@pytest.fixture
def ascending_axes() -> SpectrumAxes:
    return SpectrumAxes(
        time_s=np.arange(500, dtype=np.float64) * 0.5,
        freq_mhz=np.linspace(10.0, 400.0, 200),
        freq_axis_source="axes_table",
    )


def test_row_zero_is_the_highest_frequency(descending_axes: SpectrumAxes) -> None:
    assert descending_axes.freq_descending
    assert row_to_mhz(descending_axes, 0) == pytest.approx(65.875)
    assert row_to_mhz(descending_axes, 180) == pytest.approx(5.875)


def test_round_trip_row_to_mhz(descending_axes: SpectrumAxes) -> None:
    for row in [0, 1, 45, 90, 137, 180]:
        mhz = row_to_mhz(descending_axes, row)
        assert mhz_to_row(descending_axes, mhz) == pytest.approx(row, abs=1e-6)


def test_round_trip_col_to_seconds(descending_axes: SpectrumAxes) -> None:
    for col in [0, 7, 300, 999, 1199]:
        seconds = col_to_seconds(descending_axes, col)
        assert seconds_to_col(descending_axes, seconds) == pytest.approx(col, abs=1e-6)


def test_round_trip_on_ascending_axis(ascending_axes: SpectrumAxes) -> None:
    assert not ascending_axes.freq_descending
    for row in [0, 33, 150, 199]:
        mhz = row_to_mhz(ascending_axes, row)
        assert mhz_to_row(ascending_axes, mhz) == pytest.approx(row, abs=1e-6)


def test_box_to_physical_orders_bounds_regardless_of_direction(
    descending_axes: SpectrumAxes,
) -> None:
    physical = box_to_physical(descending_axes, row0=20, row1=60, col0=100, col1=200)

    assert physical["freq_lo_mhz"] < physical["freq_hi_mhz"]
    assert physical["t_start_s"] < physical["t_end_s"]
    # Row 20 is a higher frequency than row 59 on a descending axis.
    assert physical["freq_hi_mhz"] == pytest.approx(row_to_mhz(descending_axes, 20))
    assert physical["freq_lo_mhz"] == pytest.approx(row_to_mhz(descending_axes, 59))
    assert physical["t_start_s"] == pytest.approx(25.0)
    assert physical["t_end_s"] == pytest.approx(49.75)


def test_box_round_trip_through_physical(descending_axes: SpectrumAxes) -> None:
    for box in [(0, 10, 0, 40), (20, 60, 100, 200), (100, 181, 900, 1200)]:
        physical = box_to_physical(descending_axes, *box)
        restored = physical_to_box(
            descending_axes,
            physical["freq_lo_mhz"],
            physical["freq_hi_mhz"],
            physical["t_start_s"],
            physical["t_end_s"],
        )
        assert restored == box


def test_physical_to_box_is_clamped_to_the_array(descending_axes: SpectrumAxes) -> None:
    row0, row1, col0, col1 = physical_to_box(descending_axes, -500.0, 5000.0, -100.0, 1e6)
    assert 0 <= row0 < row1 <= descending_axes.n_freq
    assert 0 <= col0 < col1 <= descending_axes.n_time


def test_non_monotonic_axis_falls_back_to_nearest_channel() -> None:
    """A few stations repeat channel entries; nearest is the only sane answer."""
    axes = SpectrumAxes(
        time_s=np.arange(5, dtype=np.float64),
        freq_mhz=np.array([50.0, 40.0, 40.0, 45.0, 30.0]),
    )
    assert int(mhz_to_row(axes, 49.0)) == 0
    assert int(mhz_to_row(axes, 31.0)) == 4


def test_absolute_utc_and_labels(descending_axes: SpectrumAxes) -> None:
    assert descending_axes.start_datetime == datetime(2023, 6, 13, 23, 0, 59)
    assert seconds_to_utc(descending_axes, 61.0) == datetime(2023, 6, 13, 23, 2, 0)
    assert format_time_label(descending_axes, 61.0) == "23:02:00"


def test_time_label_without_a_date_is_relative() -> None:
    axes = SpectrumAxes(time_s=np.arange(10, dtype=np.float64), freq_mhz=np.arange(5.0))
    assert axes.start_datetime is None
    assert format_time_label(axes, 125.0) == "+02:05"


def test_approximate_flag_tracks_the_axis_source() -> None:
    real = SpectrumAxes(np.arange(3.0), np.arange(3.0), freq_axis_source="axes_table")
    guessed = SpectrumAxes(np.arange(3.0), np.arange(3.0), freq_axis_source="header")
    assert not real.is_approximate
    assert guessed.is_approximate


def test_axes_from_real_file_round_trip(any_real_file) -> None:
    axes = SpectrumAxes.from_metadata(read_fits_axes(any_real_file))
    box = (10, 50, 100, 400)
    physical = box_to_physical(axes, *box)
    restored = physical_to_box(
        axes,
        physical["freq_lo_mhz"],
        physical["freq_hi_mhz"],
        physical["t_start_s"],
        physical["t_end_s"],
    )
    assert restored == box
