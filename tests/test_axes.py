"""Frequency/time axis extraction, including the header-placeholder regression."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from callisto_trainer.core.fits_reader import (
    _format_date_token,
    compute_frequency_bounds,
    read_axes,
    read_fits_axes,
    read_fits_metadata,
    synthesize_axes,
)


def test_axes_table_is_preferred_over_header(axes_files) -> None:
    for path in axes_files:
        metadata = read_fits_metadata(path)
        assert metadata["freq_axis_source"] == "axes_table", path.name
        assert metadata["freq_min_mhz"] is not None
        assert metadata["freq_max_mhz"] > metadata["freq_min_mhz"]


def test_header_bounds_are_physically_impossible_but_still_reported(axes_files) -> None:
    """The legacy values are kept for A/B comparison, not because they are right.

    Across this archive the header keywords are placeholders; several stations
    yield negative frequencies, which is the clearest possible proof they do not
    describe a real radio band.
    """
    legacy_values = [read_fits_metadata(path)["legacy_freq_min_mhz"] for path in axes_files]
    assert all(value is not None for value in legacy_values)


def test_regression_alaska_anchorage_frequency_range(axes_files) -> None:
    """Named regression: header said 20-200 MHz, the true axis is 5.875-65.875."""
    target = next(
        (p for p in axes_files if p.name == "ALASKA-ANCHORAGE_20230613_2301_2306.fit.gz"),
        None,
    )
    if target is None:
        pytest.skip("specific regression file not in the sampled set")

    metadata = read_fits_metadata(target)
    assert metadata["freq_axis_source"] == "axes_table"
    assert metadata["freq_min_mhz"] == pytest.approx(5.875, abs=0.01)
    assert metadata["freq_max_mhz"] == pytest.approx(65.875, abs=0.01)
    assert metadata["legacy_freq_min_mhz"] == pytest.approx(20.0, abs=0.01)
    assert metadata["legacy_freq_max_mhz"] == pytest.approx(200.0, abs=0.01)


def test_frequency_axis_is_descending_and_aligned_to_rows(axes_files) -> None:
    """Row 0 is the highest frequency, and the axis length matches the image."""
    for path in axes_files:
        metadata = read_fits_axes(path)
        freq = metadata["freq_axis_mhz"]
        assert freq.size == metadata["n_freq"], path.name
        assert freq[0] > freq[-1], f"expected descending frequency axis in {path.name}"


def test_time_axis_starts_at_zero_and_matches_width(axes_files) -> None:
    for path in axes_files:
        metadata = read_fits_axes(path)
        time_s = metadata["time_axis_s"]
        assert time_s.size == metadata["n_time"], path.name
        assert time_s[0] == pytest.approx(0.0)
        assert np.all(np.diff(time_s) > 0)
        assert metadata["cadence_s"] == pytest.approx(0.25, abs=0.2)


def test_unnamed_axis_tables_resolve_a_real_axis(header_only_files) -> None:
    """These files write the axis table without an EXTNAME; it must still be used."""
    for path in header_only_files:
        metadata = read_fits_axes(path)
        assert metadata["freq_axis_source"] == "axes_table", path.name
        assert metadata["freq_axis_mhz"].size == metadata["n_freq"]
        # A channel-index axis would run exactly 1..n_freq; a real one does not.
        assert not (
            abs(metadata["freq_min_mhz"] - 1.0) < 1e-6
            and abs(metadata["freq_max_mhz"] - metadata["n_freq"]) < 1e-6
        )


def test_header_fallback_when_there_is_no_axis_table(no_axis_table_file) -> None:
    """With no table at all the header is all there is, and it is flagged as such."""
    metadata = read_fits_axes(no_axis_table_file)

    assert metadata["freq_axis_source"] == "header"
    # Axes are still synthesized so the canvas and coordinates always work.
    assert metadata["freq_axis_mhz"].size == metadata["n_freq"]
    assert metadata["time_axis_s"].size == metadata["n_time"]
    assert metadata["freq_min_mhz"] == metadata["legacy_freq_min_mhz"]


def test_read_axes_rejects_length_mismatch(any_real_file) -> None:
    """A table that does not describe this image must be refused, not trusted."""
    with fits.open(any_real_file, memmap=False) as hdul:
        good_time, good_freq = read_axes(hdul)
        assert good_time is not None and good_freq is not None

        assert read_axes(hdul, n_freq=good_freq.size + 1) == (None, None)
        assert read_axes(hdul, n_time=good_time.size + 7) == (None, None)


def test_synthesize_axes_shapes() -> None:
    header = {"CDELT1": 0.25, "CRVAL2": 200.0, "CDELT2": -1.0, "NAXIS2": 10}
    time_s, freq_mhz = synthesize_axes(header, n_freq=10, n_time=40)
    assert time_s.shape == (40,) and freq_mhz.shape == (10,)
    assert time_s[1] == pytest.approx(0.25)
    assert freq_mhz[0] == pytest.approx(200.0)
    assert freq_mhz[-1] == pytest.approx(191.0)


def test_compute_frequency_bounds_handles_missing_keywords() -> None:
    assert compute_frequency_bounds({}, n_freq=10) == (None, None)
    assert compute_frequency_bounds({"CRVAL2": 200.0, "CDELT2": -1.0}, n_freq=0) == (None, None)


def test_date_token_accepts_slash_separated_header_dates() -> None:
    """DATE-OBS uses slashes; the original parser silently rejected that form."""
    assert _format_date_token("2023/06/13") == "2023-06-13"
    assert _format_date_token("2023-06-13") == "2023-06-13"
    assert _format_date_token("20230613") == "2023-06-13"
    assert _format_date_token("nonsense") is None
    assert _format_date_token(None) is None


def test_rfi_channels_are_exposed_when_present(axes_files) -> None:
    found = [read_fits_axes(p)["rfi_channels_mhz"] for p in axes_files]
    assert any(channels is not None and channels.size > 0 for channels in found)
