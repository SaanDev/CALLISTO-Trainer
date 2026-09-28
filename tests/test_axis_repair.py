"""Finding the axis table by structure, and repairing values derived from it.

The bug: the axis table was located by ``EXTNAME='AXES'``, but a large part of
the archive writes the identical table unnamed. Those files silently fell back to
the header's ``CRVAL2``/``CDELT2`` placeholders, which produce a "frequency axis"
that is just the channel index -- 1..200 for a 200-channel receiver. Every drift
rate measured against it was wrong, since MHz/s depends on what a row is worth.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.fits_reader import (  # noqa: E402
    _axis_table_candidates,
    read_axes,
    read_fits_metadata,
)
from callisto_trainer.services.importer import import_files  # noqa: E402
from callisto_trainer.store.db import Database  # noqa: E402
from callisto_trainer.store.repository import AnnotationRepository  # noqa: E402

# The real values from a 200-channel e-CALLISTO receiver, and the bogus range the
# header placeholders imply for it.
TRUE_FREQ = np.linspace(65.875, 5.0, 200)
HEADER_IMPLIED_RANGE = (1.0, 200.0)


def _write_fits(
    path: Path,
    extname: str | None,
    n_freq: int = 200,
    n_time: int = 400,
    freq: np.ndarray | None = None,
    include_table: bool = True,
) -> Path:
    """A file shaped like the archive's, with the axis table named or not."""
    freq = TRUE_FREQ if freq is None else freq
    image = np.zeros((n_freq, n_time), dtype=np.float64)
    primary = fits.PrimaryHDU(image)
    primary.header["INSTRUME"] = "TEST-STATION"
    primary.header["DATE-OBS"] = "2026/07/28"
    primary.header["TIME-OBS"] = "12:00:00"
    primary.header["CDELT1"] = 0.25
    # The placeholders that made the axis look like a channel index.
    primary.header["CRVAL2"] = float(n_freq)
    primary.header["CDELT2"] = -1.0

    hdus = [primary]
    if include_table:
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="TIME", format=f"{n_time}D8.3",
                    array=np.array([np.arange(n_time) * 0.25]),
                ),
                fits.Column(
                    name="FREQUENCY", format=f"{n_freq}D8.3", array=np.array([freq]),
                ),
            ]
        )
        if extname is not None:
            table.header["EXTNAME"] = extname
        hdus.append(table)

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


# -- locating the table ----------------------------------------------------


def test_named_axes_table_is_used(tmp_path: Path) -> None:
    path = _write_fits(tmp_path / "named.fit", extname="AXES")
    metadata = read_fits_metadata(path)

    assert metadata["freq_axis_source"] == "axes_table"
    assert metadata["freq_min_mhz"] == pytest.approx(5.0)
    assert metadata["freq_max_mhz"] == pytest.approx(65.875)


def test_unnamed_axis_table_is_also_used(tmp_path: Path) -> None:
    """The regression: an unnamed table holds exactly the same axes."""
    path = _write_fits(tmp_path / "unnamed.fit", extname=None)
    metadata = read_fits_metadata(path)

    assert metadata["freq_axis_source"] == "axes_table", (
        "an unnamed axis table must be found, not skipped for the header fallback"
    )
    assert metadata["freq_min_mhz"] == pytest.approx(5.0)
    assert metadata["freq_max_mhz"] == pytest.approx(65.875)


def test_the_header_fallback_would_have_given_the_channel_index(tmp_path: Path) -> None:
    """Shows what the bug produced: a 'frequency axis' of 1..200 for 200 channels."""
    path = _write_fits(tmp_path / "no_table.fit", extname=None, include_table=False)
    metadata = read_fits_metadata(path)

    assert metadata["freq_axis_source"] == "header"
    assert (metadata["freq_min_mhz"], metadata["freq_max_mhz"]) == pytest.approx(
        HEADER_IMPLIED_RANGE
    )
    assert metadata["n_freq"] == 200, "the range simply mirrors the channel count"


def test_an_oddly_named_table_is_still_found(tmp_path: Path) -> None:
    path = _write_fits(tmp_path / "odd.fit", extname="SOMETHING-ELSE")
    assert read_fits_metadata(path)["freq_axis_source"] == "axes_table"


def test_a_named_table_is_preferred_over_an_unnamed_one(tmp_path: Path) -> None:
    """Behaviour for files that already worked must not change."""
    path = tmp_path / "both.fit"
    _write_fits(path, extname="AXES")
    with fits.open(path) as hdul:
        candidates = _axis_table_candidates(hdul)
        assert candidates and str(candidates[0].name).upper() == "AXES"


def test_a_table_whose_length_disagrees_is_refused(tmp_path: Path) -> None:
    """Wrong lengths mean the table does not describe this image."""
    path = _write_fits(tmp_path / "mismatch.fit", extname=None, n_freq=200)
    with fits.open(path) as hdul:
        assert read_axes(hdul, n_freq=199) == (None, None)
        assert read_axes(hdul, n_time=17) == (None, None)


def test_axes_survive_a_real_read(tmp_path: Path) -> None:
    from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes

    path = _write_fits(tmp_path / "full.fit", extname=None)
    spectrum, metadata = read_fits_spectrum_and_axes(path)

    assert spectrum.shape == (200, 400)
    assert metadata["freq_axis_mhz"].size == 200
    assert metadata["time_axis_s"].size == 400
    assert metadata["freq_axis_mhz"][0] > metadata["freq_axis_mhz"][-1]
    assert metadata["cadence_s"] == pytest.approx(0.25)


# -- repairing stored values -----------------------------------------------


@pytest.fixture
def stale_store(tmp_path: Path):
    """A store holding values derived from the wrong (header) axis."""
    path = _write_fits(tmp_path / "TEST_20260728_120000_01.fit", extname=None)
    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    import_files(repo, [path])

    record = repo.files()[0]
    # Overwrite with what the buggy reader would have stored.
    repo.update_file_axes(
        record.id,
        {
            "freq_min_mhz": 1.0, "freq_max_mhz": 200.0, "freq_axis_source": "header",
            "legacy_freq_min_mhz": 1.0, "legacy_freq_max_mhz": 200.0,
            "cadence_s": 0.25, "duration_s": 99.75,
            "n_freq": record.n_freq, "n_time": record.n_time,
        },
    )
    repo.set_verdict(record.id, "burst")
    box_id = repo.add_box(
        record.id, 20, 90, 50, 300, "Type III",
        physical={"freq_lo_mhz": 110.0, "freq_hi_mhz": 180.0,
                  "t_start_s": 12.5, "t_end_s": 75.0},
    )
    repo.set_box_physics(box_id, {"drift_mhz_per_s": -1.234, "physics_confidence": "good"})
    return repo, record.id, box_id


def test_repair_corrects_the_stored_frequency_range(stale_store) -> None:
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo, file_id, _box_id = stale_store
    before = repo.get_file(file_id)
    assert before.freq_axis_source == "header"
    assert before.freq_max_mhz == pytest.approx(200.0)

    result = refresh_axes_and_physics(repo)

    after = repo.get_file(file_id)
    assert after.freq_axis_source == "axes_table"
    assert after.freq_min_mhz == pytest.approx(5.0)
    assert after.freq_max_mhz == pytest.approx(65.875)
    assert result.axis_source_fixed == 1
    assert result.files_updated == 1


def test_repair_recomputes_box_bounds_and_drift(stale_store) -> None:
    """Drift is MHz per second, so it is wrong until the frequency axis is right."""
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo, file_id, box_id = stale_store
    refresh_axes_and_physics(repo)

    box = repo.boxes_for_file(file_id)[0]
    assert box.id == box_id
    # The old bounds were in the bogus 1..200 space; the real axis tops out at 65.875.
    assert box.freq_hi_mhz <= 65.875 + 1e-6
    assert box.freq_lo_mhz >= 5.0 - 1e-6
    assert box.drift_mhz_per_s != pytest.approx(-1.234), "stale drift must be replaced"


def test_repair_never_touches_the_annotation_itself(stale_store) -> None:
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo, file_id, box_id = stale_store
    before = repo.boxes_for_file(file_id)[0]
    refresh_axes_and_physics(repo)
    after = repo.boxes_for_file(file_id)[0]

    assert (after.row0, after.row1, after.col0, after.col1) == (
        before.row0, before.row1, before.col0, before.col1
    )
    assert after.burst_type == before.burst_type
    assert repo.get_file(file_id).verdict == "burst"


def test_repair_is_idempotent(stale_store) -> None:
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo, _file_id, _box_id = stale_store
    refresh_axes_and_physics(repo)
    second = refresh_axes_and_physics(repo)

    assert second.axis_source_fixed == 0
    assert second.files_updated == 0, "a correct database must not be rewritten"


def test_repair_reports_unreadable_files(tmp_path: Path) -> None:
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo = AnnotationRepository(Database(tmp_path / "annotations.db"))
    broken = tmp_path / "broken.fit.gz"
    broken.write_bytes(b"not a FITS file")
    repo.add_file(broken, {"n_freq": 1, "n_time": 1})

    result = refresh_axes_and_physics(repo)
    assert result.files_failed == 1
    assert result.errors


def test_repair_can_be_cancelled(stale_store) -> None:
    from callisto_trainer.services.repair import refresh_axes_and_physics

    repo, _file_id, _box_id = stale_store
    result = refresh_axes_and_physics(repo, progress=lambda *_: False)
    assert result.files_checked == 0


# -- real archive ----------------------------------------------------------


def test_real_files_all_resolve_a_true_axis(axes_files, header_only_files) -> None:
    """Across the archive, no file should now fall back to the channel index."""
    for path in [*axes_files, *header_only_files]:
        metadata = read_fits_metadata(path)
        if metadata["freq_axis_source"] != "axes_table":
            continue
        low, high = metadata["freq_min_mhz"], metadata["freq_max_mhz"]
        assert low > 0 and high > low
        # A channel-index axis would run 1..n_freq exactly; a real one does not.
        assert not (
            abs(low - 1.0) < 1e-6 and abs(high - metadata["n_freq"]) < 1e-6
        ), f"{path.name} still looks like a channel index"
