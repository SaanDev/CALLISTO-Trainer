"""Shared fixtures.

Some tests exercise real e-CALLISTO files from the Burst Identifier archive on
H:. Those tests skip cleanly when the drive is not mounted, so the suite still
runs on a machine that only has this project.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

ARCHIVE = Path(r"H:\Burst Identifier\data")
# Files WITH the AXES extension (5-minute recordings).
TYPES_DIR = ARCHIVE / "Types"
# Files WITHOUT the AXES extension (15-minute recordings) - exercises the fallback.
RAW_NO_BURST_DIR = ARCHIVE / "raw" / "No_Burst"


def _sample(directory: Path, pattern: str, limit: int) -> list[Path]:
    if not directory.exists():
        return []
    return list(itertools.islice(directory.glob(pattern), limit))


# The file behind the documented frequency-axis regression: its header claims
# 20-200 MHz while the AXES table gives the true 5.875-65.875 MHz.
REGRESSION_FILE = TYPES_DIR / "Type III" / "ALASKA-ANCHORAGE_20230613_2301_2306.fit.gz"


@pytest.fixture(scope="session")
def axes_files() -> list[Path]:
    """A few real files that carry the AXES BinTable extension."""
    files = _sample(TYPES_DIR, "*/*.fit.gz", 6)
    if not files:
        pytest.skip(f"Archive not available at {TYPES_DIR}")
    if REGRESSION_FILE.exists() and REGRESSION_FILE not in files:
        files.append(REGRESSION_FILE)
    return files


@pytest.fixture(scope="session")
def header_only_files() -> list[Path]:
    """Files whose axis table is written *unnamed* rather than as ``AXES``.

    These used to fall back to the header placeholders, because the reader looked
    the table up by EXTNAME. They now resolve their true axes like any other file;
    the name is kept so existing tests keep exercising this half of the archive.
    """
    files = _sample(RAW_NO_BURST_DIR, "*.fit.gz", 4)
    if not files:
        pytest.skip(f"Archive not available at {RAW_NO_BURST_DIR}")
    return files


@pytest.fixture
def no_axis_table_file(tmp_path: Path) -> Path:
    """A synthetic file with no axis table at all, to exercise the fallback.

    Written rather than sampled: every file in the archive now resolves a real
    axis, so the fallback can no longer be reached with real data.
    """
    from astropy.io import fits

    header = fits.PrimaryHDU(np.zeros((200, 400), dtype=np.float64))
    header.header["INSTRUME"] = "TEST-STATION"
    header.header["DATE-OBS"] = "2026/07/28"
    header.header["TIME-OBS"] = "12:00:00"
    header.header["CDELT1"] = 0.25
    header.header["CRVAL2"] = 200.0
    header.header["CDELT2"] = -1.0
    path = tmp_path / "TEST_20260728_120000_01.fit"
    fits.HDUList([header]).writeto(path, overwrite=True)
    return path


@pytest.fixture(scope="session")
def any_real_file(axes_files: list[Path]) -> Path:
    return axes_files[0]


@pytest.fixture
def synthetic_spectrum() -> np.ndarray:
    """A deterministic [frequency, time] spectrum with a burst-like ridge."""
    rng = np.random.RandomState(20260727)
    data = rng.rand(181, 1200).astype(np.float32) * 40.0 + 100.0
    # A drifting bright feature, roughly the shape of a Type III lane.
    for step in range(120):
        row = 20 + step
        col = 300 + step * 3
        data[row : row + 3, col : col + 25] += 90.0
    return data
