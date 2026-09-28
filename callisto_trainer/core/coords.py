"""Mapping between spectrum pixel indices and physical (MHz, UTC) coordinates.

Annotations are stored twice: as native pixel indices, which are exact and always
available, and as physical coordinates, which are portable across resamplings and
are what a future detection export needs.

Orientation reminder: the spectrum array is ``[frequency, time]``. Row 0 is the
**first row of the FITS image**, which for e-CALLISTO is normally the *highest*
frequency, so ``freq_mhz`` is usually descending. Nothing here assumes a
direction; the axis arrays are the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectrumAxes:
    """Physical axes for one dynamic spectrum.

    ``time_s[j]`` is the offset in seconds of column ``j`` from the first sample.
    ``freq_mhz[i]`` is the frequency in MHz of row ``i``.
    """

    time_s: np.ndarray
    freq_mhz: np.ndarray
    date: str | None = None
    start_time: str | None = None
    freq_axis_source: str = "none"

    @property
    def n_time(self) -> int:
        return int(self.time_s.size)

    @property
    def n_freq(self) -> int:
        return int(self.freq_mhz.size)

    @property
    def freq_descending(self) -> bool:
        return self.n_freq > 1 and float(self.freq_mhz[0]) > float(self.freq_mhz[-1])

    @property
    def is_approximate(self) -> bool:
        """True when the frequency axis came from the unreliable header keywords."""
        return self.freq_axis_source != "axes_table"

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "SpectrumAxes":
        """Build from a :func:`fits_reader.read_fits_axes` metadata dict."""
        return cls(
            time_s=np.asarray(metadata["time_axis_s"], dtype=np.float64),
            freq_mhz=np.asarray(metadata["freq_axis_mhz"], dtype=np.float64),
            date=metadata.get("date"),
            start_time=metadata.get("start_time"),
            freq_axis_source=str(metadata.get("freq_axis_source", "none")),
        )

    @property
    def start_datetime(self) -> datetime | None:
        """Observation start as a datetime, or ``None`` when unparseable."""
        if not self.date or not self.start_time:
            return None
        text = f"{self.date} {str(self.start_time).split('.')[0]}"
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _interp_index_to_value(index: float | np.ndarray, values: np.ndarray) -> Any:
    """Linearly interpolate ``values`` at fractional array positions."""
    if values.size == 0:
        return np.nan
    positions = np.arange(values.size, dtype=np.float64)
    return np.interp(index, positions, values)


def _interp_value_to_index(value: float | np.ndarray, values: np.ndarray) -> Any:
    """Invert :func:`_interp_index_to_value`, tolerating descending axes.

    Falls back to a nearest-sample search when the axis is not monotonic, which
    happens on a few stations with repeated channel entries.
    """
    if values.size == 0:
        return np.nan
    if values.size == 1:
        return np.zeros_like(np.asarray(value, dtype=np.float64))

    positions = np.arange(values.size, dtype=np.float64)
    diffs = np.diff(values)
    if np.all(diffs > 0):
        return np.interp(value, values, positions)
    if np.all(diffs < 0):
        return np.interp(value, values[::-1], positions[::-1])

    # Non-monotonic axis: nearest channel is the only well-defined answer.
    scalar = np.isscalar(value) or np.asarray(value).ndim == 0
    query = np.atleast_1d(np.asarray(value, dtype=np.float64))
    nearest = np.abs(values[None, :] - query[:, None]).argmin(axis=1).astype(np.float64)
    return float(nearest[0]) if scalar else nearest


def row_to_mhz(axes: SpectrumAxes, row: float | np.ndarray) -> Any:
    """Frequency in MHz at a (possibly fractional) row index."""
    return _interp_index_to_value(row, axes.freq_mhz)


def col_to_seconds(axes: SpectrumAxes, col: float | np.ndarray) -> Any:
    """Offset in seconds from file start at a (possibly fractional) column."""
    return _interp_index_to_value(col, axes.time_s)


def mhz_to_row(axes: SpectrumAxes, mhz: float | np.ndarray) -> Any:
    """Row index for a frequency in MHz."""
    return _interp_value_to_index(mhz, axes.freq_mhz)


def seconds_to_col(axes: SpectrumAxes, seconds: float | np.ndarray) -> Any:
    """Column index for an offset in seconds from file start."""
    return _interp_value_to_index(seconds, axes.time_s)


def seconds_to_utc(axes: SpectrumAxes, seconds: float) -> datetime | None:
    """Absolute UTC timestamp for an offset in seconds, when the date is known."""
    start = axes.start_datetime
    if start is None:
        return None
    return start + timedelta(seconds=float(seconds))


def format_time_label(axes: SpectrumAxes, seconds: float) -> str:
    """Human label for a time offset: absolute UTC when known, else ``+MM:SS``."""
    stamp = seconds_to_utc(axes, seconds)
    if stamp is not None:
        return stamp.strftime("%H:%M:%S")
    total = int(round(float(seconds)))
    return f"+{total // 60:02d}:{total % 60:02d}"


def box_to_physical(
    axes: SpectrumAxes, row0: int, row1: int, col0: int, col1: int
) -> dict[str, float]:
    """Convert a half-open pixel box to physical bounds.

    ``row1``/``col1`` are exclusive, so the last included sample is at index
    ``row1 - 1``. Frequencies are returned as ``lo``/``hi`` regardless of whether
    the underlying axis ascends or descends.
    """
    freq_a = float(row_to_mhz(axes, max(0, int(row0))))
    freq_b = float(row_to_mhz(axes, max(0, int(row1) - 1)))
    time_a = float(col_to_seconds(axes, max(0, int(col0))))
    time_b = float(col_to_seconds(axes, max(0, int(col1) - 1)))
    return {
        "freq_lo_mhz": min(freq_a, freq_b),
        "freq_hi_mhz": max(freq_a, freq_b),
        "t_start_s": min(time_a, time_b),
        "t_end_s": max(time_a, time_b),
    }


def physical_to_box(
    axes: SpectrumAxes,
    freq_lo_mhz: float,
    freq_hi_mhz: float,
    t_start_s: float,
    t_end_s: float,
) -> tuple[int, int, int, int]:
    """Convert physical bounds back to a half-open pixel box ``(r0, r1, c0, c1)``."""
    rows = sorted(
        int(round(float(mhz_to_row(axes, value)))) for value in (freq_lo_mhz, freq_hi_mhz)
    )
    cols = sorted(
        int(round(float(seconds_to_col(axes, value)))) for value in (t_start_s, t_end_s)
    )
    row0 = max(0, min(rows[0], axes.n_freq - 1))
    row1 = min(axes.n_freq, max(rows[1] + 1, row0 + 1))
    col0 = max(0, min(cols[0], axes.n_time - 1))
    col1 = min(axes.n_time, max(cols[1] + 1, col0 + 1))
    return row0, row1, col0, col1
