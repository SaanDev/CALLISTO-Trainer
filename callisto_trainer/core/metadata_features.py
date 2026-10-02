"""Tabular metadata features for the multi-input burst classifier.

Turns a manifest row (or a FITS metadata dict) into a fixed-length feature
vector used by the metadata branch of the model:

``[station_index, freq_min, freq_max, freq_span, doy_sin, doy_cos, year]``

* ``station_index`` is an integer id from a vocabulary (0 = unknown/unseen).
* frequency features are in GHz-ish units (MHz / 1000) so they sit near ``O(1)``.
* date is encoded as cyclical day-of-year (sin/cos, which generalises across
  years) plus a normalised year.

All functions are pure NumPy/Python so they are tested without PyTorch.
"""

# NOTE: Vendored from H:\Burst Identifier (src/data/metadata_features.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Iterable, Mapping

import numpy as np

# Numeric features that follow the station index in the metadata vector.
NUMERIC_FEATURES = ("freq_min", "freq_max", "freq_span", "doy_sin", "doy_cos", "year")
NUM_NUMERIC = len(NUMERIC_FEATURES)            # 6
META_VECTOR_LEN = 1 + NUM_NUMERIC              # station index + numeric features
FREQ_REF_MHZ = 1000.0
_YEAR_REF = 2010.0
_YEAR_SCALE = 20.0


def _clean_station(value: Any) -> str:
    return str(value or "").strip()


def build_station_vocab(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Map each station name to a 1-based index (0 is reserved for unknown)."""
    stations = sorted({_clean_station(r.get("station")) for r in rows if _clean_station(r.get("station"))})
    return {station: index + 1 for index, station in enumerate(stations)}


def num_stations(vocab: Mapping[str, int]) -> int:
    """Embedding size: known stations + one slot (0) for unknown."""
    return int(len(vocab)) + 1


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _date_features(value: Any) -> tuple[float, float, float]:
    """Return (sin doy, cos doy, normalised year); zeros when unparseable."""
    text = str(value or "").strip()
    parts = text.split("-")
    if len(parts) < 3:
        return 0.0, 0.0, 0.0
    try:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2][:2])
        doy = _date(year, month, day).timetuple().tm_yday
    except (ValueError, TypeError):
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.pi * doy / 365.25
    return math.sin(angle), math.cos(angle), (year - _YEAR_REF) / _YEAR_SCALE


def row_to_meta_vector(row: Mapping[str, Any], vocab: Mapping[str, int]) -> np.ndarray:
    """Build the ``[station_idx, numeric...]`` float32 vector for one row/dict."""
    station_idx = float(vocab.get(_clean_station(row.get("station")), 0))

    freq_min = _safe_float(row.get("freq_min_mhz"))
    freq_max = _safe_float(row.get("freq_max_mhz"))
    freq_min = 0.0 if freq_min is None else freq_min
    freq_max = 0.0 if freq_max is None else freq_max
    freq_span = abs(freq_max - freq_min)

    doy_sin, doy_cos, year_norm = _date_features(row.get("date"))

    return np.array(
        [
            station_idx,
            freq_min / FREQ_REF_MHZ,
            freq_max / FREQ_REF_MHZ,
            freq_span / FREQ_REF_MHZ,
            doy_sin,
            doy_cos,
            year_norm,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Trainer addition: station + date for the unified region model.
#
# The vendored vector above feeds the old whole-file binary model. The unified
# model takes a smaller one, appended after its region features:
#
#     [station_index, month_sin, month_cos, year, date_known]
#
# * ``station_index`` -- 1-based index of a station with enough training files,
#   0 for any other station (unseen, rare or missing).
# * month as a point on the yearly cycle, so December sits next to January.
# * ``year`` -- the observation date as a fractional year, clamped to the span
#   the model was trained on and scaled to [-1, 1]. A file from after the last
#   trained month is treated like that month instead of being extrapolated to.
# * ``date_known`` -- 1 when the date parsed, so "no date" is not read as a date.
#
# How much the model may lean on these is bounded in the model itself (see
# models/physics_model.py, StationDateCorrection), not here.
# ---------------------------------------------------------------------------

STATION_DATE_FEATURES = ("station_index", "month_sin", "month_cos", "year", "date_known")
STATION_DATE_LEN = len(STATION_DATE_FEATURES)
DEFAULT_MIN_STATION_FILES = 10


def station_key(value: Any) -> str:
    """How a station is matched: trimmed and case-folded.

    The name comes from the FITS ``INSTRUME`` header (falling back to the file
    name), and the same station is not always written in the same case.
    """
    return _clean_station(value).upper()


def fractional_year(value: Any) -> tuple[float, int] | None:
    """``(year + (month - 0.5) / 12, month)`` for a ``YYYY-MM...`` date, or None."""
    parts = str(value or "").strip().split("-")
    if len(parts) < 2:
        return None
    try:
        year, month = int(parts[0]), int(parts[1][:2])
    except ValueError:
        return None
    if not (1 <= month <= 12) or not (1900 <= year <= 2200):
        return None
    return year + (month - 0.5) / 12.0, month


@dataclass(frozen=True)
class StationDateEncoder:
    """Turns a file's station and date into the unified model's vector.

    Fitted once on the training split and stored in the checkpoint's config
    (``model.station_date``), so inference encodes exactly as training did.
    """

    vocab: Mapping[str, int] = field(default_factory=dict)
    year_min: float = 0.0
    year_max: float = 0.0

    @classmethod
    def fit(
        cls, rows: Iterable[Mapping[str, Any]], min_station_files: int = DEFAULT_MIN_STATION_FILES
    ) -> "StationDateEncoder":
        """Learn the station list and date span from training rows.

        A station needs ``min_station_files`` distinct files to get its own
        index: an embedding fitted to a handful of files learns those files,
        not the station. Rarer stations share the unknown slot, which is also
        what an unseen station gets at inference.
        """
        files_per_station: dict[str, set[str]] = {}
        years: list[float] = []
        seen_files: set[str] = set()
        for row in rows:
            file_id = str(row.get("file_path") or row.get("source_path") or id(row))
            key = station_key(row.get("station"))
            if key:
                files_per_station.setdefault(key, set()).add(file_id)
            if file_id in seen_files:
                continue
            seen_files.add(file_id)
            parsed = fractional_year(row.get("date"))
            if parsed is not None:
                years.append(parsed[0])
        kept = sorted(k for k, files in files_per_station.items() if len(files) >= min_station_files)
        vocab = {name: index + 1 for index, name in enumerate(kept)}
        if years:
            return cls(vocab=vocab, year_min=float(min(years)), year_max=float(max(years)))
        return cls(vocab=vocab)

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> "StationDateEncoder":
        section = section or {}
        span = section.get("year_range") or [0.0, 0.0]
        return cls(
            vocab={str(k): int(v) for k, v in (section.get("station_vocab") or {}).items()},
            year_min=float(span[0]),
            year_max=float(span[1]),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "station_vocab": dict(self.vocab),
            "year_range": [round(self.year_min, 4), round(self.year_max, 4)],
        }

    @property
    def num_stations(self) -> int:
        """Embedding rows: the known stations plus the unknown slot (0)."""
        return len(self.vocab) + 1

    def station_index(self, station: Any) -> int:
        return int(self.vocab.get(station_key(station), 0))

    def vector(self, station: Any, date: Any) -> np.ndarray:
        out = np.zeros(STATION_DATE_LEN, dtype=np.float32)
        out[0] = float(self.station_index(station))
        parsed = fractional_year(date)
        if parsed is not None:
            year_frac, month = parsed
            angle = 2.0 * math.pi * (month - 0.5) / 12.0
            out[1] = math.sin(angle)
            out[2] = math.cos(angle)
            span = self.year_max - self.year_min
            if span > 1e-6:
                clamped = min(max(year_frac, self.year_min), self.year_max)
                out[3] = 2.0 * (clamped - self.year_min) / span - 1.0
            out[4] = 1.0
        return out

    def vector_for(self, metadata: Mapping[str, Any] | None) -> np.ndarray:
        """The vector for a FITS metadata dict or a manifest row (``station``, ``date``)."""
        metadata = metadata or {}
        return self.vector(metadata.get("station"), metadata.get("date"))
