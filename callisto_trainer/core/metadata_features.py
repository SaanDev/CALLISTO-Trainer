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
from datetime import date as _date
from typing import Any, Mapping

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
