"""Read e-CALLISTO FITS dynamic spectra, metadata and physical axes.

# NOTE: Vendored from H:\\Burst Identifier (src/preprocessing/fits_reader.py).
# The spectrum reader is unchanged so preprocessed tensors stay byte-identical to
# the original pipeline. See tests/test_preprocess_parity.py.

## Frequency axis correction

The original ``compute_frequency_bounds`` derived the frequency range from the
``CRVAL2``/``CDELT2`` header keywords. In this archive those keywords are
placeholders (``CRVAL2`` ~193-200, ``CDELT2`` = -1) and do **not** describe the
real frequency coverage. Sampled across 10 stations, all 12 files disagreed with
the true axis; e.g. ``ALASKA-ANCHORAGE_20230613_2301_2306`` reports 20-200 MHz
from the header while the real range is 5.875-65.875 MHz.

The true axes live in an ``AXES`` BinTable extension holding ``TIME`` (seconds)
and ``FREQUENCY`` (MHz, stored in row order, typically **descending** so array
row 0 is the highest frequency). That extension is present in roughly half the
archive: the 5-minute ``STATION_YYYYMMDD_HHMM_HHMM.fit.gz`` files carry it, the
15-minute ``STATION_YYYYMMDD_HHMMSS_01.fit.gz`` files do not.

This module therefore prefers the ``AXES`` table, falls back to the header, and
always records which source was used in ``freq_axis_source``. The old
header-derived values are still reported as ``legacy_freq_min_mhz`` /
``legacy_freq_max_mhz`` so a model can be trained either way and compared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# Name of the BinTable extension holding the physical axes.
AXES_EXTENSION = "AXES"
# Name of the BinTable extension listing flagged interference channels.
RFI_EXTENSION = "RFI_FREQ"


def _import_astropy_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Astropy is required to read .fit.gz files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return fits


def _clean_header_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_date_token(token: str | None) -> str | None:
    """Normalise a date token to ``YYYY-MM-DD``.

    Accepts ``YYYYMMDD`` (filenames), ``YYYY-MM-DD`` and ``YYYY/MM/DD``. The
    slash form is what e-CALLISTO writes into ``DATE-OBS``; the original reader
    rejected it and silently fell back to the filename date.
    """
    if token is None:
        return None
    token = token.strip()
    if len(token) == 8 and token.isdigit():
        return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"
    if len(token) >= 10 and token[4] in "-/" and token[7] in "-/":
        return token[:10].replace("/", "-")
    return None


def _format_time_token(token: str | None) -> str | None:
    if token is None:
        return None
    token = token.strip()
    if "." in token:
        whole, frac = token.split(".", 1)
        formatted = _format_time_token(whole)
        return f"{formatted}.{frac}" if formatted else token
    if len(token) == 4 and token.isdigit():
        return f"{token[0:2]}:{token[2:4]}:00"
    if len(token) == 6 and token.isdigit():
        return f"{token[0:2]}:{token[2:4]}:{token[4:6]}"
    if len(token) >= 8 and token[2] == ":" and token[5] == ":":
        return token
    return None


def parse_filename_fallback(path: str | Path) -> dict[str, Any]:
    """Extract station, date, and start time from common e-CALLISTO filenames."""
    path = Path(path)
    name = path.name
    if name.endswith(".fit.gz"):
        stem = name[:-7]
    else:
        stem = path.stem

    parts = stem.split("_")
    metadata: dict[str, Any] = {
        "station": None,
        "date": None,
        "start_time": None,
    }
    if len(parts) >= 3:
        metadata["station"] = parts[0]
        metadata["date"] = _format_date_token(parts[1])
        metadata["start_time"] = _format_time_token(parts[2])
    return metadata


def compute_frequency_bounds(
    header: Any, n_freq: int | None = None
) -> tuple[float | None, float | None]:
    """Legacy header-only frequency bounds from ``CRVAL2``/``CDELT2``.

    Kept behaviour-compatible with the original implementation so the values the
    existing checkpoints were trained on can still be reproduced. These numbers
    are known to be wrong for this archive; prefer :func:`read_axes`.
    """
    try:
        crval = float(header.get("CRVAL2"))
        cdelt = float(header.get("CDELT2"))
    except (TypeError, ValueError):
        return None, None

    if n_freq is None:
        try:
            n_freq = int(header.get("NAXIS2"))
        except (TypeError, ValueError):
            return None, None

    if n_freq <= 0:
        return None, None

    first = crval
    last = crval + (n_freq - 1) * cdelt
    return float(min(first, last)), float(max(first, last))


def _extension_names(hdul: Any) -> set[str]:
    names = set()
    for hdu in hdul:
        name = getattr(hdu, "name", None)
        if name:
            names.add(str(name).strip().upper())
    return names


def _axis_table_candidates(hdul: Any) -> list[Any]:
    """Every extension that could hold the axes, best candidate first.

    The table is located by **structure, not by name**. Many e-CALLISTO files
    write it with ``EXTNAME='AXES'``, but a large part of the archive writes the
    identical table with no ``EXTNAME`` at all. Requiring the name silently threw
    away the real frequency axis for those files and fell back to the header
    placeholders, which produce a "frequency" axis that is just the channel index
    (1..200 for a 200-channel receiver). A named table is still preferred when
    present, so behaviour is unchanged for files that have one.
    """
    named, unnamed = [], []
    for hdu in hdul[1:]:
        if not hasattr(hdu, "columns"):
            continue
        try:
            columns = {name.strip().upper() for name in hdu.columns.names}
        except Exception:
            continue
        if not {"TIME", "FREQUENCY"} <= columns:
            continue
        if str(getattr(hdu, "name", "") or "").strip().upper() == AXES_EXTENSION:
            named.append(hdu)
        else:
            unnamed.append(hdu)
    return named + unnamed


def read_axes(
    hdul: Any, n_freq: int | None = None, n_time: int | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(time_seconds, frequency_mhz)`` from the file's axis table.

    Both arrays are returned in **array row/column order**, so ``frequency[i]``
    is the frequency of spectrum row ``i`` (normally descending) and ``time[j]``
    is the offset of column ``j`` in seconds from the first sample.

    Returns ``(None, None)`` when no extension carries usable axes, or when every
    candidate's lengths disagree with the primary image shape.
    """
    for hdu in _axis_table_candidates(hdul):
        try:
            table = hdu.data
            if table is None:
                continue
            time_s = np.asarray(table["TIME"], dtype=np.float64).ravel()
            freq_mhz = np.asarray(table["FREQUENCY"], dtype=np.float64).ravel()
        except Exception:
            continue

        if time_s.size == 0 or freq_mhz.size == 0:
            continue
        if not np.isfinite(time_s).all() or not np.isfinite(freq_mhz).all():
            continue
        # A length mismatch means the table does not describe this image; skipping
        # it is safer than mapping boxes onto the wrong physical coordinates.
        if n_time is not None and time_s.size != int(n_time):
            continue
        if n_freq is not None and freq_mhz.size != int(n_freq):
            continue

        # Express time as an offset from the first sample. Some writers store the
        # absolute second-of-day here instead of starting at zero.
        return time_s - float(time_s[0]), freq_mhz

    return None, None


def read_rfi_channels(hdul: Any) -> np.ndarray | None:
    """Return the flagged interference frequencies (MHz) from ``RFI_FREQ``."""
    if RFI_EXTENSION not in _extension_names(hdul):
        return None
    try:
        table = hdul[RFI_EXTENSION].data
        if table is None or len(table.columns.names) == 0:
            return None
        values = np.asarray(table[table.columns.names[0]], dtype=np.float64).ravel()
    except Exception:
        return None
    values = values[np.isfinite(values)]
    return values if values.size else None


def _cadence_from_header(header: Any) -> float | None:
    try:
        cadence = float(header.get("CDELT1"))
    except (TypeError, ValueError):
        return None
    return cadence if cadence > 0 else None


def synthesize_axes(header: Any, n_freq: int, n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Build approximate axes from header keywords when ``AXES`` is missing.

    The frequency axis is derived from ``CRVAL2``/``CDELT2``, which are known to
    be unreliable in this archive; callers must surface ``freq_axis_source`` so
    the approximation stays visible to the user.
    """
    cadence = _cadence_from_header(header) or 1.0
    time_s = np.arange(int(n_time), dtype=np.float64) * cadence

    try:
        crval = float(header.get("CRVAL2"))
        cdelt = float(header.get("CDELT2"))
    except (TypeError, ValueError):
        crval, cdelt = float(n_freq), -1.0
    freq_mhz = crval + np.arange(int(n_freq), dtype=np.float64) * cdelt
    return time_s, freq_mhz


def extract_metadata(path: str | Path, hdul: Any) -> dict[str, Any]:
    """Extract core metadata from an opened FITS HDU list.

    Returns JSON-serialisable scalars only (no axis arrays), because
    ``preprocess_file`` embeds this dict in the saved ``.npz``. Use
    :func:`read_fits_axes` when the full axes are needed.
    """
    fallback = parse_filename_fallback(path)
    header = hdul[0].header
    data = hdul[0].data
    shape = tuple(data.shape) if data is not None else ()
    n_freq = int(shape[-2]) if len(shape) >= 2 else None
    n_time = int(shape[-1]) if len(shape) >= 2 else None

    legacy_min, legacy_max = compute_frequency_bounds(header, n_freq=n_freq)
    time_s, freq_mhz = read_axes(hdul, n_freq=n_freq, n_time=n_time)

    if freq_mhz is not None:
        freq_min: float | None = float(np.min(freq_mhz))
        freq_max: float | None = float(np.max(freq_mhz))
        freq_axis_source = "axes_table"
    elif legacy_min is not None:
        freq_min, freq_max = legacy_min, legacy_max
        freq_axis_source = "header"
    else:
        freq_min, freq_max = None, None
        freq_axis_source = "none"

    if time_s is not None and time_s.size > 1:
        cadence_s: float | None = float(np.median(np.diff(time_s)))
        duration_s: float | None = float(time_s[-1] - time_s[0])
    else:
        cadence_s = _cadence_from_header(header)
        duration_s = float(cadence_s * (n_time - 1)) if cadence_s and n_time else None

    header_date = _format_date_token(_clean_header_string(header.get("DATE-OBS")))
    header_time = _format_time_token(_clean_header_string(header.get("TIME-OBS")))
    station = _clean_header_string(header.get("INSTRUME")) or fallback["station"]

    return {
        "station": station,
        "date": header_date or fallback["date"],
        "start_time": header_time or fallback["start_time"],
        "freq_min_mhz": freq_min,
        "freq_max_mhz": freq_max,
        "freq_axis_source": freq_axis_source,
        "legacy_freq_min_mhz": legacy_min,
        "legacy_freq_max_mhz": legacy_max,
        "cadence_s": cadence_s,
        "duration_s": duration_s,
        "n_freq": n_freq,
        "n_time": n_time,
    }


def read_fits_metadata(path: str | Path) -> dict[str, Any]:
    """Read FITS metadata without returning the dynamic spectrum array."""
    fits = _import_astropy_fits()
    with fits.open(path, memmap=False) as hdul:
        return extract_metadata(path, hdul)


def describe_structure(hdul: Any) -> list[dict[str, Any]]:
    """Summarise every HDU: name, type, dimensions. Used by the header panel."""
    summary = []
    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        try:
            shape = tuple(data.shape) if data is not None else ()
        except Exception:
            shape = ()
        summary.append(
            {
                "index": index,
                "name": str(getattr(hdu, "name", "") or f"HDU{index}"),
                "type": type(hdu).__name__,
                "shape": shape,
            }
        )
    return summary


def build_header_text(hdul: Any) -> str:
    """Format the primary header, the extension list and an axis provenance note.

    Shown verbatim in the labelling window so the operator can check what the
    instrument actually recorded, rather than trusting the app's interpretation
    of it -- which matters here, because the frequency keywords are unreliable.
    """
    lines = ["=== PRIMARY HEADER ===", repr(hdul[0].header).strip(), "", "=== EXTENSIONS ==="]
    for entry in describe_structure(hdul):
        shape = " x ".join(str(n) for n in entry["shape"]) or "-"
        lines.append(f"[{entry['index']}] {entry['name']:<12} {entry['type']:<18} {shape}")

    time_s, freq_mhz = read_axes(hdul)
    lines += ["", "=== FREQUENCY AXIS ==="]
    if freq_mhz is not None:
        lines.append(
            f"Source: AXES table -- {freq_mhz.size} channels, "
            f"{freq_mhz.min():.3f} to {freq_mhz.max():.3f} MHz, "
            f"{'descending' if freq_mhz[0] > freq_mhz[-1] else 'ascending'} "
            "(row 0 is the first value)."
        )
    else:
        low, high = compute_frequency_bounds(hdul[0].header)
        lines.append(
            "Source: CRVAL2/CDELT2 header fallback -- this file has no AXES table.\n"
            f"Reported range {low} to {high} MHz. These keywords are placeholders\n"
            "throughout this archive and are usually wrong, so treat the frequency\n"
            "axis for this file as approximate."
        )
    if time_s is not None and time_s.size > 1:
        cadence = float(np.median(np.diff(time_s)))
        lines.append(
            f"Time: {time_s.size} samples, {cadence:.3f} s cadence, "
            f"{time_s[-1] - time_s[0]:.1f} s total."
        )

    rfi = read_rfi_channels(hdul)
    if rfi is not None:
        lines += ["", f"=== FLAGGED RFI: {rfi.size} channel(s) ===", ", ".join(f"{v:.2f}" for v in rfi[:40])]
    return "\n".join(lines)


def read_header_text(path: str | Path) -> str:
    """Read a file purely to format its header (see :func:`build_header_text`)."""
    fits = _import_astropy_fits()
    with fits.open(path, memmap=False) as hdul:
        return build_header_text(hdul)


def _attach_axes(hdul: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Add ``time_axis_s`` / ``freq_axis_mhz`` / ``rfi_channels_mhz`` in place."""
    n_freq, n_time = metadata["n_freq"], metadata["n_time"]
    time_s, freq_mhz = read_axes(hdul, n_freq=n_freq, n_time=n_time)
    if time_s is None or freq_mhz is None:
        time_s, freq_mhz = synthesize_axes(hdul[0].header, n_freq or 0, n_time or 0)
    metadata["time_axis_s"] = time_s
    metadata["freq_axis_mhz"] = freq_mhz
    metadata["rfi_channels_mhz"] = read_rfi_channels(hdul)
    return metadata


def read_fits_axes(path: str | Path) -> dict[str, Any]:
    """Read metadata plus the physical axis arrays and RFI channel list.

    Returns the :func:`extract_metadata` dict with three extra keys:
    ``time_axis_s``, ``freq_axis_mhz`` (always populated, synthesized from the
    header when ``AXES`` is missing) and ``rfi_channels_mhz`` (or ``None``).
    """
    fits = _import_astropy_fits()
    with fits.open(path, memmap=False) as hdul:
        return _attach_axes(hdul, extract_metadata(path, hdul))


def read_fits_spectrum(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Read the primary 2D dynamic spectrum as ``float32`` plus metadata.

    Astropy returns e-CALLISTO primary image data in NumPy order
    ``[frequency, time]`` for the usual FITS axes ``NAXIS2, NAXIS1``.
    This function preserves that scientific orientation.
    """
    fits = _import_astropy_fits()
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No primary HDU image data found in {path}")

        spectrum = np.asarray(data, dtype=np.float32)
        spectrum = np.squeeze(spectrum)
        if spectrum.ndim != 2:
            raise ValueError(
                f"Expected a 2D dynamic spectrum in {path}, got shape {spectrum.shape}"
            )

        metadata = extract_metadata(path, hdul)

    return spectrum, metadata


def read_fits_spectrum_and_axes(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Read the spectrum together with its physical axes in a single open.

    The GUI needs both, and opening a gzipped FITS twice doubles the dominant
    cost of loading a file.
    """
    fits = _import_astropy_fits()
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No primary HDU image data found in {path}")

        spectrum = np.squeeze(np.asarray(data, dtype=np.float32))
        if spectrum.ndim != 2:
            raise ValueError(
                f"Expected a 2D dynamic spectrum in {path}, got shape {spectrum.shape}"
            )

        metadata = _attach_axes(hdul, extract_metadata(path, hdul))
        metadata["header_text"] = build_header_text(hdul)

    return spectrum, metadata
