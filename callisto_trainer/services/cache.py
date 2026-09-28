"""Loading, caching and display-decimation of normalized spectra.

Decoding a gzipped e-CALLISTO file with astropy costs 100-400 ms and dominates
everything else in the labelling loop, so the same array is never produced twice
if it can be helped:

* an in-memory LRU holds the handful of files around the cursor, bounded by
  total bytes rather than count (spectra range from 1 MB to ~28 MB);
* a disk cache stores the normalized float32 array as ``.npy``, keyed by file
  content *and* preprocessing settings, so a second pass through the queue skips
  the gzip entirely.

This module is Qt-free so it can be unit-tested headless.
"""

from __future__ import annotations

import hashlib
import json
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.crops import normalize_full_spectrum, quiet_normalized_spectrum
from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes


@dataclass
class SpectrumBundle:
    """Everything the labelling canvas needs for one file.

    Both the ``raw`` array (as stored in the FITS, uncalibrated digits) and the
    ``normalized`` one (what the model receives) are kept. The raw array costs
    nothing extra to obtain -- it has to be read to produce the normalized one --
    and holding it lets the operator compare the two views and inspect the
    underlying numbers without a second gzip decode.
    """

    file_id: int
    path: str
    normalized: np.ndarray  # [frequency, time] float32 in [0, 1]
    axes: SpectrumAxes
    metadata: dict[str, Any]
    raw: np.ndarray | None = None  # [frequency, time] float32, as stored
    # The quiet-part background (crops.quiet_normalized_spectrum): long continua
    # stay visible in it, and it is the model's third view.
    quiet: np.ndarray | None = None

    @property
    def nbytes(self) -> int:
        total = int(self.normalized.nbytes)
        for extra in (self.raw, self.quiet):
            if extra is not None:
                total += int(extra.nbytes)
        return total

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.normalized.shape[0]), int(self.normalized.shape[1]))

    @property
    def header_text(self) -> str:
        return str(self.metadata.get("header_text", ""))

    @property
    def raw_has_invalid(self) -> bool:
        """True when the stored array contains NaN or Inf samples.

        These are real and worth surfacing: the model never sees them (they are
        replaced by the finite median during preprocessing), so a file with many
        of them is one where what the model learns differs most from what the
        instrument actually recorded.
        """
        return self.raw is not None and not bool(np.isfinite(self.raw).all())

    def raw_levels(self, low_percentile: float = 1.0, high_percentile: float = 99.5) -> tuple[float, float]:
        """Display levels for the raw view.

        Raw e-CALLISTO values are uncalibrated receiver digits whose absolute
        range varies by station and gain setting, so a fixed window is
        meaningless here (unlike the normalized view, whose window is the
        model's). Percentiles keep the display readable without hiding structure,
        and ignore the NaN samples some files carry.
        """
        if self.raw is None:
            return 0.0, 1.0
        # Subsample wide arrays; a percentile over 4 million samples is wasteful
        # when a few hundred thousand give the same answer to display precision.
        data = self.raw[:, :: max(1, self.raw.shape[1] // 2000)]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.percentile(finite, low_percentile))
        high = float(np.percentile(finite, high_percentile))
        return (low, high) if high > low else (low, low + 1.0)


def preprocessing_signature(config: dict[str, Any]) -> str:
    """Short hash of the settings that affect the normalized array.

    Included in the disk cache key so changing the dB window can never serve a
    stale array that was normalized under the old settings.
    """
    prep = config.get("preprocessing", {})
    relevant = {
        "background_method": prep.get("background_method"),
        "normalization": prep.get("normalization"),
        "db_vmin": prep.get("db_vmin"),
        "db_vmax": prep.get("db_vmax"),
        "normalization_clip": prep.get("normalization_clip"),
        "epsilon": prep.get("epsilon"),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=6).hexdigest()


def load_bundle(
    file_id: int,
    path: str | Path,
    config: dict[str, Any],
    disk_cache: "DiskCache | None" = None,
    content_key: str | None = None,
) -> SpectrumBundle:
    """Read a FITS file and return its normalized spectrum plus physical axes."""
    path = Path(path)
    spectrum, metadata = read_fits_spectrum_and_axes(path)
    normalized: np.ndarray | None = None

    if disk_cache is not None and content_key:
        normalized = disk_cache.get(content_key)
    if normalized is None:
        normalized = normalize_full_spectrum(spectrum, config)
        if disk_cache is not None and content_key:
            disk_cache.put(content_key, normalized)

    try:
        quiet = quiet_normalized_spectrum(spectrum, config)
    except ValueError:  # a background method with no per-channel baseline
        quiet = None
    return SpectrumBundle(
        file_id=file_id,
        path=str(path),
        normalized=normalized,
        axes=SpectrumAxes.from_metadata(metadata),
        metadata=metadata,
        raw=spectrum,
        quiet=quiet,
    )


class DiskCache:
    """Normalized arrays persisted as ``.npy``, keyed by content + settings."""

    def __init__(self, directory: str | Path, max_bytes: int = 8 * 1024**3) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()

    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            # A truncated cache entry (e.g. killed mid-write) must never be fatal.
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, array: np.ndarray) -> None:
        path = self._path_for(key)
        temporary = path.with_suffix(".npy.tmp")
        try:
            np.save(temporary, array, allow_pickle=False)
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*.npy"))

    def prune(self) -> int:
        """Drop the least recently used entries until under the size budget."""
        with self._lock:
            entries = sorted(
                self.directory.glob("*.npy"), key=lambda p: p.stat().st_atime
            )
            total = sum(p.stat().st_size for p in entries)
            removed = 0
            for path in entries:
                if total <= self.max_bytes:
                    break
                size = path.stat().st_size
                path.unlink(missing_ok=True)
                total -= size
                removed += 1
            return removed

    def clear(self) -> None:
        for path in self.directory.glob("*.npy"):
            path.unlink(missing_ok=True)


class SpectrumCache:
    """Byte-bounded LRU of decoded spectra."""

    def __init__(self, max_bytes: int = 512 * 1024**2) -> None:
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[int, SpectrumBundle] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, file_id: int) -> SpectrumBundle | None:
        with self._lock:
            bundle = self._items.get(file_id)
            if bundle is not None:
                self._items.move_to_end(file_id)
            return bundle

    def put(self, bundle: SpectrumBundle) -> None:
        with self._lock:
            existing = self._items.pop(bundle.file_id, None)
            if existing is not None:
                self._bytes -= existing.nbytes
            self._items[bundle.file_id] = bundle
            self._bytes += bundle.nbytes
            # Always keep at least one entry, even if it alone exceeds the budget.
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= evicted.nbytes

    def contains(self, file_id: int) -> bool:
        with self._lock:
            return file_id in self._items

    def discard(self, file_id: int) -> None:
        with self._lock:
            bundle = self._items.pop(file_id, None)
            if bundle is not None:
                self._bytes -= bundle.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def decimate_for_display(
    array: np.ndarray, max_cols: int, nan_aware: bool = False
) -> tuple[np.ndarray, int]:
    """Reduce a wide spectrum along time for rendering, returning ``(array, factor)``.

    Uses **max** pooling, not mean. A Type III burst can be a two-pixel-wide
    vertical stripe in a 39,600-column file; averaging would dilute it into the
    background and the operator would never see it. Taking the maximum keeps thin
    features visible at any zoom level.

    Set ``nan_aware`` when the array may contain NaN, as raw FITS arrays do: plain
    ``max`` would let a single invalid sample blank an entire pooled block, hiding
    thousands of good samples behind one bad one. Blocks that are entirely invalid
    still come out as NaN, and render as gaps -- which is the honest result.

    Full resolution is always retained elsewhere -- crops are extracted from the
    undecimated array, so this only affects what is drawn on screen.
    """
    n_cols = int(array.shape[1])
    if max_cols <= 0 or n_cols <= max_cols:
        return array, 1

    factor = int(np.ceil(n_cols / max_cols))
    usable = (n_cols // factor) * factor
    trimmed = array[:, :usable].reshape(array.shape[0], usable // factor, factor)

    if nan_aware:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # An all-NaN block is expected, not exceptional; NaN is the answer.
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = np.nanmax(trimmed, axis=2)
            tail_source = array[:, usable:]
            tail = (
                np.nanmax(tail_source, axis=1, keepdims=True) if tail_source.size else None
            )
    else:
        pooled = trimmed.max(axis=2)
        tail_source = array[:, usable:]
        tail = tail_source.max(axis=1, keepdims=True) if tail_source.size else None

    if tail is not None:
        pooled = np.concatenate([pooled, tail], axis=1)
    return np.ascontiguousarray(pooled, dtype=np.float32), factor
