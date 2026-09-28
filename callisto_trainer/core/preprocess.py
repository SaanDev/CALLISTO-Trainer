"""Preprocess raw e-CALLISTO FITS spectra into training tensors.

Scientific pipeline (per 2D ``[frequency, time]`` spectrum):

1. Invalid-value cleanup (NaN/Inf -> finite median).
2. Background subtraction.
3. Normalization.
4. Resize to the model input shape.

The default background/normalization reproduce the e-CALLISTO FITS Analyzer
``plotutil_median_db`` view, so the training tensor matches the dynamic spectrum
a human inspects (bursts appear as bright, high-intensity regions):

* ``plotutil_median_db`` background: subtract the per-frequency (per-row) median
  over time, then scale digits to dB with ``2500 / 255 / 25.4``. (In the original
  Plotutil, ``dB = (data - global_min) * scale`` followed by a row-median
  subtraction; the global-min offset cancels, leaving ``(data - row_median) *
  scale``.)
* ``db_window`` normalization: linearly map the fixed dB window
  ``[db_vmin, db_vmax] = [-1, 8]`` to ``[0, 1]`` and clip, matching the display
  ``Normalize(vmin, vmax)`` so intense bursts stay bright and are not removed.

There is intentionally **no RFI mitigation** in this pipeline: clipping and
hot-channel masking were removing genuine high-intensity burst signal.

All numeric steps run in NumPy so the produced tensor is identical on every
machine (the GPU is used for training, not for this I/O-bound preprocessing).
"""

# NOTE: Vendored from H:\Burst Identifier (src/preprocessing/preprocess.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from callisto_trainer.core.manifest import read_manifest
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)

# Raw 8-bit digit -> voltage -> dB conversion (Plotutil's Digit2Voltage / 25.4).
PLOTUTIL_DB_SCALE = 2500.0 / 255.0 / 25.4
# Fixed display window used by the FITS Analyzer (dB above background).
PLOTUTIL_DISPLAY_LIMITS = (-1.0, 8.0)


def select_preprocessing_device(config: dict[str, Any]) -> str:
    """Validate and return the requested preprocessing device string.

    Numeric preprocessing always runs in NumPy (it is I/O-bound and this
    guarantees byte-identical tensors across machines); the device string is
    retained for logging and CLI compatibility.
    """
    requested = str(config["preprocessing"].get("device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported preprocessing.device: {requested}")
    return requested


def clean_invalid_values(spectrum: np.ndarray) -> np.ndarray:
    """Convert to float32 and replace NaN/infinite values with the finite median."""
    data = np.asarray(spectrum, dtype=np.float32)
    finite_mask = np.isfinite(data)

    if not finite_mask.any():
        return np.zeros_like(data, dtype=np.float32)

    replacement = np.median(data[finite_mask]).astype(np.float32)
    return np.where(finite_mask, data, replacement).astype(np.float32)


def subtract_background_rows(spectrum: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract a per-frequency (per-row, over time) baseline."""
    data = np.asarray(spectrum, dtype=np.float32)
    if method == "median":
        baseline = np.median(data, axis=1, keepdims=True)
    elif method == "mean":
        baseline = np.mean(data, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported row baseline method: {method}")
    return (data - baseline).astype(np.float32)


def plotutil_median_db(spectrum: np.ndarray) -> np.ndarray:
    """e-CALLISTO Plotutil 'median dB' background: row-median subtract, then dB scale."""
    centered = subtract_background_rows(spectrum, method="median")
    return (centered * np.float32(PLOTUTIL_DB_SCALE)).astype(np.float32)


def subtract_background(spectrum: np.ndarray, method: str = "plotutil_median_db") -> np.ndarray:
    """Apply background subtraction while preserving burst morphology."""
    mode = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if mode == "none":
        return np.asarray(spectrum, dtype=np.float32)
    if mode in {"plotutil_median_db", "plotutil", "plotutil_median", "ecallisto_db"}:
        return plotutil_median_db(spectrum)
    if mode == "per_frequency_median":
        return subtract_background_rows(spectrum, method="median")
    raise ValueError(f"Unsupported background subtraction method: {method}")


def robust_normalize(
    spectrum: np.ndarray,
    clip_value: float = 8.0,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Normalize using global median/MAD and clip extreme normalized values."""
    data = np.asarray(spectrum, dtype=np.float32)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    scale = 1.4826 * max(float(mad), epsilon)
    normalized = (data - median) / scale
    return np.clip(normalized, -clip_value, clip_value).astype(np.float32)


def db_window_normalize(
    spectrum: np.ndarray,
    vmin: float = -1.0,
    vmax: float = 8.0,
) -> np.ndarray:
    """Linearly map the dB window ``[vmin, vmax]`` to ``[0, 1]`` and clip.

    Mirrors ``matplotlib.colors.Normalize(vmin, vmax)`` used for display, giving
    every file the same absolute dB scale (unlike per-file robust scaling) so a
    fixed brightness corresponds to a fixed dB-above-background everywhere.
    """
    data = np.asarray(spectrum, dtype=np.float32)
    span = float(vmax) - float(vmin)
    if span <= 0:
        raise ValueError(f"db_window normalization requires vmax > vmin, got [{vmin}, {vmax}]")
    normalized = (data - float(vmin)) / span
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def normalize_spectrum(spectrum: np.ndarray, prep_cfg: dict[str, Any]) -> np.ndarray:
    """Dispatch normalization based on ``preprocessing.normalization``."""
    method = str(prep_cfg.get("normalization", "db_window")).lower()
    if method == "db_window":
        return db_window_normalize(
            spectrum,
            vmin=float(prep_cfg.get("db_vmin", PLOTUTIL_DISPLAY_LIMITS[0])),
            vmax=float(prep_cfg.get("db_vmax", PLOTUTIL_DISPLAY_LIMITS[1])),
        )
    if method == "median_mad":
        return robust_normalize(
            spectrum,
            clip_value=float(prep_cfg.get("normalization_clip", 8.0)),
            epsilon=float(prep_cfg.get("epsilon", 1.0e-6)),
        )
    raise ValueError(f"Unsupported normalization method: {method}")


def _linear_resample_weights(old_size: int, new_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute gather indices/weights for uniform-grid linear interpolation.

    Returns ``(lower, upper, frac)`` such that ``out[j] = data[lower[j]] * (1 -
    frac[j]) + data[upper[j]] * frac[j]`` reproduces ``np.interp`` on the uniform
    grids ``linspace(0, 1, old_size)`` -> ``linspace(0, 1, new_size)`` exactly,
    but vectorized across the whole axis instead of one Python loop per row/col.
    """
    if new_size == 1 or old_size == 1:
        positions = np.zeros(new_size, dtype=np.float64)
    else:
        positions = np.arange(new_size, dtype=np.float64) * (old_size - 1) / (new_size - 1)
    lower = np.floor(positions).astype(np.intp)
    np.clip(lower, 0, old_size - 1, out=lower)
    upper = np.minimum(lower + 1, old_size - 1)
    frac = (positions - lower).astype(np.float32)
    return lower, upper, frac


def _resize_axis(data: np.ndarray, new_size: int, axis: int) -> np.ndarray:
    old_size = data.shape[axis]
    if old_size == new_size:
        return data.astype(np.float32, copy=False)

    lower, upper, frac = _linear_resample_weights(old_size, new_size)

    # Vectorized linear interpolation (same result as the original per-row
    # ``np.interp`` loop, ~100x faster for 224x224 targets over 500k+ files).
    if axis == 1:
        frac_row = frac[np.newaxis, :]
        resized = data[:, lower] * (1.0 - frac_row) + data[:, upper] * frac_row
        return resized.astype(np.float32, copy=False)

    if axis == 0:
        frac_col = frac[:, np.newaxis]
        resized = data[lower, :] * (1.0 - frac_col) + data[upper, :] * frac_col
        return resized.astype(np.float32, copy=False)

    raise ValueError(f"Only 2D arrays are supported, got axis={axis}")


def resize_spectrum(spectrum: np.ndarray, target_shape: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize ``[frequency, time]`` spectrum to ``target_shape``."""
    if len(target_shape) != 2:
        raise ValueError(f"target_shape must be two integers, got {target_shape}")

    data = np.asarray(spectrum, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D spectrum, got shape {data.shape}")

    target_freq, target_time = int(target_shape[0]), int(target_shape[1])
    data = _resize_axis(data, target_time, axis=1)
    data = _resize_axis(data, target_freq, axis=0)
    return data.astype(np.float32)


def preprocess_array(spectrum: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Run the full preprocessing pipeline and return a ``[1, H, W]`` tensor."""
    prep_cfg = config["preprocessing"]
    target_shape = tuple(config["data"]["target_shape"])

    data = clean_invalid_values(spectrum)
    data = subtract_background(data, method=prep_cfg["background_method"])
    data = normalize_spectrum(data, prep_cfg)
    data = resize_spectrum(data, target_shape=target_shape)
    return data[np.newaxis, :, :].astype(np.float32)


def preprocess_file(
    file_path: str | Path,
    output_path: str | Path | None,
    config: dict[str, Any],
    label_id: int | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess one FITS file and optionally save it as ``.npz``."""
    file_path = Path(file_path)
    spectrum, metadata = read_fits_spectrum(file_path)
    tensor = preprocess_array(spectrum, config)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return tensor, metadata

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_fn = np.savez_compressed if config["preprocessing"].get("save_compressed", False) else np.savez
        save_fn(
            output_path,
            spectrum=tensor,
            label_id=-1 if label_id is None else int(label_id),
            source_file=str(file_path),
            metadata_json=json.dumps(metadata),
        )

    return tensor, metadata


# Per-worker globals set once by the pool initializer. This avoids pickling and
# re-sending the (large) config dict with every one of the 500k+ tasks, which was
# the dominant overhead of the previous ``submit``-per-row approach.
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_OVERWRITE: bool = False


def _init_preprocess_worker(config: dict[str, Any], should_overwrite: bool) -> None:
    global _WORKER_CONFIG, _WORKER_OVERWRITE
    _WORKER_CONFIG = config
    _WORKER_OVERWRITE = bool(should_overwrite)


def _preprocess_row_worker(row: dict[str, str]) -> tuple[str, str, str, str | None]:
    config = _WORKER_CONFIG
    should_overwrite = _WORKER_OVERWRITE
    output_path = Path(row["processed_path"])
    if output_path.exists() and not should_overwrite:
        return "skipped", row["file_path"], row["processed_path"], None

    try:
        preprocess_file(
            row["file_path"],
            output_path,
            config,
            label_id=int(row["label_id"]),
            overwrite=bool(should_overwrite),
        )
    except Exception as exc:
        return "failed", row["file_path"], row["processed_path"], repr(exc)

    return "processed", row["file_path"], row["processed_path"], None


def _resolve_num_workers(value: Any) -> int:
    """Resolve a configured worker count, supporting ``'auto'`` and negatives.

    ``'auto'`` / ``-1`` -> all logical CPUs. Anything else is the integer count
    (0 or 1 means run serially in the calling process).
    """
    import os

    if isinstance(value, str) and value.strip().lower() == "auto":
        return os.cpu_count() or 1
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    if count < 0:
        return os.cpu_count() or 1
    return count


def _failure_report_path(config: dict[str, Any]) -> Path:
    reports_dir = Path(config["paths"].get("reports_dir", "outputs/reports"))
    return reports_dir / "preprocessing_failures.csv"


def _write_preprocessing_failures(
    failures: list[dict[str, str]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "processed_path", "label", "split", "error"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def preprocess_manifest(
    config: dict[str, Any],
    limit: int | None = None,
    overwrite: bool | None = None,
) -> dict[str, int]:
    """Preprocess every row in ``manifest.csv``."""
    manifest_path = config["paths"]["manifest_path"]
    rows = read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    should_overwrite = config["preprocessing"]["overwrite"] if overwrite is None else overwrite
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    requested_device = select_preprocessing_device(config)
    num_workers = _resolve_num_workers(config["preprocessing"].get("num_workers", 0))

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    LOGGER.info(
        "Preprocessing numeric backend: NumPy/CPU (requested device=%s; I/O-bound, "
        "GPU is used for training). Workers=%d",
        requested_device,
        num_workers,
    )

    if num_workers > 1:
        # One initializer call per worker process sends the config once; rows are
        # streamed in chunks to amortize inter-process overhead across many files.
        chunksize = max(1, min(128, len(rows) // (num_workers * 8) or 1))
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_preprocess_worker,
            initargs=(config, bool(should_overwrite)),
        ) as executor:
            results = executor.map(_preprocess_row_worker, rows, chunksize=chunksize)
            for status, file_path, processed_path, error in tqdm(
                results, total=len(rows), desc="Preprocessing FITS files"
            ):
                counts[status] += 1
                if status == "failed":
                    LOGGER.error("Failed to preprocess %s: %s", file_path, error)
                    failures.append(
                        {
                            "file_path": file_path,
                            "processed_path": processed_path,
                            "label": "",
                            "split": "",
                            "error": str(error),
                        }
                    )
    else:
        for row in tqdm(rows, desc="Preprocessing FITS files"):
            output_path = Path(row["processed_path"])
            if output_path.exists() and not should_overwrite:
                counts["skipped"] += 1
                continue

            try:
                preprocess_file(
                    row["file_path"],
                    output_path,
                    config,
                    label_id=int(row["label_id"]),
                    overwrite=bool(should_overwrite),
                )
                counts["processed"] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append(
                    {
                        "file_path": row["file_path"],
                        "processed_path": row["processed_path"],
                        "label": row.get("label", ""),
                        "split": row.get("split", ""),
                        "error": repr(exc),
                    }
                )
                LOGGER.exception("Failed to preprocess %s: %s", row["file_path"], exc)

    if failures:
        report_path = _failure_report_path(config)
        _write_preprocessing_failures(failures, report_path)
        LOGGER.warning("Wrote %d preprocessing failures to %s", len(failures), report_path)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess e-CALLISTO FITS files")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing .npz tensors")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Recorded for logging; numeric preprocessing runs in NumPy")
    parser.add_argument("--num-workers", default=None, help="CPU preprocessing workers ('auto' = all cores, or an integer)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config["preprocessing"]["device"] = args.device
    if args.num_workers is not None:
        config["preprocessing"]["num_workers"] = args.num_workers
    counts = preprocess_manifest(config, limit=args.limit, overwrite=args.overwrite)
    LOGGER.info("Preprocessing complete: %s", counts)


if __name__ == "__main__":
    main()
