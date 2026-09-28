"""Build and validate the dataset manifest."""

# NOTE: Vendored from H:\Burst Identifier (src/data/manifest.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from callisto_trainer.core.fits_reader import parse_filename_fallback, read_fits_metadata
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)
MANIFEST_COLUMNS = [
    "file_path",
    "processed_path",
    "label",
    "label_id",
    "station",
    "date",
    "start_time",
    "freq_min_mhz",
    "freq_max_mhz",
    "n_freq",
    "n_time",
    "split",
]


def parse_filename_metadata(path: str | Path) -> dict[str, Any]:
    """Public filename parser used by manifest generation and tests."""
    return parse_filename_fallback(path)


def _class_dirs_from_config(config: dict[str, Any]) -> dict[str, Path]:
    classes = config["data"]["classes"]
    for candidate in config["paths"]["raw_dir_candidates"]:
        base = Path(candidate)
        class_dirs = {label: base / label for label in classes}
        if all(path.exists() for path in class_dirs.values()):
            if any(next(path.glob("*.fit.gz"), None) is not None for path in class_dirs.values()):
                return class_dirs

    raise FileNotFoundError(
        "Could not find raw class folders. Expected Burst/ and No_Burst/ either "
        "at the workspace root or under data/raw/."
    )


def _processed_path_for(file_path: Path, label: str, processed_dir: str | Path) -> Path:
    file_name = file_path.name
    if file_name.endswith(".fit.gz"):
        npz_name = file_name[:-7] + ".npz"
    else:
        npz_name = file_path.stem + ".npz"
    return Path(processed_dir) / label / npz_name


def _metadata_for_file(path: Path, read_headers: bool = True) -> dict[str, Any]:
    metadata = parse_filename_metadata(path)
    if not read_headers:
        return metadata

    try:
        header_metadata = read_fits_metadata(path)
    except Exception as exc:
        LOGGER.warning("Falling back to filename metadata for %s: %s", path, exc)
        return metadata

    for key, value in header_metadata.items():
        if value is not None and value != "":
            metadata[key] = value
    return metadata


def _build_row(
    file_path: str | Path,
    label: str,
    label_id: int,
    processed_dir: str | Path,
    read_headers: bool,
) -> dict[str, Any]:
    """Build a single manifest row (reads FITS header metadata when enabled)."""
    file_path = Path(file_path)
    metadata = _metadata_for_file(file_path, read_headers=read_headers)
    return {
        "file_path": str(file_path),
        "processed_path": str(_processed_path_for(file_path, label, processed_dir)),
        "label": label,
        "label_id": int(label_id),
        "station": metadata.get("station"),
        "date": metadata.get("date"),
        "start_time": metadata.get("start_time"),
        "freq_min_mhz": metadata.get("freq_min_mhz"),
        "freq_max_mhz": metadata.get("freq_max_mhz"),
        "n_freq": metadata.get("n_freq"),
        "n_time": metadata.get("n_time"),
        "split": "",
    }


# Per-worker globals set once by the pool initializer so the (constant)
# processed_dir and header flag are not re-pickled with every one of the ~587k
# tasks. Header reads are I/O + gzip-decompress bound, so they parallelize well.
_WORKER_PROCESSED_DIR: str | None = None
_WORKER_READ_HEADERS: bool = True


def _init_manifest_worker(processed_dir: str, read_headers: bool) -> None:
    global _WORKER_PROCESSED_DIR, _WORKER_READ_HEADERS
    _WORKER_PROCESSED_DIR = processed_dir
    _WORKER_READ_HEADERS = bool(read_headers)


def _manifest_row_worker(task: tuple[str, str, int]) -> dict[str, Any]:
    file_path, label, label_id = task
    return _build_row(file_path, label, label_id, _WORKER_PROCESSED_DIR, _WORKER_READ_HEADERS)


def _resolve_workers(value: Any) -> int:
    """Resolve a worker count, supporting ``'auto'`` and negatives.

    ``'auto'`` / negative -> all logical CPUs. Otherwise the integer value
    (0 or 1 runs serially in the calling process).
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return os.cpu_count() or 1
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    if count < 0:
        return os.cpu_count() or 1
    return count


def event_key_for_path(path: str | Path) -> tuple[str, ...]:
    """Return a station-agnostic event key for a raw FITS filename.

    e-CALLISTO names files ``STATION_DATE_STARTtime_ENDtime.fit.gz``. A single
    solar burst is recorded by many stations at the same timestamp, so the key
    deliberately drops the station and keys on ``(date, start, end)``. All
    recordings of one event therefore share a key and can be kept in the same
    split to prevent train/test leakage. Filenames that do not match the
    expected pattern fall back to their own unique key (one group per file).
    """
    name = Path(path).name
    if name.endswith(".fit.gz"):
        stem = name[:-7]
    else:
        stem = Path(name).stem

    parts = stem.split("_")
    if len(parts) >= 4:
        return (parts[-3], parts[-2], parts[-1])
    return (stem,)


def assign_stratified_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_by_event: bool = True,
) -> list[dict[str, Any]]:
    """Assign train/val/test split while preserving class balance.

    When ``group_by_event`` is true (the default), every recording of the same
    solar event is assigned to the same split. This prevents data leakage where
    sibling recordings of one burst captured by different stations land in both
    train and test and inflate the reported score. Splitting is done over whole
    event groups, stratified by label, so class balance is preserved.

    Set ``group_by_event=False`` to reproduce the older per-file split.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    rng = random.Random(seed)

    # Build one group per event (or per file when grouping is disabled).
    groups: dict[tuple[str, ...], list[int]] = {}
    group_label: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(rows):
        key = event_key_for_path(row["file_path"]) if group_by_event else (str(index),)
        groups.setdefault(key, []).append(index)
        group_label.setdefault(key, int(row["label_id"]))

    # Stratify the groups by label, then split groups (not files) per label.
    keys_by_label: dict[int, list[tuple[str, ...]]] = {}
    for key, label_id in group_label.items():
        keys_by_label.setdefault(label_id, []).append(key)

    for label_id in sorted(keys_by_label):
        shuffled = sorted(keys_by_label[label_id])  # deterministic base order
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        train_keys = set(shuffled[:n_train])
        val_keys = set(shuffled[n_train : n_train + n_val])

        for key in shuffled:
            if key in train_keys:
                split = "train"
            elif key in val_keys:
                split = "val"
            else:
                split = "test"
            for index in groups[key]:
                rows[index]["split"] = split

    return rows


def build_manifest(
    config: dict[str, Any],
    read_headers: bool = True,
    num_workers: Any | None = None,
) -> list[dict[str, Any]]:
    """Scan raw FITS folders and return manifest rows.

    The per-file FITS header read is the slow part (gzip + astropy over ~500k
    files). It is parallelized across CPU cores by default; pass ``num_workers``
    to override, otherwise ``preprocessing.num_workers`` from the config is used
    (``'auto'`` = all cores). Row order is deterministic regardless of workers.
    """
    class_dirs = _class_dirs_from_config(config)
    classes = config["data"]["classes"]
    processed_dir = config["paths"]["processed_dir"]
    headers_enabled = read_headers

    if read_headers:
        try:
            import astropy.io.fits  # noqa: F401
        except ImportError:
            LOGGER.warning(
                "Astropy is not installed; manifest will use filename metadata only. "
                "Install dependencies with: pip install -r requirements.txt"
            )
            headers_enabled = False

    # Gather tasks in deterministic (label order, sorted filename) order.
    tasks: list[tuple[str, str, int]] = []
    for label, label_id in sorted(classes.items(), key=lambda item: item[1]):
        class_dir = class_dirs[label]
        files = sorted(class_dir.glob("*.fit.gz"))
        LOGGER.info("Found %d files for %s in %s", len(files), label, class_dir)
        for file_path in files:
            tasks.append((str(file_path), label, int(label_id)))

    workers_value = config.get("preprocessing", {}).get("num_workers", 0) if num_workers is None else num_workers
    workers = _resolve_workers(workers_value)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    if headers_enabled and workers > 1 and len(tasks) > 1:
        LOGGER.info("Reading FITS headers with %d parallel workers", workers)
        chunksize = max(1, min(256, len(tasks) // (workers * 8) or 1))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_manifest_worker,
            initargs=(str(processed_dir), headers_enabled),
        ) as executor:
            rows = list(
                tqdm(
                    executor.map(_manifest_row_worker, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Reading FITS metadata",
                )
            )
    else:
        rows = [
            _build_row(file_path, label, label_id, processed_dir, headers_enabled)
            for file_path, label, label_id in tqdm(tasks, desc="Reading FITS metadata")
        ]

    split_cfg = config["data"]["split"]
    return assign_stratified_split(
        rows,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
        group_by_event=bool(split_cfg.get("group_by_event", True)),
    )


def write_manifest(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write manifest rows to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    """Read a manifest CSV into dictionaries."""
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    """Validate required columns and return class/split summaries."""
    missing_columns = set(MANIFEST_COLUMNS) - set(rows[0].keys() if rows else [])
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    label_split_counts = Counter((row["label"], row["split"]) for row in rows)

    missing_files = [row["file_path"] for row in rows if not Path(row["file_path"]).exists()]
    if missing_files:
        examples = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"Manifest references missing raw files: {examples}")

    return {
        "labels": label_counts,
        "splits": split_counts,
        "label_splits": label_split_counts,
    }


def count_event_leakage(rows: list[dict[str, Any]]) -> int:
    """Return how many event groups have files in more than one split.

    A healthy event-grouped manifest returns 0. Any positive number means the
    same solar event appears in multiple splits, which leaks information from
    train into val/test.
    """
    splits_by_event: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
        key = event_key_for_path(row["file_path"])
        splits_by_event.setdefault(key, set()).add(row.get("split", ""))
    return sum(1 for splits in splits_by_event.values() if len(splits) > 1)


def validate_expected_counts(
    summary: dict[str, Counter],
    expected_counts: dict[str, Any] | None,
    strict: bool = False,
) -> bool:
    """Check optional expected class counts from config.

    Returns ``True`` when counts match (or none were configured). On a mismatch:
    a warning is logged and ``False`` returned by default, so a stale config
    value can never discard an already-built manifest. Set ``strict=True`` to
    raise instead (callers should still write the manifest *before* calling this
    so the expensive scan is never lost).
    """
    if not expected_counts:
        return True

    actual_counts = summary["labels"]
    mismatches: list[str] = []
    for label, expected_value in expected_counts.items():
        expected_count = int(expected_value)
        actual_count = int(actual_counts.get(label, 0))
        if actual_count != expected_count:
            mismatches.append(f"{label}: expected {expected_count}, found {actual_count}")

    if not mismatches:
        return True

    message = (
        "Raw dataset counts do not match config data.expected_counts. "
        + "; ".join(mismatches)
        + ". Update data.expected_counts to your actual file counts if this is expected."
    )
    if strict:
        raise ValueError(message)
    LOGGER.warning(message)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build e-CALLISTO burst manifest.csv")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--no-header-metadata",
        action="store_true",
        help="Skip FITS header reads and use filename metadata only",
    )
    parser.add_argument(
        "--num-workers",
        default=None,
        help="Parallel FITS-header workers ('auto' = all cores, or an integer). "
        "Defaults to preprocessing.num_workers from the config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rows = build_manifest(
        config,
        read_headers=not args.no_header_metadata,
        num_workers=args.num_workers,
    )
    summary = validate_manifest(rows)

    # Write the manifest FIRST so the (multi-hour) scan is persisted and can
    # never be discarded by a downstream check. Only then run the optional
    # expected-counts sanity check (a warning by default).
    manifest_path = config["paths"]["manifest_path"]
    write_manifest(rows, manifest_path)
    LOGGER.info("Wrote %s rows to %s", len(rows), manifest_path)

    data_cfg = config.get("data", {})
    counts_ok = validate_expected_counts(
        summary,
        data_cfg.get("expected_counts"),
        strict=bool(data_cfg.get("strict_expected_counts", False)),
    )

    leakage = count_event_leakage(rows)
    LOGGER.info("Label counts: %s", dict(summary["labels"]))
    LOGGER.info("Split counts: %s", dict(summary["splits"]))
    LOGGER.info("Label/split counts: %s", dict(summary["label_splits"]))
    if not counts_ok:
        LOGGER.warning(
            "Manifest written despite count mismatch above; review data.expected_counts."
        )
    if leakage:
        LOGGER.warning("Event leakage: %d events span multiple splits", leakage)
    else:
        LOGGER.info("Event leakage check passed: no event spans multiple splits")


if __name__ == "__main__":
    main()
