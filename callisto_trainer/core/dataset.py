"""PyTorch dataset for preprocessed e-CALLISTO burst tensors."""

# NOTE: Vendored from H:\Burst Identifier (src/data/dataset.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from callisto_trainer.core.augmentations import SpectrumAugmenter
from callisto_trainer.core.metadata_features import (
    DEFAULT_MIN_STATION_FILES,
    StationDateEncoder,
    build_station_vocab,
    row_to_meta_vector,
)
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _read_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CallistoBurstDataset(Dataset):
    """Dataset that reads preprocessed ``.npz`` tensors listed in a manifest.

    When ``return_metadata_features`` is true, each item also yields a float32
    metadata vector ``[station_idx, numeric...]`` built from the manifest row
    (used by the multi-input model).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        return_metadata: bool = False,
        transform: Any | None = None,
        drop_missing_processed: bool = True,
        metadata_vocab: dict[str, int] | None = None,
        return_metadata_features: bool = False,
        return_physics_features: bool = False,
        rows: list[dict[str, str]] | None = None,
        feature_set: str | None = None,
        station_date: StationDateEncoder | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.return_metadata = return_metadata
        self.transform = transform
        self.drop_missing_processed = drop_missing_processed
        self.metadata_vocab = metadata_vocab or {}
        self.return_metadata_features = return_metadata_features
        self.return_physics_features = return_physics_features
        # Trainer addition: which feature vector the physics branch takes. The
        # original eight physics features are rebuilt from manifest columns;
        # larger sets are read verbatim from each sample's .npz.
        self.feature_set = feature_set or "physics_v1"
        # ``rows`` lets the caller pass split rows that were already read and
        # existence-filtered once (see ``get_dataloaders``), avoiding a full
        # re-parse of the manifest and a second existence scan per split.
        self.rows = rows if rows is not None else self._load_rows()

        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {self.manifest_path}")

        # Trainer addition: each file's station + date vector, appended after the
        # region features for a model with a station/date correction. Built from
        # the manifest once, not per sample per epoch.
        self.station_date_vectors: np.ndarray | None = None
        if station_date is not None and return_physics_features:
            self.station_date_vectors = np.stack(
                [station_date.vector_for(row) for row in self.rows]
            )

    def _load_rows(self) -> list[dict[str, str]]:
        rows = _read_manifest_rows(self.manifest_path)
        split_rows = [row for row in rows if row["split"] == self.split]
        if not self.drop_missing_processed:
            return split_rows

        available_rows = [row for row in split_rows if Path(row["processed_path"]).exists()]
        missing_count = len(split_rows) - len(available_rows)
        if missing_count:
            first_missing = next(
                row["processed_path"]
                for row in split_rows
                if not Path(row["processed_path"]).exists()
            )
            LOGGER.warning(
                "Skipping %d %s rows because preprocessed tensors are missing. First missing: %s",
                missing_count,
                self.split,
                first_missing,
            )
        return available_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        processed_path = Path(row["processed_path"])
        # No pre-load exists() check here on purpose: it would add one filesystem
        # stat() per sample per epoch (~500k/epoch). np.load already raises
        # FileNotFoundError for a genuinely missing file; we add the run hint.
        stored_features = None
        try:
            with np.load(processed_path, allow_pickle=False) as loaded:
                spectrum = loaded["spectrum"].astype(np.float32)
                if self.return_physics_features and self.feature_set != "physics_v1":
                    stored_features = loaded["features"].astype(np.float32)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing preprocessed tensor: {processed_path}. "
                "Run python -m callisto_trainer.core.preprocess first."
            ) from exc

        tensor = torch.from_numpy(spectrum).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        label = torch.tensor(float(row["label_id"]), dtype=torch.float32)

        if self.return_physics_features:
            if stored_features is not None:
                if self.station_date_vectors is not None:
                    stored_features = np.concatenate(
                        [stored_features, self.station_date_vectors[index]]
                    )
                return tensor, label, torch.from_numpy(stored_features).float()
            # Trainer addition: measured drift rate and burst extent, read from
            # the manifest columns the exporter wrote.
            from callisto_trainer.core.burst_physics import (
                physics_from_row,
                physics_to_vector,
            )

            physics = torch.from_numpy(physics_to_vector(physics_from_row(row))).float()
            return tensor, label, physics
        if self.return_metadata_features:
            meta = torch.from_numpy(row_to_meta_vector(row, self.metadata_vocab)).float()
            return tensor, label, meta
        if self.return_metadata:
            return tensor, label, row
        return tensor, label


def _auto_worker_count() -> int:
    """A safe automatic DataLoader worker count.

    Capped deliberately: under Windows ``spawn`` (and macOS) every worker is a
    fresh process that re-imports torch and receives a pickled copy of the split
    row list, so "all logical cores" can exhaust RAM and stall before the first
    batch reaches the GPU. The cap keeps per-worker memory and spawn time sane;
    raise ``training.num_workers`` explicitly if your machine has headroom.
    """
    import os
    import sys

    cpu = os.cpu_count() or 1
    cap = 4 if sys.platform.startswith("win") else 8
    return max(1, min(cap, cpu))


def _resolve_num_workers(value: Any) -> int:
    """Resolve a DataLoader worker count, supporting ``'auto'`` and negatives.

    ``'auto'`` / negative -> a safe, capped automatic count (see
    ``_auto_worker_count``). Any explicit non-negative integer is used as-is.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return _auto_worker_count()
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    if count < 0:
        return _auto_worker_count()
    return count


def _filter_existing(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    """Drop rows whose preprocessed tensor is missing (single pass per split)."""
    LOGGER.info(
        "Checking %d %s tensors exist (set data.assume_processed_complete: true "
        "to skip this once preprocessing is verified complete)...",
        len(rows),
        split,
    )
    available = [row for row in rows if Path(row["processed_path"]).exists()]
    missing = len(rows) - len(available)
    if missing:
        first_missing = next(
            row["processed_path"] for row in rows if not Path(row["processed_path"]).exists()
        )
        LOGGER.warning(
            "Skipping %d %s rows because preprocessed tensors are missing. First missing: %s",
            missing,
            split,
            first_missing,
        )
    return available


def _station_date_encoder(
    config: dict[str, Any], train_rows: list[dict[str, str]]
) -> StationDateEncoder | None:
    """The station/date encoder of ``model.station_date``, fitted if it is new.

    Fitted on the training split only, and written back into the config so it
    is saved with every checkpoint and reused at inference. A config that
    already carries a station list (resuming, or evaluating a checkpoint) keeps
    it: refitting would renumber the stations under trained embeddings.
    """
    section = config.get("model", {}).get("station_date") or {}
    if not bool(section.get("enabled", False)):
        return None
    if section.get("station_vocab"):
        encoder = StationDateEncoder.from_config(section)
    else:
        encoder = StationDateEncoder.fit(
            train_rows,
            min_station_files=int(section.get("min_station_files", DEFAULT_MIN_STATION_FILES)),
        )
        section.update(encoder.to_config())
        config["model"]["station_date"] = section
    LOGGER.info(
        "Station/date correction (cap %.2f): %d station(s) with their own index, "
        "dates %.2f-%.2f",
        float(section.get("cap", 1.0)),
        len(encoder.vocab),
        encoder.year_min,
        encoder.year_max,
    )
    return encoder


def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Create train/validation/test DataLoaders from the manifest.

    When ``model.use_metadata`` is set, a station vocabulary is built from the
    full manifest and stored back into ``config['model']['station_vocab']`` so it
    is saved in the checkpoint and reused at inference.

    The manifest is read exactly once and grouped by split here (instead of once
    per split inside each dataset). Existence filtering is also done once; set
    ``data.assume_processed_complete: true`` to skip it entirely after a clean
    preprocessing run, which removes ~500k filesystem stat() calls at startup.
    """
    manifest_path = config["paths"]["manifest_path"]
    batch_size = int(config["training"]["batch_size"])
    num_workers = _resolve_num_workers(config["training"].get("num_workers", 0))
    data_cfg = config.get("data", {})
    drop_missing_processed = bool(data_cfg.get("drop_missing_processed", True))
    assume_complete = bool(data_cfg.get("assume_processed_complete", False))
    use_cuda = torch.cuda.is_available()
    pin_memory = bool(config["training"].get("pin_memory", True)) and use_cuda
    persistent_workers = bool(config["training"].get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(config["training"].get("prefetch_factor", 2))
    LOGGER.info(
        "DataLoader config: num_workers=%d (requested %r), batch_size=%d, "
        "prefetch_factor=%d, pin_memory=%s, persistent_workers=%s",
        num_workers,
        config["training"].get("num_workers", 0),
        batch_size,
        prefetch_factor,
        pin_memory,
        persistent_workers,
    )

    # Read + parse the manifest a single time, then bucket rows by split.
    all_rows = _read_manifest_rows(manifest_path)
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in all_rows:
        bucket = rows_by_split.get(row["split"])
        if bucket is not None:
            bucket.append(row)

    use_metadata = bool(config.get("model", {}).get("use_metadata", False))
    use_physics = bool(config.get("model", {}).get("use_physics", False))
    feature_set = config.get("model", {}).get("feature_set")
    metadata_vocab: dict[str, int] | None = None
    if use_metadata:
        metadata_vocab = config["model"].get("station_vocab")
        if not metadata_vocab:
            metadata_vocab = build_station_vocab(all_rows)
            config["model"]["station_vocab"] = metadata_vocab
        LOGGER.info("Metadata conditioning enabled: %d known stations", len(metadata_vocab))

    station_date = _station_date_encoder(config, rows_by_split["train"]) if use_physics else None

    train_transform = SpectrumAugmenter(config) if config.get("augmentation", {}).get("enabled", False) else None

    def _rows_for(split: str) -> list[dict[str, str]]:
        split_rows = rows_by_split[split]
        if drop_missing_processed and not assume_complete:
            return _filter_existing(split_rows, split)
        return split_rows

    def _make(split: str, transform: Any | None) -> CallistoBurstDataset:
        return CallistoBurstDataset(
            manifest_path,
            split=split,
            transform=transform,
            drop_missing_processed=drop_missing_processed,
            metadata_vocab=metadata_vocab,
            return_metadata_features=use_metadata,
            return_physics_features=use_physics,
            rows=_rows_for(split),
            feature_set=feature_set,
            station_date=station_date,
        )

    datasets = {
        "train": _make("train", train_transform),
        "val": _make("val", None),
        "test": _make("test", None),
    }

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
    }
