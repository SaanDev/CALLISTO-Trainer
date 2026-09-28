"""The stored burst parameters of drawn boxes, in one place.

The Label tab calculates a box's parameters the moment it is drawn, resized or
retyped; the backfill and the axis repair recalculate stored ones. All of them
come through :func:`measure_box`, which uses the box geometry for Type II and
Type III (``burst_physics.box_parameters``). The model's own features are a
separate, per-region pixel measurement made by the exporter and the predictor.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from callisto_trainer.core.burst_physics import BurstPhysics, box_parameters
from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.crops import normalize_full_spectrum
from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.store.db import PHYSICS_COLUMNS
from callisto_trainer.store.repository import AnnotationRepository, BoxRecord

LOGGER = get_logger(__name__)

_STORED = {name for name, _type in PHYSICS_COLUMNS}


def physics_to_columns(physics: BurstPhysics) -> dict[str, Any]:
    """Map a measurement onto the database column names."""
    payload = asdict(physics)
    payload["physics_confidence"] = payload.pop("confidence", "none")
    payload["edge_clipped"] = int(bool(payload.get("edge_clipped")))
    return {key: value for key, value in payload.items() if key in _STORED}


def measure_box(
    normalized: np.ndarray, axes: SpectrumAxes, box: BoxRecord
) -> BurstPhysics:
    """A drawn box's burst parameters, for its current type."""
    return box_parameters(
        axes, box.row0, box.row1, box.col0, box.col1, box.burst_type, normalized=normalized
    )


def measure_and_store(
    repository: AnnotationRepository,
    normalized: np.ndarray,
    axes: SpectrumAxes,
    box: BoxRecord,
) -> BurstPhysics:
    """Measure one box and persist the result."""
    physics = measure_box(normalized, axes, box)
    repository.set_box_physics(box.id, physics_to_columns(physics))
    return physics


def backfill_physics(
    repository: AnnotationRepository,
    only_missing: bool = True,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> dict[str, int]:
    """Measure every stored box, reading each source file exactly once.

    Used after upgrading a database annotated before physics existed, and after
    changing the measurement so older boxes do not keep stale numbers: with
    ``only_missing`` it recalculates every box without parameters or with
    parameters from an earlier method.
    """
    from callisto_trainer.core.burst_physics import box_parameters_current

    missing = None
    if only_missing:
        missing = set(repository.boxes_missing_physics())
        missing |= {
            box.id for _, box in repository.iter_training_boxes()
            if not box_parameters_current(box.physics, box.burst_type)
        }

    grouped: dict[int, tuple[Any, list[BoxRecord]]] = {}
    # Every burst box that can reach a training set.
    for record, box in repository.iter_training_boxes():
        if missing is not None and box.id not in missing:
            continue
        grouped.setdefault(record.id, (record, []))[1].append(box)

    counts = {"measured": 0, "unmeasurable": 0, "failed": 0, "files": len(grouped)}
    for index, (record, boxes) in enumerate(grouped.values()):
        if progress is not None and progress(index, len(grouped), record.file_name) is False:
            break
        try:
            spectrum, metadata = read_fits_spectrum_and_axes(record.path)
            from callisto_trainer.core.config import load_config

            normalized = normalize_full_spectrum(spectrum, load_config())
            axes = SpectrumAxes.from_metadata(metadata)
        except Exception as exc:
            counts["failed"] += len(boxes)
            LOGGER.warning("Could not measure boxes in %s: %r", record.path, exc)
            continue

        for box in boxes:
            physics = measure_and_store(repository, normalized, axes, box)
            counts["measured" if physics.measured else "unmeasurable"] += 1

    LOGGER.info("Physics backfill: %s", counts)
    return counts
