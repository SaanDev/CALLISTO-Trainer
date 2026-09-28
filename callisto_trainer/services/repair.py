"""Re-derive stored values after a reader correction.

Some things in the database are *derived* from the FITS files rather than entered
by the operator: a file's frequency range and cadence, and every box's physical
bounds and measured drift rate. When the reader improves, those stored values are
stale, and nothing else in the app will notice.

The case this was written for: the axis table was located by ``EXTNAME='AXES'``,
but a large part of the archive writes the identical table unnamed. Those files
fell back to the header's ``CRVAL2``/``CDELT2`` placeholders, which yield a
"frequency axis" that is simply the channel index -- 1..200 for a 200-channel
receiver. Every drift rate measured against that axis was wrong, because MHz/s
depends directly on what a row is worth in MHz.

Annotations are never touched: boxes keep their pixels, their type and their
verdict. Only the values computed *from* the file are recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from callisto_trainer.core.coords import SpectrumAxes, box_to_physical
from callisto_trainer.core.crops import normalize_full_spectrum
from callisto_trainer.core.fits_reader import read_fits_metadata, read_fits_spectrum_and_axes
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.physics_service import measure_box, physics_to_columns
from callisto_trainer.store.repository import AnnotationRepository

LOGGER = get_logger(__name__)


@dataclass
class RepairResult:
    """What a repair pass changed."""

    files_checked: int = 0
    files_updated: int = 0
    files_failed: int = 0
    axis_source_fixed: int = 0
    boxes_remeasured: int = 0
    boxes_failed: int = 0
    examples: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.files_checked:,} file(s) checked"]
        if self.axis_source_fixed:
            parts.append(f"{self.axis_source_fixed:,} frequency axes corrected")
        if self.files_updated:
            parts.append(f"{self.files_updated:,} updated")
        if self.boxes_remeasured:
            parts.append(f"{self.boxes_remeasured:,} burst(s) re-measured")
        if self.files_failed:
            parts.append(f"{self.files_failed:,} unreadable")
        return ", ".join(parts)


def _changed(record: Any, metadata: dict[str, Any]) -> bool:
    """Whether re-reading produced different axis-derived values."""
    def differs(stored: float | None, fresh: float | None) -> bool:
        if stored is None or fresh is None:
            return stored is not fresh
        return abs(float(stored) - float(fresh)) > 1e-6

    return (
        record.freq_axis_source != metadata.get("freq_axis_source")
        or differs(record.freq_min_mhz, metadata.get("freq_min_mhz"))
        or differs(record.freq_max_mhz, metadata.get("freq_max_mhz"))
        or differs(record.cadence_s, metadata.get("cadence_s"))
    )


def refresh_axes_and_physics(
    repository: AnnotationRepository,
    only_changed: bool = True,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> RepairResult:
    """Re-read every file's axes and re-measure any boxes whose axis changed.

    Header-only reads are cheap, so every file is checked. The expensive full
    read plus physics measurement happens only for files that actually have
    boxes *and* whose axis moved -- with ``only_changed`` off, for every file
    with boxes.
    """
    result = RepairResult()
    records = repository.all_files()
    total = len(records)

    for index, record in enumerate(records):
        if progress is not None and progress(index, total, record.file_name) is False:
            LOGGER.info("Repair cancelled after %d files", index)
            break

        result.files_checked += 1
        try:
            metadata = read_fits_metadata(record.path)
        except Exception as exc:
            result.files_failed += 1
            result.errors.append((record.path, repr(exc)))
            continue

        axis_fixed = (
            record.freq_axis_source != "axes_table"
            and metadata.get("freq_axis_source") == "axes_table"
        )
        needs_update = _changed(record, metadata)

        if needs_update:
            repository.update_file_axes(record.id, metadata)
            result.files_updated += 1
            if axis_fixed:
                result.axis_source_fixed += 1
                if len(result.examples) < 8:
                    result.examples.append(
                        f"{record.file_name}: "
                        f"{record.freq_min_mhz:.0f}-{record.freq_max_mhz:.0f} -> "
                        f"{metadata['freq_min_mhz']:.2f}-{metadata['freq_max_mhz']:.2f} MHz"
                    )

        boxes = repository.boxes_for_file(record.id) if record.box_count else []
        if not boxes or (only_changed and not needs_update):
            continue

        # The drift rate is MHz per second, so it is only meaningful once the row
        # -> MHz mapping is right. Re-measure from the corrected axis.
        try:
            spectrum, full_metadata = read_fits_spectrum_and_axes(record.path)
            normalized = normalize_full_spectrum(spectrum, repository_config())
            axes = SpectrumAxes.from_metadata(full_metadata)
        except Exception as exc:
            result.boxes_failed += len(boxes)
            result.errors.append((record.path, repr(exc)))
            continue

        for box in boxes:
            try:
                repository.update_box_physical(
                    box.id, box_to_physical(axes, box.row0, box.row1, box.col0, box.col1)
                )
                repository.set_box_physics(
                    box.id, physics_to_columns(measure_box(normalized, axes, box))
                )
                result.boxes_remeasured += 1
            except Exception as exc:
                result.boxes_failed += 1
                result.errors.append((record.path, repr(exc)))

    LOGGER.info("Repair pass: %s", result.summary())
    return result


def repository_config() -> dict[str, Any]:
    """The pipeline config used for normalization during a repair pass."""
    from callisto_trainer.core.config import load_config

    return load_config()
