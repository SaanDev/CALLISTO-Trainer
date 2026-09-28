"""Turn annotations into immutable, trainable dataset snapshots.

Three tracks come out of the same store:

* **unified** (the default) -- one ``.npz`` per *region*, over up to seven
  classes: No_Burst, RFI, Type II, Type III, Type IIIG, Type IV, Other.
  Positives are the drawn burst boxes plus finder regions inside them; negatives
  are the regions the same finder proposes outside them and in no-burst files,
  plus synthetic interference. Negatives that measure like interference are
  labelled RFI automatically (see ``core/rfi_labels.py``); nothing is drawn as
  RFI. Each sample carries three image views (the exact crop, a wide context
  strip, and that strip on a quiet-part background so long continua stay
  visible) and a vector of region features, so the model can see what
  separates interference from a burst. A snapshot also lists every
  file with its verdict (``files.csv``) so a model can be judged file by file --
  which is how it is used -- not just crop by crop.
* **type** -- one ``.npz`` per confirmed burst box, over the burst types only.
  The older stage-2 track, kept so existing checkpoints remain reproducible.
* **binary** -- one ``.npz`` per file, labelled from the operator's burst /
  no-burst verdict. Identical in form to the original Burst Identifier dataset,
  so the vendored binary trainer runs on it unchanged.

Each export is written to a timestamped directory and never mutated afterwards,
so a checkpoint can always be traced back to the exact data that produced it.

## Split integrity

Splitting reuses the upstream :func:`assign_stratified_split` with
``group_by_event=True``. Because the group key comes from the *source filename*,
every crop from one file -- and every station's recording of one solar event --
lands in the same split. Without that, two boxes cut from the same spectrum could
end up in train and test, and the reported score would be inflated by a model
that had effectively already seen the answer.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml

from callisto_trainer import __version__
from callisto_trainer.core.crops import (
    CropConfig,
    PixelBox,
    crop_from_normalized,
    normalize_full_spectrum,
    whole_file_box,
)
from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.fits_reader import read_fits_spectrum, read_fits_spectrum_and_axes
from callisto_trainer.core.region_features import REGION_FEATURES
from callisto_trainer.core.type_priors import normalise_shares
from callisto_trainer.core.taxonomy import (
    BOX_TYPES,
    MIN_SUBCLASS_BOXES,
    NO_BURST,
    RFI,
    UNIFIED_LABEL_ORDER,
    fold_rare_subclasses,
    ordered_classes,
)
from callisto_trainer.services.physics_service import physics_to_columns
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.manifest import (
    assign_stratified_split,
    count_event_leakage,
    event_key_for_path,
)
from callisto_trainer.store.repository import (
    BURST_TYPES,
    VERDICT_BURST,
    VERDICT_NO_BURST,
    AnnotationRepository,
    BoxRecord,
    FileRecord,
)

LOGGER = get_logger(__name__)

# Upstream manifest columns plus the box provenance this tool adds.
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
    # Trainer additions: provenance back to the exact annotation.
    "box_id",
    "row0",
    "row1",
    "col0",
    "col1",
    "freq_lo_mhz",
    "freq_hi_mhz",
    "t_start_s",
    "t_end_s",
    "freq_axis_source",
    # Unified track: where this crop came from (a drawn box, or mined background),
    # and for an RFI sample the interference signature that named it.
    "region_source",
    "rfi_kind",
]

# Measured burst physics, written per row by the unified exporter and read back
# by the dataset's physics branch. See core/burst_physics.py.
PHYSICS_MANIFEST_COLUMNS = [
    "freq_start_mhz", "freq_end_mhz", "freq_high_mhz", "freq_low_mhz",
    "time_start_s", "time_end_s", "duration_s", "bandwidth_mhz",
    "drift_mhz_per_s", "relative_drift_per_s",
    "fit_quality", "track_samples", "track_axis", "edge_clipped",
    "physics_confidence", "burst_count",
]
# Region features, for inspecting a snapshot by eye. The model reads the exact
# vector from each sample's .npz, never these rounded-through-text copies.
REGION_MANIFEST_COLUMNS = [f"rf_{name}" for name in REGION_FEATURES]
MANIFEST_COLUMNS = MANIFEST_COLUMNS + PHYSICS_MANIFEST_COLUMNS + REGION_MANIFEST_COLUMNS

# One row per file in a unified snapshot: what file-level evaluation judges.
FILES_MANIFEST_COLUMNS = [
    "file_path", "file_name", "station", "verdict", "split", "burst_boxes",
]

BINARY_CLASSES = {"No_Burst": 0, "Burst": 1}
# Every burst type; a snapshot keeps only those it has examples of.
TYPE_CLASSES = {name: index for index, name in enumerate(BURST_TYPES)}

# The unified model: one softmax over background, RFI and every burst type,
# applied to a region. Index 0 is No_Burst. This is the full set; a snapshot
# holds the subset it has examples of, after folding rare subclasses, and records
# the mapping it actually used.
NO_BURST_CLASS = NO_BURST
UNIFIED_CLASSES = {name: index for index, name in enumerate(UNIFIED_LABEL_ORDER)}

# Default mix of the unified export. Background outnumbers bursts by far in real
# use, and the old 1:1 mix was one reason the model called too much a burst.
DEFAULT_NEGATIVE_RATIO = 3.0
# Share of confirmed quiet files that also contribute a synthetic-RFI example.
DEFAULT_SYNTHETIC_RFI_RATIO = 0.2
# Whether unified samples carry the third, quiet-background view (see
# core/crops.py): it keeps continua lasting most of a file (Type IV) visible.
DEFAULT_QUIET_VIEW = True
# Synthetic examples kept per injected pattern.
SYNTHETIC_PER_FILE = 3
# False-alarm budget the post-training calibration tunes the threshold to: at
# most this fraction of held-out quiet files may be flagged as containing a burst.
DEFAULT_MAX_FALSE_ALARM_RATE = 0.05

# Filename tag per sample source, so a sample's origin is visible in its name.
_SOURCE_TAGS = {
    "manual": "b",
    "matched_region": "m",
    "burst_file_background": "n",
    "quiet_file": "q",
    "synthetic_rfi": "s",
}

# Snapshot directory name -> (menu label, trainer task name). Unified is first
# because it is the default: one model answering detection, typing and location.
SNAPSHOT_KINDS: list[tuple[str, str, str]] = [
    ("unified", "Unified — burst, type and location (recommended)", "unified"),
    ("types", "Burst type only (legacy stage 2)", "type"),
    ("binary", "Burst / no burst only (legacy stage 1)", "binary"),
]


def task_for_kind(kind: str) -> str:
    """Trainer task name for a snapshot directory name."""
    for directory, _label, task in SNAPSHOT_KINDS:
        if directory == kind:
            return task
    return "binary"


@dataclass
class ExportResult:
    """What an export produced, and whether it is fit to train on."""

    kind: str
    directory: Path
    manifest_path: Path
    rows: int = 0
    written: int = 0
    failed: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    class_split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    event_leakage: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    # Unified track: finder regions that matched a drawn box (kept as positives)
    # and those in the ambiguous band (dropped).
    matched_regions: int = 0
    ambiguous_regions: int = 0
    measured_physics: int = 0
    # Unified track: carrier fragments inside burst boxes that were dropped
    # rather than labelled as bursts, synthetic RFI examples added, negatives
    # labelled RFI automatically (by signature), classes folded into their
    # fallback, and the per-file listing for file-level evaluation.
    carrier_regions_dropped: int = 0
    line_regions_dropped: int = 0
    # Overlapping boxes: regions labelled by the smaller of two boxes of
    # different types, or by same-type boxes together; and regions dropped
    # because they were split between two types.
    overlap_labelled: dict[str, int] = field(default_factory=dict)
    mixed_type_regions_dropped: int = 0
    synthetic_rfi: int = 0
    automatic_rfi: dict[str, int] = field(default_factory=dict)
    folded: dict[str, str] = field(default_factory=dict)
    classes: dict[str, int] = field(default_factory=dict)
    files_manifest_path: Path | None = None

    def summary(self) -> str:
        classes = ", ".join(f"{name}: {count:,}" for name, count in sorted(self.class_counts.items()))
        return f"{self.written:,} samples ({classes})"

    def blocking_problems(self, minimum_per_split: int = 1) -> list[str]:
        """Reasons this snapshot cannot be trained on, in plain language."""
        problems: list[str] = []
        if self.written == 0:
            problems.append("The export is empty - nothing has been labelled yet.")
        for class_name, splits in sorted(self.class_split_counts.items()):
            for split in ("train", "val", "test"):
                count = splits.get(split, 0)
                if count < minimum_per_split:
                    problems.append(
                        f"'{class_name}' has {count} sample(s) in the {split} split; "
                        f"at least {minimum_per_split} is required. Label more of this class."
                    )
        if self.event_leakage:
            problems.append(
                f"{self.event_leakage} solar event(s) span multiple splits, which would "
                "inflate the reported score."
            )
        return problems

    def balance_warnings(self, ratio_limit: float = 3.0) -> list[str]:
        """Imbalance that will not stop training but will distort what it reports.

        Worth surfacing loudly: a set that is 86% one class produces a model whose
        headline F1 beats nothing, and whose tuned threshold collapses to the edge
        of its search range. The metrics look fine; the model misses real events.
        """
        warnings: list[str] = []
        if not self.class_counts or len(self.class_counts) < 2:
            return warnings

        largest = max(self.class_counts.items(), key=lambda item: item[1])
        smallest = min(self.class_counts.items(), key=lambda item: item[1])
        if smallest[1] == 0:
            return warnings

        ratio = largest[1] / smallest[1]
        if ratio >= ratio_limit:
            share = largest[1] / max(1, self.written)
            warnings.append(
                f"Class imbalance {ratio:.1f}:1 - '{largest[0]}' has {largest[1]:,} samples "
                f"({share:.0%} of the set) against {smallest[1]:,} for '{smallest[0]}'. "
                f"A model that always answers '{largest[0]}' would already score "
                f"{2 * share / (1 + share):.2f} F1, so headline scores will look good "
                f"regardless of whether anything was learned. Label more '{smallest[0]}' "
                "examples."
            )
        if smallest[1] < 50:
            warnings.append(
                f"Only {smallest[1]:,} '{smallest[0]}' sample(s) in total. That is very "
                "few to learn a class from; expect it to be missed often in real use."
            )
        return warnings


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _npz_stem(path: str) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith(".fit.gz") else Path(name).stem


def _write_npz(
    path: Path,
    tensor: np.ndarray,
    label_id: int | None,
    source: str,
    metadata: dict,
    features: np.ndarray | None = None,
) -> None:
    """Write one sample. ``label_id`` None leaves the id to the manifest alone.

    The unified export does that: its class ids are only final once every file
    has been read (automatic RFI can fold into No_Burst), and training reads the
    id from the manifest in any case. The label *name* is in the metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(
        spectrum=tensor,
        source_file=str(source),
        metadata_json=json.dumps(metadata),
    )
    if label_id is not None:
        payload["label_id"] = int(label_id)
    if features is not None:
        payload["features"] = np.asarray(features, dtype=np.float32)
    np.savez(path, **payload)


def _base_row(record: FileRecord, processed_path: Path, label: str, label_id: int) -> dict[str, Any]:
    return {
        "file_path": record.path,
        "processed_path": str(processed_path),
        "label": label,
        "label_id": label_id,
        "station": record.station or "",
        "date": record.obs_date or "",
        "start_time": record.obs_time or "",
        "freq_min_mhz": record.freq_min_mhz if record.freq_min_mhz is not None else "",
        "freq_max_mhz": record.freq_max_mhz if record.freq_max_mhz is not None else "",
        "n_freq": record.n_freq or "",
        "n_time": record.n_time or "",
        "split": "",
        "box_id": "",
        "row0": "",
        "row1": "",
        "col0": "",
        "col1": "",
        "freq_lo_mhz": "",
        "freq_hi_mhz": "",
        "t_start_s": "",
        "t_end_s": "",
        "freq_axis_source": record.freq_axis_source,
        "region_source": "",
        "rfi_kind": "",
        **{name: "" for name in PHYSICS_MANIFEST_COLUMNS},
        **{name: "" for name in REGION_MANIFEST_COLUMNS},
    }


def _finalise(
    rows: list[dict[str, Any]],
    result: ExportResult,
    pipeline_config: dict[str, Any],
    file_splits: dict[str, str] | None = None,
) -> None:
    """Assign splits, tally counts and write the manifest.

    With ``file_splits`` every row takes its file's split; otherwise rows are
    split among themselves, grouped by event.
    """
    if file_splits is not None:
        for row in rows:
            row["split"] = file_splits[row["file_path"]]
    else:
        split_cfg = pipeline_config["data"]["split"]
        rows = assign_stratified_split(
            rows,
            train_ratio=float(split_cfg["train"]),
            val_ratio=float(split_cfg["val"]),
            test_ratio=float(split_cfg["test"]),
            seed=int(split_cfg["seed"]),
            group_by_event=True,
        )

    for row in rows:
        label, split = row["label"], row["split"]
        result.class_counts[label] = result.class_counts.get(label, 0) + 1
        result.split_counts[split] = result.split_counts.get(split, 0) + 1
        result.class_split_counts.setdefault(label, {})
        result.class_split_counts[label][split] = (
            result.class_split_counts[label].get(split, 0) + 1
        )

    result.event_leakage = count_event_leakage(rows)
    result.rows = len(rows)

    with result.manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_snapshot(
    result: ExportResult,
    pipeline_config: dict[str, Any],
    classes: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "kind": result.kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tool_version": __version__,
        "classes": classes,
        "samples": result.written,
        "failed": result.failed,
        "class_counts": result.class_counts,
        "split_counts": result.split_counts,
        "class_split_counts": result.class_split_counts,
        "event_leakage": result.event_leakage,
        "preprocessing": pipeline_config["preprocessing"],
        "crops": pipeline_config.get("crops", {}),
        "split": pipeline_config["data"]["split"],
        "target_shape": pipeline_config["data"]["target_shape"],
    }
    if extra:
        payload.update(extra)
    with (result.directory / "snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# -- type (crop) export ----------------------------------------------------


def export_type_dataset(
    repository: AnnotationRepository,
    datasets_dir: str | Path,
    pipeline_config: dict[str, Any],
    outputs_dir: str | Path,
    progress: Callable[[int, int, str], bool | None] | None = None,
    min_subclass_boxes: int = MIN_SUBCLASS_BOXES,
) -> ExportResult:
    """Write one crop tensor per confirmed burst box on a burst-marked file.

    Only burst boxes are used. A rarely used class (Type IIIG, Type IV) is
    folded into its fallback exactly as the unified export does, and only the
    burst types with examples become classes.
    """
    directory = Path(datasets_dir) / "types" / _run_id()
    directory.mkdir(parents=True, exist_ok=True)
    result = ExportResult("types", directory, directory / "manifest.csv")
    crop_config = CropConfig.from_config(pipeline_config)

    # Group by file so each gzipped spectrum is decoded and normalized once,
    # however many bursts were marked in it.
    grouped: dict[int, tuple[FileRecord, list[BoxRecord]]] = {}
    for record, box in repository.iter_labeled_boxes():
        if box.burst_type not in BURST_TYPES:
            continue
        grouped.setdefault(record.id, (record, []))[1].append(box)

    drawn = Counter(box.burst_type for _, boxes in grouped.values() for box in boxes)
    label_map, folded = fold_rare_subclasses(drawn, min_subclass_boxes)
    present = {label_map[name] for name in drawn if name in TYPE_CLASSES}
    type_classes = {
        name: index for index, name in enumerate(n for n in BURST_TYPES if n in present)
    }
    result.classes = type_classes
    result.folded = {child: parent for child, parent in folded.items() if drawn.get(child)}

    rows: list[dict[str, Any]] = []
    total = len(grouped)
    for index, (record, boxes) in enumerate(grouped.values()):
        if progress is not None and progress(index, total, record.file_name) is False:
            break
        try:
            spectrum, _ = read_fits_spectrum(record.path)
            normalized = normalize_full_spectrum(spectrum, pipeline_config)
        except Exception as exc:
            result.failed += len(boxes)
            result.errors.append((record.path, repr(exc)))
            LOGGER.warning("Could not export crops from %s: %r", record.path, exc)
            continue

        for box in boxes:
            if box.burst_type not in TYPE_CLASSES:
                result.failed += 1
                result.errors.append((record.path, f"unknown burst type {box.burst_type!r}"))
                continue
            try:
                tensor = crop_from_normalized(
                    normalized, PixelBox(box.row0, box.row1, box.col0, box.col1), crop_config
                )
            except ValueError as exc:
                result.failed += 1
                result.errors.append((record.path, repr(exc)))
                continue

            label = label_map[box.burst_type]
            label_id = type_classes[label]
            processed = directory / "npz" / f"{_npz_stem(record.path)}__b{box.id}.npz"
            _write_npz(
                processed,
                tensor,
                label_id,
                record.path,
                {
                    "station": record.station,
                    "date": record.obs_date,
                    "start_time": record.obs_time,
                    "box_id": box.id,
                    "burst_type": box.burst_type,
                    "pixel_box": [box.row0, box.row1, box.col0, box.col1],
                    "freq_lo_mhz": box.freq_lo_mhz,
                    "freq_hi_mhz": box.freq_hi_mhz,
                    "t_start_s": box.t_start_s,
                    "t_end_s": box.t_end_s,
                    "freq_axis_source": record.freq_axis_source,
                },
            )

            row = _base_row(record, processed, label, label_id)
            row.update(
                {
                    "box_id": box.id,
                    "row0": box.row0,
                    "row1": box.row1,
                    "col0": box.col0,
                    "col1": box.col1,
                    "freq_lo_mhz": box.freq_lo_mhz if box.freq_lo_mhz is not None else "",
                    "freq_hi_mhz": box.freq_hi_mhz if box.freq_hi_mhz is not None else "",
                    "t_start_s": box.t_start_s if box.t_start_s is not None else "",
                    "t_end_s": box.t_end_s if box.t_end_s is not None else "",
                }
            )
            rows.append(row)
            result.written += 1

    _finalise(rows, result, pipeline_config)
    _write_snapshot(
        result,
        pipeline_config,
        type_classes,
        {"source_files": len(grouped), "folded": result.folded},
    )
    write_training_config(result, pipeline_config, outputs_dir, type_classes, task="type")
    LOGGER.info("Type export: %s -> %s", result.summary(), directory)
    return result


# -- unified (region) export -----------------------------------------------


def export_unified_dataset(
    repository: AnnotationRepository,
    datasets_dir: str | Path,
    pipeline_config: dict[str, Any],
    outputs_dir: str | Path,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    max_matched_per_box: int = 2,
    progress: Callable[[int, int, str], bool | None] | None = None,
    hard_negative_checkpoint: str | Path | None = None,
    synthetic_rfi_ratio: float = DEFAULT_SYNTHETIC_RFI_RATIO,
    min_subclass_boxes: int = MIN_SUBCLASS_BOXES,
    drop_line_positives: bool = True,
    type_frequencies: dict[str, float] | None = None,
    quiet_view: bool = DEFAULT_QUIET_VIEW,
) -> ExportResult:
    """One sample per region, over background, RFI and every burst type.

    Where the samples come from:

    * **burst files** -- every drawn burst box, the finder regions inside them,
      and a few negatives from what the finder proposes outside them;
    * **quiet files** -- the regions the inference-time finder proposes, all
      negatives by the operator's own verdict; with ``synthetic_rfi_ratio`` > 0,
      some of them also get a synthetic interference pattern painted on (see
      :mod:`callisto_trainer.core.synthetic_rfi`);
    * ``hard_negative_checkpoint`` -- a previous unified model; when given, the
      negatives it most wanted to call a burst are mined first, so every
      retraining concentrates on the last model's own false positives.

    Every negative is then **labelled RFI automatically** when its measured
    features carry an interference signature (``core/rfi_labels.py``) and left
    No_Burst otherwise; nothing is drawn as RFI. When fewer than
    ``min_subclass_boxes`` RFI samples result, or a split has none, RFI is folded
    into No_Burst. A burst class with fewer than ``min_subclass_boxes`` drawn
    boxes is folded into its fallback (Type IIIG into Type III, Type IV into
    Other), so a fresh dataset trains from day one. The snapshot records both.

    ``drop_line_positives`` leaves line-shaped finder regions inside burst boxes
    out of the burst classes (see ``negatives.assign_regions``): it stops carrier
    segments crossing a generous box being taught as bursts, at the price of
    fewer examples of genuinely thin lanes.

    ``type_frequencies`` are how often each burst type really occurs (see
    ``core/type_priors.py``); they go into the training config, where they set
    how the trained model weighs the types when it is unsure.

    ``quiet_view`` adds a third view to every sample: the context strip on a
    quiet-part background (``crops.quiet_normalized_spectrum``), which keeps a
    continuum lasting most of the file visible where the median background
    flattens it. Snapshots grow by half.
    """
    from callisto_trainer.core.burst_physics import cadence_seconds
    from callisto_trainer.core.negatives import (
        MINING_POOL,
        assign_regions,
        negative_budget,
        select_negatives,
    )
    from callisto_trainer.core.region_finder import (
        DEFAULT_MIN_AREA,
        find_candidate_regions,
        resolve_threshold,
    )
    from callisto_trainer.core.region_inputs import (
        V2_FEATURE_SET,
        V2_VIEWS,
        V3_VIEWS,
        RegionEncoder,
        RegionInputSpec,
    )
    from callisto_trainer.core.rfi_labels import interference_kind
    from callisto_trainer.core.synthetic_rfi import inject_rfi

    directory = Path(datasets_dir) / "unified" / _run_id()
    directory.mkdir(parents=True, exist_ok=True)
    result = ExportResult("unified", directory, directory / "manifest.csv")
    encoder = RegionEncoder(
        pipeline_config,
        RegionInputSpec(views=V3_VIEWS if quiet_view else V2_VIEWS, feature_set=V2_FEATURE_SET),
    )

    grouped: dict[int, tuple[FileRecord, list[BoxRecord]]] = {}
    for record, box in repository.iter_training_boxes():
        if box.burst_type not in BOX_TYPES:
            result.failed += 1
            result.errors.append((record.path, f"unknown box type {box.burst_type!r}"))
            continue
        grouped.setdefault(record.id, (record, []))[1].append(box)
    burst_files = list(grouped.values())
    quiet_files = repository.files_with_verdict([VERDICT_NO_BURST])

    drawn = Counter(box.burst_type for _, boxes in grouped.values() for box in boxes)
    label_map, folded = fold_rare_subclasses(drawn, min_subclass_boxes)

    # Each drawn burst box yields itself plus up to max_matched_per_box
    # finder-shaped copies, so the negative budget is sized against that total.
    burst_boxes = sum(drawn.values())
    positives = burst_boxes * (1 + max_matched_per_box)
    per_quiet, per_burst = negative_budget(
        positives, len(quiet_files), len(burst_files), target_ratio=negative_ratio
    )
    LOGGER.info(
        "Unified export: %d positive box(es); folded %s; mining up to %d negative(s) per "
        "quiet file and %d per burst file",
        positives, folded or "nothing", per_quiet, per_burst,
    )

    hard_negatives = None
    if hard_negative_checkpoint is not None:
        from callisto_trainer.core.inference import CascadePredictor

        hard_negatives = CascadePredictor(
            pipeline_config, unified_checkpoint=hard_negative_checkpoint
        )
        LOGGER.info("Mining hard negatives with %s", hard_negative_checkpoint)

    def hardness(normalized, axes, rfi, candidates, spectrum) -> list[float]:
        if not candidates:
            return []
        if hard_negatives is None:
            return [float(item.peak) for item in candidates]
        return [
            float(value) for value in hard_negatives.burst_evidence_for(
                normalized, axes, [item.as_box() for item in candidates], rfi_channels=rfi,
                quiet=hard_negatives.quiet_for(spectrum),
            )
        ]

    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    def emit(record, normalized, axes, context, box: PixelBox, label, source, sample_id,
             rfi_kind: str = "", quiet=None):
        """Write one sample -- both views, its features -- and its manifest row.

        A negative is named RFI here when its features carry an interference
        signature. Class ids are assigned once every sample is known.
        """
        label = label_map.get(label, label)
        try:
            encoded = encoder.encode(normalized, box, axes, context, quiet=quiet)
        except ValueError as exc:
            result.failed += 1
            result.errors.append((record.path, repr(exc)))
            return
        if encoded.physics is not None and encoded.physics.measured:
            result.measured_physics += 1
        if label == NO_BURST:
            kind = interference_kind(encoded.region, box.as_tuple())
            if kind is not None:
                label, rfi_kind = RFI, kind
                result.automatic_rfi[kind] = result.automatic_rfi.get(kind, 0) + 1

        tag = _SOURCE_TAGS.get(source, "x")
        processed = directory / "npz" / f"{_npz_stem(record.path)}__{tag}{sample_id}.npz"
        _write_npz(
            processed,
            encoded.image,
            None,
            record.path,
            {
                "station": record.station,
                "date": record.obs_date,
                "start_time": record.obs_time,
                "label": label,
                "rfi_kind": rfi_kind,
                "region_source": source,
                "pixel_box": [box.row0, box.row1, box.col0, box.col1],
                "freq_axis_source": record.freq_axis_source,
                "views": list(encoder.spec.views),
                "feature_set": encoder.spec.feature_set,
            },
            features=encoded.features,
        )
        row = _base_row(record, processed, label, -1)
        row.update(
            {
                "box_id": sample_id,
                "row0": box.row0,
                "row1": box.row1,
                "col0": box.col0,
                "col1": box.col1,
                "region_source": source,
                "rfi_kind": rfi_kind,
                **physics_to_columns(encoded.physics),
                **{
                    f"rf_{name}": round(float(value), 5)
                    for name, value in (encoded.region or {}).items()
                },
            }
        )
        rows.append(row)
        result.written += 1

    def read(record):
        spectrum, metadata = read_fits_spectrum_and_axes(record.path)
        normalized = normalize_full_spectrum(spectrum, pipeline_config)
        axes = SpectrumAxes.from_metadata(metadata)
        rfi = metadata.get("rfi_channels_mhz")
        return (
            normalized, axes, rfi, encoder.context(normalized, axes, rfi),
            encoder.quiet(spectrum), spectrum,
        )

    def file_entry(record, boxes) -> dict[str, Any]:
        return {
            "file_path": record.path,
            "file_name": record.file_name,
            "station": record.station or "",
            "verdict": record.verdict or "",
            "split": "",
            "burst_boxes": json.dumps(
                [[b.row0, b.row1, b.col0, b.col1, b.burst_type] for b in boxes]
            ),
        }

    def emit_matched(record, normalized, axes, context, assigned, wanted: set[str], source,
                     quiet=None):
        """Finder regions assigned to a drawn box, capped per box."""
        chosen = [
            (index, region) for index, region in enumerate(assigned) if region.label in wanted
        ]
        # Strongest containment first, so the cap keeps the regions most solidly
        # inside the box rather than whatever the finder listed first.
        chosen.sort(key=lambda pair: (pair[1].containment, pair[1].peak), reverse=True)
        taken: dict[Any, int] = {}
        for index, region in chosen:
            if taken.get(region.matched_box_id, 0) >= max_matched_per_box:
                continue
            taken[region.matched_box_id] = taken.get(region.matched_box_id, 0) + 1
            result.matched_regions += 1
            emit(record, normalized, axes, context, region.as_box(), region.label,
                 source, f"{record.id}_{index}", quiet=quiet)

    def count_dropped(assigned) -> None:
        for region in assigned:
            if region.overlap:
                result.overlap_labelled[region.overlap] = (
                    result.overlap_labelled.get(region.overlap, 0) + 1
                )
            if region.label is None:
                result.ambiguous_regions += 1
                if region.reason.startswith("carrier"):
                    result.carrier_regions_dropped += 1
                elif region.reason.startswith("line"):
                    result.line_regions_dropped += 1
                elif region.reason == "mixed types":
                    result.mixed_type_regions_dropped += 1

    total = len(burst_files) + len(quiet_files)
    index = 0
    burst_types = set(BURST_TYPES)

    # Burst files: drawn boxes, finder regions inside them, negatives outside.
    for record, boxes in burst_files:
        if progress is not None and progress(index, total, record.file_name) is False:
            break
        index += 1
        try:
            normalized, axes, rfi, context, quiet, spectrum = read(record)
        except Exception as exc:
            result.failed += len(boxes)
            result.errors.append((record.path, repr(exc)))
            LOGGER.warning("Could not export from %s: %r", record.path, exc)
            continue
        files.append(file_entry(record, boxes))

        for box in boxes:
            emit(record, normalized, axes, context,
                 PixelBox(box.row0, box.row1, box.col0, box.col1), box.burst_type,
                 "manual", box.id, quiet=quiet)

        # Finder-shaped positives are essential: without them the burst classes
        # would be hand-drawn boxes while every negative came from the finder,
        # and the model would separate the two by region *shape* rather than by
        # content. See negatives.assign_regions.
        assigned = assign_regions(
            normalized, boxes, context=context, max_candidates=MINING_POOL,
            drop_lines=drop_line_positives,
        )
        count_dropped(assigned)
        emit_matched(record, normalized, axes, context, assigned, burst_types, "matched_region",
                     quiet=quiet)

        background = [(i, r) for i, r in enumerate(assigned) if r.label == NO_BURST]
        scores = hardness(normalized, axes, rfi, [r for _, r in background], spectrum)
        for position, region in select_negatives(
            background, scores, per_burst, seed=record.id
        ):
            emit(record, normalized, axes, context, region.as_box(), NO_BURST,
                 "burst_file_background", f"{record.id}_{position}", quiet=quiet)

    # Confirmed quiet files: whatever the finder proposes is, by the operator's
    # own verdict, not a burst.
    for record in quiet_files:
        if progress is not None and progress(index, total, record.file_name) is False:
            break
        index += 1
        try:
            normalized, axes, rfi, context, quiet, spectrum = read(record)
        except Exception as exc:
            result.failed += 1
            result.errors.append((record.path, repr(exc)))
            continue
        files.append(file_entry(record, []))

        proposals = find_candidate_regions(
            normalized, threshold=resolve_threshold(normalized), min_area=DEFAULT_MIN_AREA,
            max_candidates=MINING_POOL,
        )
        scores = hardness(normalized, axes, rfi, proposals, spectrum)
        for position, proposal in select_negatives(
            list(enumerate(proposals)), scores, per_quiet, seed=record.id
        ):
            emit(record, normalized, axes, context, proposal.as_box(), NO_BURST,
                 "quiet_file", f"{record.id}_{position}", quiet=quiet)

        rng = np.random.default_rng([int(pipeline_config["data"]["split"]["seed"]), record.id])
        if synthetic_rfi_ratio > 0 and rng.random() < synthetic_rfi_ratio:
            injected = inject_rfi(normalized, rng, cadence_s=cadence_seconds(axes))
            level = resolve_threshold(injected.normalized)
            injected_context = encoder.context(injected.normalized, axes, rfi)
            # The same pattern, raised by the same amount, on the quiet background.
            injected_quiet = None if quiet is None else np.clip(
                quiet + (injected.normalized - normalized), 0.0, 1.0
            ).astype(np.float32)
            kept = 0
            for number, proposal in enumerate(find_candidate_regions(
                injected.normalized, threshold=level, min_area=DEFAULT_MIN_AREA,
                max_candidates=MINING_POOL,
            )):
                window = (
                    slice(proposal.row0, proposal.row1), slice(proposal.col0, proposal.col1)
                )
                bright = injected.normalized[window] >= level
                on_pattern = float((bright & injected.mask[window]).sum()) / max(1, bright.sum())
                # Only candidates that are the pattern, not real background it
                # happened to merge with.
                if on_pattern < 0.5:
                    continue
                emit(record, injected.normalized, axes, injected_context, proposal.as_box(),
                     RFI, "synthetic_rfi", f"{record.id}_{number}",
                     rfi_kind=f"synthetic {injected.pattern}", quiet=injected_quiet)
                kept += 1
                if kept >= SYNTHETIC_PER_FILE:
                    break
            result.synthetic_rfi += kept

    splits = _split_files(files, pipeline_config)
    rfi_folded = _fold_scarce_rfi(rows, splits, min_subclass_boxes)
    if rfi_folded:
        folded = {**folded, RFI: NO_BURST}
    classes = ordered_classes({row["label"] for row in rows} | {NO_BURST})
    for row in rows:
        row["label_id"] = classes[row["label"]]
    result.classes = classes
    result.folded = folded

    _finalise(rows, result, pipeline_config, file_splits=splits)
    result.files_manifest_path = _write_files_manifest(directory, files, splits)
    _write_snapshot(
        result,
        pipeline_config,
        classes,
        {
            "source_files": len(files),
            "views": list(encoder.spec.views),
            "feature_set": encoder.spec.feature_set,
            "folded": folded,
            "drawn_boxes": dict(drawn),
            "negative_mining": {
                "per_quiet_file": per_quiet,
                "per_burst_file": per_burst,
                "target_ratio": negative_ratio,
                "hard_negative_checkpoint": (
                    str(hard_negative_checkpoint) if hard_negative_checkpoint else None
                ),
                "note": "negatives are inference-time candidate regions, not random background",
            },
            "rfi": {
                "automatic": dict(result.automatic_rfi),
                "synthetic": result.synthetic_rfi,
                "synthetic_file_ratio": synthetic_rfi_ratio,
                "folded_into_no_burst": rfi_folded,
                "note": "RFI is never drawn: negatives with an interference signature "
                "(core/rfi_labels.py) are labelled RFI, the rest No_Burst",
            },
            "physics": {
                "measured": result.measured_physics,
                "of_samples": result.written,
                "note": "drift rate and burst extent measured inside each region; "
                "unmeasurable regions carry an explicit flag so the model can "
                "distinguish them from a genuine zero",
            },
            "region_assignment": {
                "matched_regions": result.matched_regions,
                "ambiguous_dropped": result.ambiguous_regions,
                "carrier_fragments_dropped": result.carrier_regions_dropped,
                "line_regions_dropped": result.line_regions_dropped,
                "drop_line_positives": drop_line_positives,
                "overlapping_boxes": {
                    "labelled_by_smaller_box": result.overlap_labelled.get("smaller box", 0),
                    "labelled_by_joined_same_type_boxes": result.overlap_labelled.get(
                        "joined boxes", 0
                    ),
                    "mixed_types_dropped": result.mixed_type_regions_dropped,
                    "note": "each pixel belongs to the smallest box covering it; a region "
                    "takes the type owning >= 60% of it (see negatives.assign_regions)",
                },
                "note": "finder regions overlapping a drawn box are added as positives so "
                "both classes share the inference-time region distribution",
            },
        },
    )
    write_training_config(
        result, pipeline_config, outputs_dir, classes, task="unified",
        type_frequencies=type_frequencies, views=encoder.spec.views,
    )
    LOGGER.info("Unified export: %s -> %s", result.summary(), directory)
    return result


def _fold_scarce_rfi(
    rows: list[dict[str, Any]], splits: dict[str, str], minimum: int
) -> bool:
    """Relabel automatic RFI as No_Burst when there is too little of it to learn.

    Too little means fewer than ``minimum`` samples, or a split with none -- a
    class missing from a split blocks training, and this one is the exporter's
    own creation, so it must never be what blocks the operator. The samples keep
    their ``rfi_kind``; only the class they train as changes. Returns whether
    RFI was folded.
    """
    rfi_rows = [row for row in rows if row["label"] == RFI]
    if not rfi_rows:
        return False
    per_split = Counter(splits.get(row["file_path"], "") for row in rfi_rows)
    if len(rfi_rows) >= minimum and all(per_split.get(s, 0) for s in ("train", "val", "test")):
        return False
    for row in rfi_rows:
        row["label"] = NO_BURST
    return True


def _split_files(files: list[dict[str, Any]], pipeline_config: dict[str, Any]) -> dict[str, str]:
    """Train / val / test per *file*, stratified by verdict and grouped by event.

    Splitting files rather than samples makes the split a property of the data
    alone. Split over samples, it depended on how many regions the finder
    proposed in each file -- a quiet file with no candidates had no samples and
    no split -- so changing a mining or finder setting reshuffled which files
    were held out, and two snapshots could not be compared on the same test
    files. Every recording of one solar event still lands in one split.
    """
    split_cfg = pipeline_config["data"]["split"]
    placeholders = [
        {"file_path": entry["file_path"], "label_id": int(entry["verdict"] == VERDICT_BURST)}
        for entry in files
    ]
    assign_stratified_split(
        placeholders,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
        group_by_event=True,
    )
    return {row["file_path"]: row["split"] for row in placeholders}


def _write_files_manifest(
    directory: Path, files: list[dict[str, Any]], splits: dict[str, str]
) -> Path:
    """Write ``files.csv``: every exported file, its verdict and its split.

    A quiet file whose finder found nothing has no samples, but it still belongs
    in file-level evaluation -- a correctly silent quiet file is exactly the
    outcome being measured.
    """
    for entry in files:
        entry["split"] = splits[entry["file_path"]]

    path = directory / "files.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILES_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(files)
    return path


# -- binary (whole-file) export -------------------------------------------


def export_binary_dataset(
    repository: AnnotationRepository,
    datasets_dir: str | Path,
    pipeline_config: dict[str, Any],
    outputs_dir: str | Path,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> ExportResult:
    """Write one whole-file tensor per file the operator gave a burst verdict."""
    directory = Path(datasets_dir) / "binary" / _run_id()
    directory.mkdir(parents=True, exist_ok=True)
    result = ExportResult("binary", directory, directory / "manifest.csv")

    records = repository.files_with_verdict([VERDICT_BURST, VERDICT_NO_BURST])
    rows: list[dict[str, Any]] = []
    total = len(records)

    for index, record in enumerate(records):
        if progress is not None and progress(index, total, record.file_name) is False:
            break
        label = "Burst" if record.verdict == VERDICT_BURST else "No_Burst"
        label_id = BINARY_CLASSES[label]
        try:
            spectrum, _ = read_fits_spectrum(record.path)
            normalized = normalize_full_spectrum(spectrum, pipeline_config)
            # apply_margin=False: the whole file already is the sample, and this
            # keeps the tensor bit-identical to the upstream preprocessing.
            tensor = crop_from_normalized(
                normalized,
                whole_file_box(normalized.shape),
                CropConfig.from_config(pipeline_config),
                apply_margin=False,
            )
        except Exception as exc:
            result.failed += 1
            result.errors.append((record.path, repr(exc)))
            LOGGER.warning("Could not export %s: %r", record.path, exc)
            continue

        processed = directory / "npz" / f"{_npz_stem(record.path)}.npz"
        _write_npz(
            processed,
            tensor,
            label_id,
            record.path,
            {
                "station": record.station,
                "date": record.obs_date,
                "start_time": record.obs_time,
                "freq_min_mhz": record.freq_min_mhz,
                "freq_max_mhz": record.freq_max_mhz,
                "freq_axis_source": record.freq_axis_source,
            },
        )
        rows.append(_base_row(record, processed, label, label_id))
        result.written += 1

    _finalise(rows, result, pipeline_config)
    _write_snapshot(result, pipeline_config, BINARY_CLASSES)
    write_training_config(result, pipeline_config, outputs_dir, BINARY_CLASSES, task="binary")
    LOGGER.info("Binary export: %s -> %s", result.summary(), directory)
    return result


# -- generated training configuration -------------------------------------


def write_training_config(
    result: ExportResult,
    pipeline_config: dict[str, Any],
    outputs_dir: str | Path,
    classes: dict[str, int],
    task: str,
    type_frequencies: dict[str, float] | None = None,
    views: Iterable[str] | None = None,
) -> Path:
    """Write a ready-to-run YAML pointing at this snapshot.

    The Train tab edits and runs this file, so every run is reproducible from
    disk alone -- the config sits beside the exact data it was trained on.
    """
    run_name = result.directory.name
    outputs = Path(outputs_dir) / f"{task}_{run_name}"
    # "unified" and "type" are both multiclass, image-only, crop-based models and
    # share all their training settings; only the class set differs.
    is_multiclass = task in ("type", "unified")

    config: dict[str, Any] = {
        "project": {"name": f"callisto_trainer_{task}_{run_name}"},
        "paths": {
            "raw_dir_candidates": [str(result.directory)],
            "manifest_path": str(result.manifest_path),
            "processed_dir": str(result.directory / "npz"),
            "checkpoint_dir": str(outputs / "checkpoints"),
            "figures_dir": str(outputs / "figures"),
            "reports_dir": str(outputs / "reports"),
        },
        "data": {
            "classes": classes,
            "target_shape": list(pipeline_config["data"]["target_shape"]),
            "drop_missing_processed": True,
            "assume_processed_complete": True,  # the exporter just wrote them all
            "split": dict(pipeline_config["data"]["split"]),
        },
        "preprocessing": dict(pipeline_config["preprocessing"]),
        "crops": dict(pipeline_config.get("crops", {})),
        "augmentation": dict(pipeline_config["augmentation"]),
        "performance": dict(pipeline_config["performance"]),
        "model": {
            "name": "resnet18",
            "in_channels": 1,
            "dropout": 0.25,
            "pretrained": True,
            "num_classes": len(classes) if is_multiclass else 1,
            # The type model is deliberately image-only: what separates Type II
            # from Type III is the frequency-drift morphology, not the station.
            "use_metadata": not is_multiclass,
            "station_emb_dim": 8,
            # Fuse measured burst physics (drift rate, extent, fit quality) with
            # the image backbone. Only the unified track measures them, and drift
            # rate is what physically separates the burst types.
            "use_physics": task == "unified",
        },
        "training": {
            "batch_size": 64,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "epochs": 60 if is_multiclass else 40,
            "learning_rate": 0.0003 if is_multiclass else 0.001,
            # The crop tracks have few distinct examples per class and a large
            # pretrained backbone, so they carry the heavier decay.
            "weight_decay": 0.01 if is_multiclass else 0.0001,
            "label_smoothing": 0.05,
            # 0 = no early stopping, so the epoch budget above runs in full.
            # best.pt still selects on validation score, so nothing is lost by
            # training past a plateau. Raise it to stop on a plateau again.
            "patience": 0,
            "seed": 42,
            "monitor": "macro_f1" if is_multiclass else "pr_auc",
            "scheduler": "cosine" if is_multiclass else "plateau",
            "keep_checkpoints": 3,
            "class_balance": (
                # exponent damps the inverse-frequency correction; see
                # train_type._make_criterion. 1.0 would put a ~19x spread across
                # this archive's classes, concentrated on the ones with the
                # fewest distinct bursts to learn from.
                {"strategy": "auto_class_weights", "exponent": 0.5}
                if is_multiclass
                else {"strategy": "auto_pos_weight"}
            ),
        },
    }
    if not is_multiclass:
        config["training"].update(
            {
                "threshold": 0.5,
                "auto_threshold": True,
                # f1 keeps burst recall central, which is what matters here: a
                # missed burst is a lost event, a false alarm is a glance. Set
                # "balanced_accuracy" instead if false alarms become the problem;
                # it weights both classes equally and is less swayed by the
                # positive-heavy class ratio typical of these datasets.
                "threshold_metric": "f1",
            }
        )

    if task == "unified":
        from callisto_trainer.core.region_finder import (
            DEFAULT_HYSTERESIS,
            DEFAULT_MAX_REGIONS,
            DEFAULT_MIN_AREA,
            DEFAULT_REGION_THRESHOLD,
        )
        from callisto_trainer.core.region_inputs import V2_FEATURE_SET, V2_VIEWS

        config["model"].update(
            {
                # Two or three views of every region (crop, context, and the
                # context on a quiet-part background) through one shared
                # backbone, plus the interference features. See
                # core/region_inputs.py.
                "views": list(views or V2_VIEWS),
                "feature_set": V2_FEATURE_SET,
            }
        )
        config["training"].update(
            {
                # Half detection (does the model rank bursts above background
                # and RFI), half typing (among real bursts, is the type right).
                # Macro-F1 over all classes weighted a rare burst type as heavily
                # as rejecting interference, and never looked at ranking at all.
                "monitor": "unified_score",
                # Burst types are balanced among themselves; background and RFI
                # keep full weight, so the loss is not cheapened for exactly the
                # samples that decide false alarms. See train_type._make_criterion.
                "class_balance": {
                    "strategy": "detection_balanced",
                    "exponent": 0.5,
                    "background_weight": 1.0,
                },
            }
        )
        # After training, the burst threshold is tuned on the validation files so
        # that at most this share of quiet files is flagged. The region finder
        # settings are the ones the threshold is tuned for; Predict uses them.
        config["calibration"] = {
            "split": "val",
            "max_false_alarm_rate": DEFAULT_MAX_FALSE_ALARM_RATE,
            "files_manifest": str(result.directory / "files.csv"),
        }
        config["inference"] = {
            "region_finder": {
                "threshold": DEFAULT_REGION_THRESHOLD,
                "adaptive": True,
                "min_area": DEFAULT_MIN_AREA,
                "max_regions": DEFAULT_MAX_REGIONS,
                # The finder the training regions were mined with; inference
                # must propose regions the same way.
                "hysteresis": DEFAULT_HYSTERESIS,
            },
            # How often each burst type really occurs. After training the type
            # probabilities are shifted from the training mix toward these, with
            # a strength chosen on the validation regions; see
            # core/type_priors.py. Edit the shares here or in the Dataset tab.
            "type_priors": {
                "observed_shares": normalise_shares(type_frequencies),
                "strength": "auto",
            },
        }

    path = result.directory / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    return path


def list_snapshots(datasets_dir: str | Path, kind: str) -> list[Path]:
    """Existing snapshots of one kind, newest first."""
    root = Path(datasets_dir) / kind
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "manifest.csv").exists()),
        reverse=True,
    )


def read_snapshot_info(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "snapshot.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def snapshot_size_bytes(directory: str | Path) -> int:
    """Total bytes on disk under ``directory``. Unreadable entries count as zero."""
    total = 0
    for path in Path(directory).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:  # vanished mid-walk, or permission denied
            continue
    return total


def delete_snapshot(datasets_dir: str | Path, directory: str | Path) -> int:
    """Delete one snapshot directory and return the bytes reclaimed.

    Snapshots are the provenance record for every model trained from them, and a
    single one can hold five thousand files, so this refuses anything it cannot
    positively identify as a snapshot rather than trusting the caller:

    * the path must resolve to somewhere **inside** ``datasets_dir``, which stops
      a caller from turning a stale or crafted path into a recursive delete of an
      unrelated tree;
    * it must be a directory two levels down (``datasets/<kind>/<run>``), so the
      kind directory itself -- holding every snapshot of that kind -- can never be
      removed by a path that merely looks plausible;
    * it must actually contain a snapshot's own files.

    Raises ``ValueError`` when any of that fails, ``FileNotFoundError`` when the
    directory is already gone.
    """
    root = Path(datasets_dir).resolve()
    target = Path(directory).resolve()

    if not target.exists():
        raise FileNotFoundError(f"Snapshot no longer exists: {target}")
    if not target.is_dir():
        raise ValueError(f"Not a directory: {target}")

    try:
        relative = target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Refusing to delete {target}: it is outside the datasets directory {root}"
        ) from None
    if len(relative.parts) != 2:
        raise ValueError(
            f"Refusing to delete {target}: expected a datasets/<kind>/<run> directory, "
            f"got {relative.as_posix()!r}"
        )
    if not ((target / "manifest.csv").exists() or (target / "snapshot.json").exists()):
        raise ValueError(
            f"Refusing to delete {target}: it has no manifest.csv or snapshot.json, "
            "so it does not look like a snapshot"
        )

    freed = snapshot_size_bytes(target)
    shutil.rmtree(target)
    LOGGER.info("Deleted snapshot %s (%.1f MB)", target, freed / 1e6)
    return freed
