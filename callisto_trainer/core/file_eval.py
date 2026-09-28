"""File-level evaluation and false-alarm calibration for the unified model.

## Why crop metrics were not enough

A trained model used to report 98.8% of background crops rejected on its test
split, and still flagged far too many quiet files in real use. Both numbers were
true. At inference a file is examined region by region -- up to a dozen
candidate regions each -- and it is called a burst if *any* of them is. A 1.2%
per-region false-positive rate over twelve regions is roughly a 13% per-file
false-alarm rate, before any station the model has never seen. Crop metrics
cannot show that, because they never aggregate over a file.

So this module runs the model exactly as the Predict tab does, over every
held-out file, and measures what an operator actually experiences:

* **false alarms** -- quiet files in which some region was called a burst;
* **detected bursts** -- burst files in which a region *on a drawn burst* was
  called a burst (a detection that lands on unrelated interference in a burst
  file is a lucky false positive, not a detection, and is counted as a stray);
* **strays** -- regions called a burst in burst files that lie on no drawn burst.

## Calibration

The decision threshold is then chosen on the validation files: as many burst
files found as possible while no more than ``max_false_alarm_rate`` of quiet
files are flagged, then as few false alarms as possible at that recall (see
:func:`choose_threshold`). That is the operating point the operator asked for --
"few false alarms" -- stated directly, instead of hoping argmax lands somewhere
sensible. The threshold, the budget and the finder settings it was tuned with
are stored in the checkpoint, and the Predict tab uses them.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.negatives import union_containment

LOGGER = get_logger(__name__)

# A region counts as landing on a drawn burst when at least this much of it is
# inside the box. Loose on purpose: the question is "did it find that burst", and
# finder regions often spill over a generous box's edge.
ON_BURST_CONTAINMENT = 0.25


@dataclass
class FileTruth:
    """One row of a snapshot's ``files.csv``."""

    file_path: str
    file_name: str
    station: str
    verdict: str
    split: str
    burst_boxes: list[tuple[int, int, int, int, str]] = field(default_factory=list)

    @property
    def is_burst(self) -> bool:
        return self.verdict == "burst"


def read_files_manifest(path: str | Path, split: str | None = None) -> list[FileTruth]:
    truths: list[FileTruth] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row.get("split") != split:
                continue
            truths.append(
                FileTruth(
                    file_path=row["file_path"],
                    file_name=row.get("file_name") or Path(row["file_path"]).name,
                    station=row.get("station", ""),
                    verdict=row.get("verdict", ""),
                    split=row.get("split", ""),
                    burst_boxes=[tuple(box) for box in json.loads(row.get("burst_boxes") or "[]")],
                )
            )
    return truths


@dataclass
class FileScore:
    """What the model made of one file, before any threshold is applied."""

    file_path: str
    file_name: str
    station: str
    is_burst: bool
    # Highest burst evidence over all examined regions, and over the regions
    # lying on a drawn burst (the "did it find the burst" score).
    file_score: float = 0.0
    on_burst_score: float = 0.0
    # Burst evidence of each region lying on no drawn burst, for stray counts.
    stray_scores: list[float] = field(default_factory=list)
    regions_examined: int = 0
    error: str | None = None


def lands_on_burst(region: Sequence[int], burst_boxes: Sequence[Sequence[Any]]) -> bool:
    """Whether a region lies at least :data:`ON_BURST_CONTAINMENT` on drawn bursts.

    Measured over all the drawn bursts at once, so a detection straddling two
    overlapping boxes counts, as it does in training.
    """
    if not burst_boxes:
        return False
    return union_containment(region, [box[:4] for box in burst_boxes]) >= ON_BURST_CONTAINMENT


def score_files(
    predictor: Any,
    truths: Sequence[FileTruth],
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> list[FileScore]:
    """Run ``predictor`` (a unified ``CascadePredictor``) over every file."""
    from callisto_trainer.core.coords import SpectrumAxes
    from callisto_trainer.core.crops import normalize_full_spectrum
    from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes

    scores: list[FileScore] = []
    for index, truth in enumerate(truths):
        if progress is not None and progress(index, len(truths), truth.file_name) is False:
            break
        score = FileScore(
            file_path=truth.file_path,
            file_name=truth.file_name,
            station=truth.station,
            is_burst=truth.is_burst,
        )
        try:
            spectrum, metadata = read_fits_spectrum_and_axes(truth.file_path)
            normalized = normalize_full_spectrum(spectrum, predictor.config)
            axes = SpectrumAxes.from_metadata(metadata)
            regions = predictor.examine(
                normalized, axes, rfi_channels=metadata.get("rfi_channels_mhz"),
                quiet=predictor.quiet_for(spectrum),
            )
        except Exception as exc:
            score.error = repr(exc)
            LOGGER.warning("Could not score %s: %r", truth.file_path, exc)
            scores.append(score)
            continue

        score.regions_examined = len(regions)
        for region in regions:
            evidence = float(region.burst_evidence or 0.0)
            score.file_score = max(score.file_score, evidence)
            box = (region.row0, region.row1, region.col0, region.col1)
            on_burst = lands_on_burst(box, truth.burst_boxes)
            if on_burst:
                score.on_burst_score = max(score.on_burst_score, evidence)
            elif truth.is_burst:
                score.stray_scores.append(evidence)
        scores.append(score)
    return scores


@dataclass
class ThresholdChoice:
    threshold: float
    false_alarm_rate: float
    burst_recall: float
    quiet_files: int
    burst_files: int
    met_budget: bool
    note: str = ""


def choose_threshold(
    scores: Sequence[FileScore],
    max_false_alarm_rate: float = 0.05,
    floor: float = 0.05,
) -> ThresholdChoice:
    """The operating point: most bursts found within the false-alarm budget.

    In order: (1) among thresholds flagging at most ``max_false_alarm_rate`` of
    quiet files, the highest burst recall; (2) at that recall, the fewest false
    alarms; (3) the *middle* of the range of thresholds achieving both.

    Step 2 matters more than it looks. On real validation files recall sat flat
    across most of the threshold range while false alarms fell steeply, so
    "the lowest threshold within budget" -- this function's first rule -- spent
    the whole budget for no extra burst: 7 false alarms on held-out quiet files
    where 0-1 would have found as many. Step 3 then keeps a margin on both
    sides, so the choice does not sit on the edge of one validation file.

    ``floor`` keeps any flicker of evidence from counting as a burst.
    """
    quiet = np.array([s.file_score for s in scores if not s.is_burst and s.error is None])
    bursts = np.array([s.on_burst_score for s in scores if s.is_burst and s.error is None])

    def at(threshold: float) -> tuple[float, float]:
        far = float((quiet >= threshold).mean()) if quiet.size else 0.0
        recall = float((bursts >= threshold).mean()) if bursts.size else 0.0
        return far, recall

    if quiet.size == 0:
        far, recall = at(0.5)
        return ThresholdChoice(
            0.5, far, recall, 0, int(bursts.size), False,
            "no quiet files in this split, so the false-alarm rate cannot be measured; "
            "0.5 used",
        )

    # The rates only change at observed scores: just above a quiet score (that
    # file stops being flagged) and exactly at a burst score (the last value that
    # still finds it). Those are the only thresholds worth testing.
    candidates = sorted(
        value
        for value in {float(floor), *np.nextafter(quiet, 2.0).tolist(), *bursts.tolist()}
        if floor <= value <= 1.0
    )
    feasible = [
        (threshold, *at(threshold))
        for threshold in candidates
        if at(threshold)[0] <= max_false_alarm_rate + 1e-12
    ]

    note = ""
    if not feasible:
        # Even the strongest quiet region scores ~1.0: no threshold meets the
        # budget. Say so rather than pretending.
        best = 1.0
        note = (
            "no threshold keeps false alarms within budget: some quiet files score as "
            "high as real bursts. Check them for an unlabelled burst, then retrain with "
            "hard negatives from this model"
        )
    else:
        top_recall = max(recall for _, _, recall in feasible)
        at_top = [(t, far) for t, far, recall in feasible if recall >= top_recall - 1e-12]
        fewest = min(far for _, far in at_top)
        span = [t for t, far in at_top if far <= fewest + 1e-12]
        best = 0.5 * (min(span) + max(span))
        if max(span) <= floor + 1e-12:
            note = f"threshold held at its floor of {floor:g}"
    far, recall = at(best)
    return ThresholdChoice(
        threshold=float(best),
        false_alarm_rate=far,
        burst_recall=recall,
        quiet_files=int(quiet.size),
        burst_files=int(bursts.size),
        met_budget=far <= max_false_alarm_rate + 1e-12,
        note=note,
    )


def file_level_report(scores: Sequence[FileScore], threshold: float) -> dict[str, Any]:
    """What an operator would see at ``threshold``: false alarms, detections, strays."""
    valid = [s for s in scores if s.error is None]
    quiet = [s for s in valid if not s.is_burst]
    bursts = [s for s in valid if s.is_burst]
    false_alarms = [s for s in quiet if s.file_score >= threshold]
    detected = [s for s in bursts if s.on_burst_score >= threshold]
    missed = [s for s in bursts if s.on_burst_score < threshold]
    strays = sum(sum(1 for v in s.stray_scores if v >= threshold) for s in bursts)

    by_station: dict[str, dict[str, int]] = {}
    for s in quiet:
        entry = by_station.setdefault(s.station or "?", {"quiet_files": 0, "false_alarms": 0})
        entry["quiet_files"] += 1
        entry["false_alarms"] += int(s.file_score >= threshold)

    def rate(part: int, whole: int) -> float:
        return part / whole if whole else float("nan")

    return {
        "threshold": float(threshold),
        "quiet_files": len(quiet),
        "false_alarms": len(false_alarms),
        "false_alarm_rate": rate(len(false_alarms), len(quiet)),
        "burst_files": len(bursts),
        "bursts_detected": len(detected),
        "burst_recall": rate(len(detected), len(bursts)),
        "stray_detections": strays,
        "stray_per_burst_file": rate(strays, len(bursts)),
        "unreadable": len(scores) - len(valid),
        "false_alarms_by_station": by_station,
        "false_alarm_files": [
            {"file_path": s.file_path, "station": s.station, "score": s.file_score}
            for s in sorted(false_alarms, key=lambda s: -s.file_score)
        ],
        "missed_burst_files": [
            {"file_path": s.file_path, "station": s.station, "score": s.on_burst_score}
            for s in sorted(missed, key=lambda s: s.on_burst_score)
        ],
    }


def write_file_report(report: dict[str, Any], reports_dir: str | Path, split: str) -> Path:
    """``{split}_file_metrics.json`` plus a CSV of the files to go and look at."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{split}_file_metrics.json"
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    with (reports_dir / f"{split}_file_mistakes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_path", "station", "kind", "score"]
        )
        writer.writeheader()
        for item in report.get("false_alarm_files", []):
            writer.writerow({**item, "kind": "false alarm"})
        for item in report.get("missed_burst_files", []):
            writer.writerow({**item, "kind": "missed burst"})
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def copy_calibration(source: str | Path, destination: str | Path) -> None:
    """Give ``destination`` the calibrated ``inference`` block of ``source``.

    For the per-epoch copy of the best checkpoint, which holds the same weights
    as ``best.pt`` and should decide the same way.
    """
    import torch

    inference = (
        torch.load(source, map_location="cpu", weights_only=False).get("config", {}) or {}
    ).get("inference")
    if not inference:
        return
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    checkpoint.setdefault("config", {})["inference"] = inference
    torch.save(checkpoint, destination)


def calibrate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str | None = None,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> dict[str, Any]:
    """Tune the burst threshold on held-out files and store it in the checkpoint.

    Writes ``inference.burst_threshold`` (and how it was chosen) into the
    checkpoint's embedded config, so every consumer of the checkpoint -- the
    Evaluate and Predict tabs, a model export -- picks it up. Returns the
    calibration record; it is also written to ``{split}_file_calibration.json``.
    """
    import torch

    from callisto_trainer.core.inference import CascadePredictor

    calibration = dict(config.get("calibration", {}) or {})
    split = split or str(calibration.get("split", "val"))
    budget = float(calibration.get("max_false_alarm_rate", 0.05))
    manifest = calibration.get("files_manifest") or str(
        Path(config["paths"]["manifest_path"]).with_name("files.csv")
    )
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"No files manifest at {manifest}. Re-export the snapshot: file-level "
            "calibration needs the per-file listing that newer exports write."
        )

    truths = read_files_manifest(manifest, split=split)
    predictor = CascadePredictor(config, unified_checkpoint=checkpoint_path, burst_threshold=None)
    LOGGER.info("Calibrating on %d %s file(s), false-alarm budget %.1f%%",
                len(truths), split, 100 * budget)
    scores = score_files(predictor, truths, progress=progress)
    choice = choose_threshold(scores, max_false_alarm_rate=budget)
    report = file_level_report(scores, choice.threshold)

    record = {
        "split": split,
        "max_false_alarm_rate": budget,
        **{key: value for key, value in asdict(choice).items()},
        "stray_per_burst_file": report["stray_per_burst_file"],
        "region_finder": predictor.region_finder_settings(),
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = checkpoint.setdefault("config", {})
    inference = dict(embedded.get("inference", {}) or {})
    inference["burst_threshold"] = choice.threshold
    inference["calibration"] = record
    inference["region_finder"] = predictor.region_finder_settings()
    embedded["inference"] = inference
    torch.save(checkpoint, checkpoint_path)

    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{split}_file_calibration.json").write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
    )
    write_file_report(report, reports_dir, split)
    LOGGER.info(
        "Calibrated threshold %.4f: %.1f%% of %d quiet files flagged, %.1f%% of %d burst "
        "files found%s",
        choice.threshold, 100 * choice.false_alarm_rate, choice.quiet_files,
        100 * (choice.burst_recall if not math.isnan(choice.burst_recall) else 0.0),
        choice.burst_files, f" ({choice.note})" if choice.note else "",
    )
    return record


def calibrate_type_priors(
    config: dict[str, Any], checkpoint_path: str | Path, split: str = "val"
) -> dict[str, Any]:
    """Fit the type-frequency correction and store it in the checkpoint.

    Measures what the model was trained to expect (each class's training
    samples times its loss weight), compares it with how often each type really
    occurs (``inference.type_priors.observed_shares``), and chooses how strongly
    to correct on the ``split`` regions -- see :mod:`callisto_trainer.core.type_priors`.
    Writes ``inference.type_priors`` into the checkpoint and
    ``{split}_type_priors.json`` beside the other reports. The burst threshold
    is unaffected: the correction only moves probability between burst types.
    """
    from collections import Counter

    import torch

    from callisto_trainer.core import type_priors
    from callisto_trainer.core.evaluate_type import predict_split
    from callisto_trainer.core.train_type import class_weights

    y_true, probabilities, class_names, model_config, rows = predict_split(
        config, checkpoint_path, split, with_rows=True
    )
    settings = dict(((model_config.get("inference") or {}).get("type_priors")) or {})
    setting = settings.get("strength_setting", settings.get("strength", "auto"))

    with Path(model_config["paths"]["manifest_path"]).open(encoding="utf-8", newline="") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    counts = Counter(int(row["label_id"]) for row in train_rows)
    weights = class_weights(model_config, counts, class_names) or [1.0] * len(class_names)
    boxes = Counter(row["label"] for row in train_rows if row.get("region_source") == "manual")

    observed = type_priors.class_shares(settings.get("observed_shares"), class_names, boxes)
    trained = type_priors.training_shares(
        {name: counts.get(i, 0) for i, name in enumerate(class_names)},
        dict(zip(class_names, weights)),
        class_names,
    )
    adjustment = type_priors.log_adjustment(observed, trained)
    if setting in (None, "auto"):
        strength, reports = type_priors.choose_strength(
            y_true, probabilities, class_names, observed, adjustment,
            groups=[row.get("file_path", "") for row in rows],
        )
    else:
        strength = float(setting)
        reports = [
            type_priors.real_world_type_report(
                y_true, probabilities, class_names, observed, adjustment, value
            )
            for value in sorted({0.0, strength})
        ]

    record = {
        "observed_shares": type_priors.normalise_shares(settings.get("observed_shares")),
        "class_shares": observed,
        "training_shares": trained,
        "adjustment": adjustment,
        "strength": float(strength),
        "strength_setting": setting if setting is not None else "auto",
        "chosen_on": split,
        "reports": reports,
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = checkpoint.setdefault("config", {})
    inference = dict(embedded.get("inference", {}) or {})
    inference["type_priors"] = record
    embedded["inference"] = inference
    torch.save(checkpoint, checkpoint_path)

    reports_dir = Path(model_config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{split}_type_priors.json").write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
    )
    chosen = next((r for r in reports if r and r.get("strength") == strength), {})
    LOGGER.info(
        "Type frequencies: strength %.2f chosen on %s (real-world macro-F1 %.3f, accuracy %.3f); "
        "training mix %s, observed %s",
        strength, split, chosen.get("real_world_macro_f1", float("nan")),
        chosen.get("real_world_accuracy", float("nan")),
        {k: round(v, 3) for k, v in trained.items()}, {k: round(v, 3) for k, v in observed.items()},
    )
    return record
