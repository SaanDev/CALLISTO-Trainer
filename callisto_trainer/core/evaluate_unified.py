"""Evaluate the unified model: per class, burst versus background, and per file.

The multiclass report answers "did it name the right type". The rollup answers
the operationally more important question, "did it notice a burst at all", by
collapsing the burst classes into one and the rejections (No_Burst, RFI) into
another. A model can score a respectable macro-F1 while confusing Type II with
Type III -- annoying but recoverable -- or while calling real bursts background,
which is not.

Typing is also reported **as it would be at the observed type frequencies**
(see :mod:`callisto_trainer.core.type_priors`), with and without the
checkpoint's type-frequency correction, because a labelled test split holds the
rare types far out of proportion and flatters a model that over-calls them.

Both of those are still crop-level. The file-level report (see
:mod:`callisto_trainer.core.file_eval`) runs the model exactly as the Predict tab
does, over every held-out file, at the calibrated threshold: how many quiet files
it would flag, how many burst files it would find. That is the number to judge a
model by.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from callisto_trainer.core.config import load_config
from callisto_trainer.core.evaluate_type import evaluate_type_model
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.taxonomy import NON_BURST_LABELS, TYPE_III, TYPE_IIIG

LOGGER = get_logger(__name__)

NO_BURST = "No_Burst"


def burst_rollup(metrics: dict[str, Any]) -> dict[str, Any]:
    """Collapse the confusion matrix to burst vs not-a-burst.

    Built from the matrix already computed, so it cannot disagree with the
    multiclass report it accompanies. RFI counts as a correct rejection, the
    same as No_Burst: both mean "not a burst".
    """
    class_names = list(metrics.get("class_names", []))
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=float)
    if matrix.size == 0 or NO_BURST not in class_names:
        return {}

    background = [i for i, name in enumerate(class_names) if name in NON_BURST_LABELS]
    burst_rows = [i for i in range(len(class_names)) if i not in background]

    # Positive = "is a burst of some type".
    tp = float(matrix[np.ix_(burst_rows, burst_rows)].sum())
    fn = float(matrix[np.ix_(burst_rows, background)].sum())
    fp = float(matrix[np.ix_(background, burst_rows)].sum())
    tn = float(matrix[np.ix_(background, background)].sum())

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else float("nan")
    )
    # Of the bursts it did notice, how often was the type also right?
    type_correct = float(np.trace(matrix[np.ix_(burst_rows, burst_rows)]))
    rollup = {
        "burst_recall": recall,
        "burst_precision": precision,
        "no_burst_specificity": specificity,
        "burst_f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "type_accuracy_given_detected": float(type_correct / tp) if tp else float("nan"),
    }
    # Type III versus Type IIIG is a subclass slip, not a typing failure; report
    # typing accuracy with the two counted as one family as well.
    if TYPE_III in class_names and TYPE_IIIG in class_names:
        family = {class_names.index(TYPE_III), class_names.index(TYPE_IIIG)}
        slips = sum(
            matrix[i, j] for i in family for j in family if i != j
        )
        rollup["type_accuracy_given_detected_family"] = (
            float((type_correct + slips) / tp) if tp else float("nan")
        )
        rollup["iii_iiig_confusions"] = int(slips)
    return rollup


def evaluate_unified_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
    files: bool = True,
) -> dict[str, Any]:
    """Run the multiclass evaluation, add the rollup, and judge the split per file."""
    metrics = evaluate_type_model(config, checkpoint_path, split=split)
    rollup = burst_rollup(metrics)
    if not rollup:
        return metrics

    metrics["burst_rollup"] = rollup
    real_world = _real_world_typing(config, checkpoint_path, split)
    if real_world:
        metrics["real_world_typing"] = real_world
    reports_dir = Path(config["paths"]["reports_dir"])
    report_path = reports_dir / f"{split}_type_metrics.json"
    if files:
        metrics["file_level"] = _file_level(config, checkpoint_path, split)
    if report_path.exists():
        # evaluate_type_model already wrote the file; re-write it with the rollup
        # so the Evaluate tab reads one consistent document.
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=_nan_to_none)

    LOGGER.info(
        "Burst rollup: recall=%.4f precision=%.4f specificity=%.4f "
        "(missed %d of %d real bursts); type correct on %.1f%% of detections",
        rollup["burst_recall"],
        rollup["burst_precision"],
        rollup["no_burst_specificity"],
        rollup["false_negatives"],
        rollup["true_positives"] + rollup["false_negatives"],
        100.0 * rollup["type_accuracy_given_detected"],
    )
    return metrics


def _real_world_typing(
    config: dict[str, Any], checkpoint_path: str | Path, split: str
) -> dict[str, Any]:
    """Typing at the observed frequencies: as trained, and as corrected."""
    from callisto_trainer.core.evaluate_type import predict_split
    from callisto_trainer.core.inference import checkpoint_inference_settings
    from callisto_trainer.core.type_priors import real_world_type_report

    priors = checkpoint_inference_settings(checkpoint_path).get("type_priors") or {}
    shares, adjustment = priors.get("class_shares"), priors.get("adjustment")
    if not shares or not adjustment:
        return {}
    y_true, probabilities, class_names, _ = predict_split(config, checkpoint_path, split)
    strength = float(priors.get("strength") or 0.0)
    as_trained = real_world_type_report(y_true, probabilities, class_names, shares)
    corrected = real_world_type_report(
        y_true, probabilities, class_names, shares, adjustment, strength
    )
    LOGGER.info(
        "Typing at observed frequencies (%s): macro-F1 %.3f as trained, %.3f corrected "
        "(strength %.2f)",
        split, as_trained.get("real_world_macro_f1", float("nan")),
        corrected.get("real_world_macro_f1", float("nan")), strength,
    )
    return {"class_shares": shares, "as_trained": as_trained, "corrected": corrected}


def _file_level(config: dict[str, Any], checkpoint_path: str | Path, split: str) -> dict[str, Any]:
    """File-level metrics at the checkpoint's own threshold, or a reason there are none."""
    from callisto_trainer.core.file_eval import (
        file_level_report,
        read_files_manifest,
        score_files,
        write_file_report,
    )
    from callisto_trainer.core.inference import CascadePredictor, checkpoint_inference_settings

    manifest = (config.get("calibration", {}) or {}).get("files_manifest") or str(
        Path(config["paths"]["manifest_path"]).with_name("files.csv")
    )
    if not Path(manifest).exists():
        return {"skipped": "this snapshot has no files.csv; re-export it for file-level metrics"}

    # Judge the model with the region finder its threshold was calibrated for.
    finder = checkpoint_inference_settings(checkpoint_path).get("region_finder") or {}
    predictor = CascadePredictor(
        config,
        unified_checkpoint=checkpoint_path,
        **{
            key: finder[name]
            for key, name in (
                ("region_threshold", "threshold"),
                ("min_area", "min_area"),
                ("max_regions", "max_regions"),
                ("adaptive_threshold", "adaptive"),
                ("hysteresis", "hysteresis"),
            )
            if name in finder
        },
    )
    truths = read_files_manifest(manifest, split=split)
    threshold = predictor.burst_threshold
    scores = score_files(predictor, truths)
    report = file_level_report(scores, threshold if threshold is not None else 0.5)
    report["threshold_calibrated"] = predictor.calibrated_threshold is not None
    write_file_report(report, config["paths"]["reports_dir"], split)
    LOGGER.info(
        "File level (%s, threshold %.3f): %d of %d quiet files flagged (%.1f%%), "
        "%d of %d burst files found, %d stray detection(s)",
        split, report["threshold"], report["false_alarms"], report["quiet_files"],
        100 * (report["false_alarm_rate"] if report["quiet_files"] else 0.0),
        report["bursts_detected"], report["burst_files"], report["stray_detections"],
    )
    return report


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the unified burst model")
    parser.add_argument("--config", required=True, help="Path to the snapshot's config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--no-files", action="store_true", help="Skip the file-level evaluation"
    )
    args = parser.parse_args()

    metrics = evaluate_unified_model(
        load_config(args.config), args.checkpoint, split=args.split, files=not args.no_files
    )
    LOGGER.info(
        "Unified evaluation: accuracy=%.4f macro_f1=%.4f",
        metrics.get("accuracy", float("nan")),
        metrics.get("macro_f1", float("nan")),
    )


if __name__ == "__main__":
    main()
