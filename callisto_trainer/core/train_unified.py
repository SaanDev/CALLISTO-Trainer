"""Train the unified burst model: one head over regions, then calibrate it.

Classes are background (``No_Burst``), ``RFI`` (named automatically at export)
and the burst types, and the
input is a region: two views of it plus region features (see
:mod:`callisto_trainer.core.region_inputs`). A single softmax answers every
question the two-model cascade used to need: whether a region is a burst at all,
and which type it is. Applied over the candidate regions of a file it also
yields *where*.

The training loop itself is the multiclass one in
:mod:`callisto_trainer.core.train_type` -- same backbone handling, same
checkpointing -- with the unified task's own class weighting and selection
metric chosen by the snapshot's config.

After training, two things are calibrated on the validation split:

* the **burst threshold**, on the validation *files*: the most bursts found
  while no more than the configured share of quiet files is flagged (see
  :mod:`callisto_trainer.core.file_eval`). Crop-level scores cannot show how
  often a file with a dozen candidate regions is wrongly called a burst; this
  step measures it directly and sets the operating point from it;
* the **type-frequency correction**, on the validation *regions*: how strongly
  to shift the type probabilities from the training mix toward how often each
  type really occurs (see :mod:`callisto_trainer.core.type_priors`).
"""

from __future__ import annotations

import argparse

from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.progress import emit_progress
from callisto_trainer.core.train_type import fit_type

LOGGER = get_logger(__name__)


def calibrate_after_training(config: dict, checkpoint_path: str) -> dict | None:
    """Calibrate ``checkpoint_path`` and report progress; never fails the run."""
    from callisto_trainer.core.file_eval import calibrate_checkpoint

    def report(index: int, total: int, name: str) -> bool:
        if index % 10 == 0 or index == total - 1:
            emit_progress(
                {"event": "calibrating", "task": "unified", "index": index,
                 "total": total, "file": name}
            )
        return True

    try:
        record = calibrate_checkpoint(config, checkpoint_path, progress=report)
    except Exception as exc:
        # The trained weights are already saved; a calibration failure leaves
        # the model usable on its argmax decision, and says why.
        LOGGER.exception("Calibration failed; the model keeps its argmax decision")
        emit_progress({"event": "calibrated", "task": "unified", "error": repr(exc)})
        return None
    emit_progress({"event": "calibrated", "task": "unified", **record})
    return record


def calibrate_types_after_training(config: dict, checkpoint_path: str) -> dict | None:
    """Fit the type-frequency correction; never fails the run."""
    from callisto_trainer.core.file_eval import calibrate_type_priors

    try:
        record = calibrate_type_priors(config, checkpoint_path, split="val")
    except Exception as exc:
        LOGGER.exception("Type-frequency calibration failed; types are decided as trained")
        emit_progress({"event": "type_priors", "task": "unified", "error": repr(exc)})
        return None
    emit_progress(
        {
            "event": "type_priors",
            "task": "unified",
            "strength": record["strength"],
            "training_shares": record["training_shares"],
            "class_shares": record["class_shares"],
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified burst model")
    parser.add_argument("--config", required=True, help="Path to the snapshot's config.yaml")
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Skip tuning the burst threshold on the validation files",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    classes = config["data"]["classes"]
    if "No_Burst" not in classes:
        raise ValueError(
            "The unified model expects a No_Burst class in data.classes; got "
            f"{sorted(classes)}. Export a 'unified' snapshot from the Dataset tab."
        )

    result = fit_type(config)
    LOGGER.info("Unified training complete: %s", result)
    if not args.no_calibration and config.get("calibration"):
        record = calibrate_after_training(config, result["best_alias"])
        types = calibrate_types_after_training(config, result["best_alias"])
        epoch_copy = result.get("best_checkpoint")
        if (record or types) and epoch_copy and epoch_copy != result["best_alias"]:
            from callisto_trainer.core.file_eval import copy_calibration

            try:
                copy_calibration(result["best_alias"], epoch_copy)
            except OSError as exc:
                LOGGER.warning("Could not copy the calibration to %s: %r", epoch_copy, exc)


if __name__ == "__main__":
    main()
