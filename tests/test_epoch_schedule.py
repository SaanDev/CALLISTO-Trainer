"""The epoch budget must be honoured: a run ends when the schedule ends.

Early stopping used to be unconditional, so a run configured for 40 epochs could
finish at 11 with nothing in the GUI saying why. These tests pin both directions
-- the default runs every epoch, and a configured patience still stops -- because
the difference is invisible from a checkpoint and only shows up in wall-clock
time nobody is watching.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from callisto_trainer.core.config import load_config
from callisto_trainer.core.train_binary import fit
from callisto_trainer.core.train_type import fit_type
from callisto_trainer.store.export import MANIFEST_COLUMNS


def _write_dataset(tmp_path: Path, num_classes: int, per_class: int = 6) -> Path:
    """A minimum viable manifest: a few constant tensors per class, all splits."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    rng = np.random.RandomState(0)
    for label_id in range(num_classes):
        for index in range(per_class):
            # A class-dependent level plus noise: separable, so the monitored
            # score climbs early and then plateaus -- exactly the shape that
            # triggers early stopping.
            tensor = np.clip(
                rng.normal(0.2 + 0.5 * label_id, 0.05, (1, 224, 224)), 0, 1
            ).astype(np.float32)
            path = npz_dir / f"c{label_id}_{index}.npz"
            np.savez(path, spectrum=tensor, label_id=label_id, source_file=str(path),
                     metadata_json=json.dumps({}))
            for split in ("train", "val", "test"):
                row = {column: "" for column in MANIFEST_COLUMNS}
                row.update(
                    {
                        "file_path": str(path),
                        "processed_path": str(path),
                        "label": f"class{label_id}",
                        "label_id": label_id,
                        "split": split,
                    }
                )
                rows.append(row)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _config(tmp_path: Path, epochs: int, patience: int, num_classes: int) -> dict[str, Any]:
    config = load_config()
    config["paths"]["manifest_path"] = str(_write_dataset(tmp_path, num_classes))
    config["paths"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["data"]["classes"] = {f"class{i}": i for i in range(num_classes)}
    config["model"].update(
        {"name": "simple_cnn", "pretrained": False, "use_metadata": False}
    )
    config["performance"].update({"device": "cpu", "mixed_precision": False,
                                  "channels_last": False, "torch_compile": False})
    config["augmentation"]["enabled"] = False
    config["training"].update(
        {"epochs": epochs, "patience": patience, "batch_size": 4, "num_workers": 0,
         "pin_memory": False, "persistent_workers": False, "keep_checkpoints": 1,
         "scheduler": "cosine"}
    )
    return config


# -- the default: the whole schedule runs ---------------------------------


@pytest.mark.parametrize(
    "task, num_classes", [("binary", 2), ("type", 3)], ids=["binary", "type"]
)
def test_default_patience_runs_every_configured_epoch(
    tmp_path: Path, task: str, num_classes: int
) -> None:
    epochs = 5
    config = _config(tmp_path, epochs=epochs, patience=0, num_classes=num_classes)
    result = fit(config) if task == "binary" else fit_type(config)

    assert result["epochs_run"] == epochs, "the run stopped short of its schedule"
    assert result["total_epochs"] == epochs
    assert result["stopped_early"] is False

    history = json.loads(Path(result["history_path"]).read_text(encoding="utf-8"))
    assert [record["epoch"] for record in history] == list(range(1, epochs + 1))


def test_shipped_default_disables_early_stopping() -> None:
    """The config a fresh install trains with must not cut runs short."""
    assert int(load_config()["training"]["patience"]) == 0


def test_generated_snapshot_config_disables_early_stopping(tmp_path: Path) -> None:
    """A snapshot exported from the Dataset tab inherits the same default."""
    from callisto_trainer.store.export import ExportResult, write_training_config

    directory = tmp_path / "snap"
    directory.mkdir()
    result = ExportResult("binary", directory, directory / "manifest.csv")
    write_training_config(
        result, load_config(), tmp_path / "outputs", {"No_Burst": 0, "Burst": 1}, task="binary"
    )

    import yaml

    written = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    assert written["training"]["patience"] == 0


# -- opting back in --------------------------------------------------------


def test_a_configured_patience_still_stops_early(tmp_path: Path) -> None:
    """Early stopping is preserved as an opt-in, not deleted."""
    config = _config(tmp_path, epochs=40, patience=1, num_classes=2)
    # Freeze the model so the score can never improve: with patience 1 the run
    # must end as soon as one epoch fails to beat the first.
    config["training"]["learning_rate"] = 1e-12
    result = fit(config)

    assert result["stopped_early"] is True
    assert result["epochs_run"] < 40
    assert result["total_epochs"] == 40
