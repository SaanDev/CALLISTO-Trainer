"""Listing and removing trained models.

These are destructive operations on real directories, so the safety properties
matter as much as the happy path: nothing outside the configured directories may
be touched, pruning must never remove the model you selected, and archiving must
leave a checkpoint that still loads and predicts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.models.model_factory import create_model  # noqa: E402
from callisto_trainer.services.model_registry import (  # noqa: E402
    TASK_TO_SNAPSHOT_KIND,
    archive_run,
    delete_run,
    format_bytes,
    list_runs,
    prune_epoch_copies,
    total_bytes,
)


def _make_run(
    root: Path,
    task: str = "unified",
    run_id: str = "20260728_120000",
    epoch_copies: int = 3,
    with_snapshot: bool = True,
    with_optimizer: bool = True,
    epochs: int = 12,
) -> Path:
    """A realistic run directory: checkpoints, history, reports and a snapshot."""
    outputs = root / "outputs" / f"{task}_{run_id}"
    checkpoints = outputs / "checkpoints"
    checkpoints.mkdir(parents=True)
    (outputs / "reports").mkdir()
    (outputs / "reports" / "test_type_metrics.json").write_text("{}", encoding="utf-8")

    config = load_config()
    config["data"]["classes"] = {"No_Burst": 0, "Type II": 1, "Type III": 2, "Other": 3}
    config["model"].update({"name": "simple_cnn", "num_classes": 4, "use_metadata": False})
    model = create_model("simple_cnn", in_channels=1, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters())
    if with_optimizer:
        # Take one real step: an AdamW that never stepped has an empty "state",
        # so there would be nothing for archiving to strip and the test would
        # pass vacuously.
        loss = model(torch.rand(2, 1, 224, 224)).sum()
        loss.backward()
        optimizer.step()

    payload = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if with_optimizer else {},
        "config": config,
        "metrics": {"val": {"macro_f1": 0.81}},
    }
    for name in ("best.pt", "last.pt"):
        torch.save(payload, checkpoints / name)
    for index in range(epoch_copies):
        torch.save(payload, checkpoints / f"best_epoch_{index:03d}_20260728_1200{index:02d}_1.pt")

    history = [
        {"epoch": i + 1, "train": {"loss": 1.0}, "val": {"macro_f1": 0.5 + i * 0.02}}
        for i in range(epochs)
    ]
    (checkpoints / "training_history.json").write_text(json.dumps(history), encoding="utf-8")

    if with_snapshot:
        snapshot = root / "datasets" / TASK_TO_SNAPSHOT_KIND[task] / run_id
        (snapshot / "npz").mkdir(parents=True)
        (snapshot / "manifest.csv").write_text("file_path\n", encoding="utf-8")
        (snapshot / "snapshot.json").write_text(
            json.dumps({"kind": "unified", "samples": 2089}), encoding="utf-8"
        )
        (snapshot / "npz" / "a.npz").write_bytes(b"x" * 4096)
    return outputs


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _make_run(tmp_path, task="unified", run_id="20260728_120000")
    _make_run(tmp_path, task="type", run_id="20260727_090000", epoch_copies=1)
    _make_run(tmp_path, task="binary", run_id="20260726_080000", with_snapshot=False)
    return tmp_path


# -- listing ---------------------------------------------------------------


def test_lists_every_run_newest_first(project: Path) -> None:
    runs = list_runs(project / "outputs", project / "datasets")

    assert [run.task for run in runs] == ["unified", "type", "binary"]
    assert runs[0].run_id == "20260728_120000"
    assert all(run.size_bytes > 0 for run in runs)


def test_reports_score_epochs_and_checkpoints(project: Path) -> None:
    run = list_runs(project / "outputs", project / "datasets")[0]

    assert run.epochs_trained == 12
    assert run.best_score == pytest.approx(0.72, abs=0.01)
    assert run.monitor == "macro_f1"
    assert run.has_best
    assert run.checkpoint_count == 5  # best + last + 3 epoch copies
    assert run.epoch_copy_count == 3
    assert run.trained_at.startswith("2026-07-28")


def test_links_the_dataset_snapshot(project: Path) -> None:
    runs = list_runs(project / "outputs", project / "datasets")
    unified = runs[0]

    assert unified.snapshot_dir is not None
    assert unified.snapshot_dir.name == "20260728_120000"
    assert unified.snapshot_samples == 2089
    assert unified.snapshot_bytes > 0

    binary = next(run for run in runs if run.task == "binary")
    assert binary.snapshot_dir is None, "a missing snapshot must not be invented"


def test_ignores_unrelated_directories(project: Path) -> None:
    (project / "outputs" / "some_other_thing").mkdir()
    (project / "outputs" / "notarun").mkdir()
    assert len(list_runs(project / "outputs")) == 3


def test_missing_outputs_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "nope") == []


def test_run_without_history_still_lists(tmp_path: Path) -> None:
    outputs = _make_run(tmp_path)
    (outputs / "checkpoints" / "training_history.json").unlink()

    run = list_runs(tmp_path / "outputs")[0]
    assert run.best_score is None
    assert run.describe_score() == "-"


# -- pruning ---------------------------------------------------------------


def test_pruning_removes_only_the_epoch_copies(project: Path) -> None:
    outputs = project / "outputs"
    run = list_runs(outputs, project / "datasets")[0]
    before = run.size_bytes

    freed = prune_epoch_copies(run, outputs)

    checkpoints = run.directory / "checkpoints"
    assert (checkpoints / "best.pt").exists(), "the selected model must survive"
    assert (checkpoints / "last.pt").exists(), "resuming must stay possible"
    assert list(checkpoints.glob("*_epoch_*.pt")) == []
    assert freed > 0

    after = list_runs(outputs, project / "datasets")[0]
    assert after.size_bytes < before
    assert after.epoch_copy_count == 0


def test_pruning_twice_is_harmless(project: Path) -> None:
    outputs = project / "outputs"
    run = list_runs(outputs)[0]
    prune_epoch_copies(run, outputs)
    assert prune_epoch_copies(list_runs(outputs)[0], outputs) == 0


# -- archiving -------------------------------------------------------------


def test_archiving_strips_optimizer_state_but_keeps_the_model(project: Path) -> None:
    outputs = project / "outputs"
    run = list_runs(outputs)[0]
    before = (run.directory / "checkpoints" / "best.pt").stat().st_size

    freed = archive_run(run, outputs)

    checkpoint = torch.load(
        run.directory / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    assert not checkpoint["optimizer_state"], "optimizer state should be gone"
    assert checkpoint["model_state"], "weights must remain"
    assert checkpoint["config"], "the input contract must remain"
    assert checkpoint.get("archived") is True
    assert freed > 0
    assert (run.directory / "checkpoints" / "best.pt").stat().st_size < before


def test_an_archived_checkpoint_still_loads_for_inference(project: Path) -> None:
    """The whole point: archiving must not cost you the model."""
    from callisto_trainer.core.predict import load_type_model_for_inference

    outputs = project / "outputs"
    run = list_runs(outputs)[0]
    archive_run(run, outputs)

    model, _config, device, class_names = load_type_model_for_inference(
        run.directory / "checkpoints" / "best.pt", load_config()
    )
    assert class_names == ["No_Burst", "Type II", "Type III", "Other"]
    with torch.no_grad():
        output = model(torch.rand(1, 1, 224, 224).to(device))
    assert output.shape == (1, 4)


def test_archiving_twice_is_harmless(project: Path) -> None:
    outputs = project / "outputs"
    assert archive_run(list_runs(outputs)[0], outputs) > 0
    assert archive_run(list_runs(outputs)[0], outputs) == 0, "nothing left to strip"


def test_archiving_a_run_with_no_optimizer_state_is_a_no_op(tmp_path: Path) -> None:
    """Never rewrite a checkpoint when there is nothing to gain."""
    _make_run(tmp_path, with_optimizer=False)
    outputs = tmp_path / "outputs"
    run = list_runs(outputs)[0]
    before = (run.directory / "checkpoints" / "best.pt").stat().st_size

    assert archive_run(run, outputs) == 0
    assert (run.directory / "checkpoints" / "best.pt").stat().st_size == before


def test_archiving_leaves_no_temporary_file_behind(project: Path) -> None:
    outputs = project / "outputs"
    run = list_runs(outputs)[0]
    archive_run(run, outputs)
    assert list((run.directory / "checkpoints").glob("*.tmp")) == []


# -- deletion --------------------------------------------------------------


def test_deleting_a_run_keeps_its_snapshot_by_default(project: Path) -> None:
    outputs, datasets = project / "outputs", project / "datasets"
    run = list_runs(outputs, datasets)[0]
    snapshot = run.snapshot_dir

    delete_run(run, outputs, datasets, include_snapshot=False)

    assert not run.directory.exists()
    assert snapshot.exists(), "the data must survive so the model can be retrained"
    assert len(list_runs(outputs, datasets)) == 2


def test_deleting_with_the_snapshot_removes_both(project: Path) -> None:
    outputs, datasets = project / "outputs", project / "datasets"
    run = list_runs(outputs, datasets)[0]
    snapshot = run.snapshot_dir

    freed = delete_run(run, outputs, datasets, include_snapshot=True)

    assert not run.directory.exists()
    assert not snapshot.exists()
    assert freed >= run.size_bytes + run.snapshot_bytes


def test_deleting_does_not_touch_other_runs(project: Path) -> None:
    outputs, datasets = project / "outputs", project / "datasets"
    runs = list_runs(outputs, datasets)
    delete_run(runs[0], outputs, datasets, include_snapshot=True)

    remaining = list_runs(outputs, datasets)
    assert {run.task for run in remaining} == {"type", "binary"}
    assert all(run.directory.exists() for run in remaining)


def test_refuses_to_delete_outside_the_managed_directories(project: Path, tmp_path: Path) -> None:
    """A path that escapes the project must be refused, not trusted."""
    outsider = tmp_path / "somewhere_else"
    outsider.mkdir()
    run = list_runs(project / "outputs", project / "datasets")[0]
    run.directory = outsider

    with pytest.raises(ValueError, match="Refusing to delete"):
        delete_run(run, project / "outputs")
    assert outsider.exists()


def test_refuses_a_snapshot_outside_the_datasets_directory(project: Path, tmp_path: Path) -> None:
    outsider = tmp_path / "elsewhere"
    outsider.mkdir()
    run = list_runs(project / "outputs", project / "datasets")[0]
    run.snapshot_dir = outsider

    with pytest.raises(ValueError, match="Refusing to delete"):
        delete_run(run, project / "outputs", project / "datasets", include_snapshot=True)
    assert outsider.exists()


# -- reporting -------------------------------------------------------------


def test_reclaimable_covers_epoch_copies_and_optimizer_state(project: Path) -> None:
    run = list_runs(project / "outputs")[0]
    assert run.reclaimable_bytes == run.epoch_copy_bytes + run.optimizer_state_bytes
    assert run.reclaimable_bytes < run.size_bytes, "some of the run is the model itself"


def test_total_bytes_can_include_or_exclude_snapshots(project: Path) -> None:
    runs = list_runs(project / "outputs", project / "datasets")
    assert total_bytes(runs, include_snapshots=True) > total_bytes(runs, include_snapshots=False)


def test_byte_formatting_is_readable() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048).endswith("KB")
    assert "MB" in format_bytes(5 * 1024**2)
    assert "GB" in format_bytes(3 * 1024**3)


# -- dialog ----------------------------------------------------------------


@pytest.fixture
def manager(project: Path):
    from PySide6.QtWidgets import QApplication

    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.model_manager import ModelManagerDialog

    QApplication.instance() or QApplication([])
    settings = AppSettings(
        project_root=project,
        database_path=project / "data" / "annotations.db",
        display_cache_dir=project / "cache",
        datasets_dir=project / "datasets",
        outputs_dir=project / "outputs",
    )
    dialog = ModelManagerDialog(settings)
    yield dialog
    dialog.close()


def test_dialog_lists_the_runs(manager) -> None:
    assert manager.table.rowCount() == 3
    assert "reclaimable" in manager.summary.text()


def test_dialog_actions_need_a_selection(manager) -> None:
    assert not manager.delete_button.isEnabled()
    assert not manager.prune_button.isEnabled()

    manager.table.selectRow(0)
    assert manager.delete_button.isEnabled()
    assert manager.prune_button.isEnabled()
    assert "Free space (" in manager.prune_button.text()


def test_dialog_disables_snapshot_deletion_when_there_is_none(manager) -> None:
    rows = {
        manager.table.item(row, 0).text(): row for row in range(manager.table.rowCount())
    }
    manager.table.selectRow(rows["binary_20260726_080000"])
    assert manager.delete_button.isEnabled()
    assert not manager.delete_all_button.isEnabled(), "that run has no snapshot"


def test_dialog_selection_maps_to_runs(manager) -> None:
    manager.table.selectAll()
    assert len(manager.selected_runs()) == 3


def test_dialog_refresh_picks_up_external_deletion(manager, project: Path) -> None:
    import shutil

    shutil.rmtree(project / "outputs" / "binary_20260726_080000")
    manager.refresh()
    assert manager.table.rowCount() == 2
