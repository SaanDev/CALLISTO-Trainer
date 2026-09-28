"""Training subprocess plumbing: progress protocol, runner and tab guardrails."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.core.progress import (  # noqa: E402
    PROGRESS_PREFIX,
    emit_progress,
    parse_progress,
)
from callisto_trainer.services.train_runner import TrainingJob, TrainingRunner  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


# -- progress protocol -----------------------------------------------------


def test_progress_round_trip(capsys) -> None:
    emit_progress({"event": "epoch", "epoch": 3, "val_loss": 0.25})
    captured = capsys.readouterr().out.strip()

    assert captured.startswith(PROGRESS_PREFIX)
    assert parse_progress(captured) == {"event": "epoch", "epoch": 3, "val_loss": 0.25}


def test_ordinary_lines_are_not_progress() -> None:
    assert parse_progress("2026-07-27 | INFO | training started") is None
    assert parse_progress("") is None
    assert parse_progress(PROGRESS_PREFIX + "{not json") is None
    assert parse_progress(PROGRESS_PREFIX + "[1, 2, 3]") is None, "only objects are records"


def test_emit_never_raises_on_unserialisable_values(capsys) -> None:
    emit_progress({"path": Path("C:/x"), "array": object()})
    assert parse_progress(capsys.readouterr().out.strip()) is not None


# -- job construction ------------------------------------------------------


def test_train_job_targets_the_right_module(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")

    type_job = TrainingJob.train("type", config)
    binary_job = TrainingJob.train("binary", config)

    assert type_job.module.endswith("train_type")
    assert binary_job.module.endswith("train_binary")
    assert type_job.arguments()[:3] == ["-u", "-m", type_job.module]
    assert "--config" in type_job.arguments()


def test_evaluate_job_passes_checkpoint_and_split(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    job = TrainingJob.evaluate("type", config, tmp_path / "best.pt", split="val")

    arguments = job.arguments()
    assert job.module.endswith("evaluate_type")
    assert "--checkpoint" in arguments
    assert arguments[arguments.index("--split") + 1] == "val"


# -- runner ----------------------------------------------------------------


def _run_fake(qapp, tmp_path: Path, epochs: int = 3, fail: bool = False):
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")

    extra = ("--epochs", str(epochs)) + (("--fail",) if fail else ())
    job = TrainingJob(
        task="type", config_path=config, module="_fake_trainer", extra_args=extra
    )

    runner = TrainingRunner(TESTS_DIR)
    records: list[dict] = []
    lines: list[str] = []
    done: list[int] = []
    runner.progress.connect(records.append)
    runner.output.connect(lines.append)
    runner.finished.connect(lambda code, _job: done.append(code))

    assert runner.start(job)

    deadline = time.monotonic() + 30
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert done, "the fake trainer never finished"
    return runner, records, lines, done[0]


def test_runner_streams_progress_and_log_separately(qapp, tmp_path: Path) -> None:
    _runner, records, lines, exit_code = _run_fake(qapp, tmp_path, epochs=3)

    assert exit_code == 0
    events = [record["event"] for record in records]
    assert events == ["start", "epoch", "epoch", "epoch", "finished"]

    epochs = [record for record in records if record["event"] == "epoch"]
    assert [record["epoch"] for record in epochs] == [1, 2, 3]
    assert epochs[0]["val_loss"] > epochs[-1]["val_loss"]

    # Ordinary output must reach the log view, not the progress channel.
    assert any("ordinary log line" in line for line in lines)
    assert not any(line.startswith(PROGRESS_PREFIX) for line in lines)


def test_runner_reports_a_nonzero_exit(qapp, tmp_path: Path) -> None:
    _runner, records, _lines, exit_code = _run_fake(qapp, tmp_path, epochs=2, fail=True)

    assert exit_code == 3
    assert [record["event"] for record in records] == ["start", "epoch", "epoch"]


def test_runner_refuses_a_second_concurrent_run(qapp, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    job = TrainingJob(
        task="type", config_path=config, module="_fake_trainer", extra_args=("--epochs", "20")
    )
    runner = TrainingRunner(TESTS_DIR)
    failures: list[str] = []
    runner.failed.connect(failures.append)

    assert runner.start(job)
    assert not runner.start(job)
    assert failures and "already in progress" in failures[0]

    runner.stop()
    deadline = time.monotonic() + 10
    while runner.is_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not runner.is_running


def test_child_inherits_the_full_system_environment(qapp, tmp_path: Path) -> None:
    """Regression: the training process must not start with a stripped environment.

    ``QProcess.processEnvironment()`` returns an empty object when none has been
    set, so building on it wiped PATH, TEMP and USERNAME from the child. Torch
    calls ``getpass.getuser()`` to pick its cache directory, which without
    USERNAME falls through to the Unix-only ``pwd`` module -- surfacing as a
    misleading "torchvision is required" ImportError partway into training.
    """
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    job = TrainingJob(
        task="type", config_path=config, module="_fake_trainer", extra_args=("--report-env",)
    )

    runner = TrainingRunner(TESTS_DIR)
    records: list[dict] = []
    done: list[int] = []
    runner.progress.connect(records.append)
    runner.finished.connect(lambda code, _job: done.append(code))
    assert runner.start(job)

    deadline = time.monotonic() + 30
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert done and done[0] == 0
    report = next(record for record in records if record["event"] == "env")

    assert not str(report["user"]).startswith("FAILED"), (
        f"getpass.getuser() failed in the child: {report['user']}"
    )
    assert report["has_username"], "USERNAME/USER missing from the child environment"
    assert report["has_path"], "PATH missing from the child environment"
    assert report["var_count"] > 5, (
        f"child saw only {report['var_count']} variables; the environment was replaced, "
        "not extended"
    )
    assert report["unbuffered"] == "1"
    assert str(TESTS_DIR) in report["pythonpath"]


def test_existing_pythonpath_is_preserved(qapp, tmp_path: Path, monkeypatch) -> None:
    """A PYTHONPATH the user already relies on must be prepended to, not replaced."""
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "someones_libs"))

    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    job = TrainingJob(
        task="type", config_path=config, module="_fake_trainer", extra_args=("--report-env",)
    )

    runner = TrainingRunner(TESTS_DIR)
    records: list[dict] = []
    done: list[int] = []
    runner.progress.connect(records.append)
    runner.finished.connect(lambda code, _job: done.append(code))
    assert runner.start(job)

    deadline = time.monotonic() + 30
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    report = next(record for record in records if record["event"] == "env")
    assert str(TESTS_DIR) in report["pythonpath"]
    assert "someones_libs" in report["pythonpath"]


def test_runner_rejects_a_missing_config(qapp, tmp_path: Path) -> None:
    runner = TrainingRunner(TESTS_DIR)
    failures: list[str] = []
    runner.failed.connect(failures.append)

    assert not runner.start(TrainingJob.train("type", tmp_path / "nope.yaml"))
    assert failures and "not found" in failures[0].lower()


# -- train tab guardrails --------------------------------------------------


@pytest.fixture
def train_tab(qapp, tmp_path: Path):
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.train_tab import TrainTab

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    settings.ensure_directories()
    tab = TrainTab(settings)
    yield tab
    tab.shutdown()


def test_thin_class_blocks_training(train_tab) -> None:
    """An empty split would crash class weighting; catch it before launching."""
    info = {
        "samples": 40,
        "class_split_counts": {
            "Type II": {"train": 20, "val": 5, "test": 5},
            "Type III": {"train": 8, "val": 2, "test": 0},
        },
    }
    problems = train_tab._blocking_problems(info)
    assert problems
    assert any("Type III" in problem and "test" in problem for problem in problems)


def test_healthy_snapshot_has_no_blocking_problems(train_tab) -> None:
    info = {
        "samples": 90,
        "class_split_counts": {
            name: {"train": 21, "val": 4, "test": 5} for name in ("Type II", "Type III", "Other")
        },
        "event_leakage": 0,
    }
    assert train_tab._blocking_problems(info) == []


def test_event_leakage_blocks_training(train_tab) -> None:
    info = {
        "samples": 90,
        "class_split_counts": {
            name: {"train": 21, "val": 4, "test": 5} for name in ("Type II", "Type III", "Other")
        },
        "event_leakage": 3,
    }
    problems = train_tab._blocking_problems(info)
    assert problems and "inflated" in problems[0]


def test_empty_snapshot_blocks_training(train_tab) -> None:
    assert train_tab._blocking_problems({"samples": 0}) != []


def test_no_snapshot_disables_the_start_button(train_tab) -> None:
    train_tab.refresh_snapshots()
    assert train_tab.current_snapshot() is None
    assert not train_tab.start_button.isEnabled()
