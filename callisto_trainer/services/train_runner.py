"""Run training and evaluation as child processes and stream their progress.

Training never runs inside the GUI process. Three reasons, in order of how
painful they are: mixing a CUDA context with a Qt event loop is a well-known
source of hangs; an out-of-memory crash would otherwise take the window and the
operator's place in the queue with it; and keeping it separate means labelling
can continue while a model trains.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.progress import parse_progress

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingJob:
    """One training or evaluation invocation."""

    task: str  # "type" | "binary"
    config_path: Path
    module: str
    extra_args: tuple[str, ...] = ()

    TRAIN_MODULES = {
        "unified": "callisto_trainer.core.train_unified",
        "type": "callisto_trainer.core.train_type",
        "binary": "callisto_trainer.core.train_binary",
    }
    EVALUATE_MODULES = {
        "unified": "callisto_trainer.core.evaluate_unified",
        "type": "callisto_trainer.core.evaluate_type",
        "binary": "callisto_trainer.core.evaluate_binary",
    }

    @classmethod
    def train(cls, task: str, config_path: str | Path) -> "TrainingJob":
        module = cls.TRAIN_MODULES.get(task, cls.TRAIN_MODULES["binary"])
        return cls(task=task, config_path=Path(config_path), module=module)

    @classmethod
    def evaluate(
        cls, task: str, config_path: str | Path, checkpoint: str | Path, split: str = "test"
    ) -> "TrainingJob":
        module = cls.EVALUATE_MODULES.get(task, cls.EVALUATE_MODULES["binary"])
        return cls(
            task=task,
            config_path=Path(config_path),
            module=module,
            extra_args=("--checkpoint", str(checkpoint), "--split", split),
        )

    def arguments(self) -> list[str]:
        return ["-u", "-m", self.module, "--config", str(self.config_path), *self.extra_args]


class TrainingRunner(QObject):
    """Owns at most one child process and re-emits its output."""

    started = Signal(object)          # TrainingJob
    progress = Signal(dict)           # parsed progress record
    output = Signal(str)              # raw log line
    finished = Signal(int, object)    # exit code, TrainingJob
    failed = Signal(str)

    def __init__(self, project_root: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self._process: QProcess | None = None
        self._job: TrainingJob | None = None
        self._buffer = ""

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def job(self) -> TrainingJob | None:
        return self._job

    def start(self, job: TrainingJob) -> bool:
        if self.is_running:
            self.failed.emit("A run is already in progress.")
            return False
        if not job.config_path.exists():
            self.failed.emit(f"Config not found: {job.config_path}")
            return False

        process = QProcess(self)
        process.setWorkingDirectory(str(self.project_root))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        process.setProcessEnvironment(self._child_environment())

        process.readyReadStandardOutput.connect(self._drain)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self._process = process
        self._job = job
        self._buffer = ""

        LOGGER.info("Launching %s %s", sys.executable, " ".join(job.arguments()))
        process.start(sys.executable, job.arguments())
        if not process.waitForStarted(10000):
            self.failed.emit("Could not start the training process.")
            self._process = None
            return False

        self.started.emit(job)
        return True

    def _child_environment(self) -> QProcessEnvironment:
        """The training process's environment: the full system one, plus our bits.

        It must start from ``systemEnvironment()``. ``QProcess.processEnvironment()``
        returns an *empty* object when none has been set, so building on it and
        calling ``setProcessEnvironment`` replaces the child's entire environment
        rather than extending it -- dropping PATH, TEMP, the CUDA variables and
        USERNAME. Torch reads USERNAME via ``getpass.getuser()`` when choosing its
        cache directory, and without it falls through to the Unix-only ``pwd``
        module, so importing torchvision died with a misleading
        "torchvision is required" error.
        """
        environment = QProcessEnvironment.systemEnvironment()
        # Prepend rather than overwrite: a user may already rely on PYTHONPATH.
        existing = environment.value("PYTHONPATH", "")
        root = str(self.project_root)
        environment.insert(
            "PYTHONPATH", f"{root}{os.pathsep}{existing}" if existing else root
        )
        environment.insert("PYTHONUNBUFFERED", "1")
        return environment

    def stop(self) -> None:
        """Ask the run to stop; the last checkpoint is already on disk."""
        if not self.is_running or self._process is None:
            return
        self._process.terminate()
        if not self._process.waitForFinished(5000):
            self._process.kill()

    # -- output handling ---------------------------------------------------

    def _drain(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buffer += chunk
        # Keep any trailing partial line for the next chunk.
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            record = parse_progress(line)
            if record is not None:
                self.progress.emit(record)
            else:
                self.output.emit(line)

    def _on_finished(self, exit_code: int, _status) -> None:
        if self._buffer.strip():
            record = parse_progress(self._buffer.strip())
            if record is not None:
                self.progress.emit(record)
            else:
                self.output.emit(self._buffer.strip())
        self._buffer = ""
        job, self._job, self._process = self._job, None, None
        self.finished.emit(int(exit_code), job)

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.failed.emit(
                "The training process failed to start. Check that the virtual "
                "environment has torch installed."
            )
