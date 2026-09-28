"""Application-level paths and settings.

Separate from :mod:`callisto_trainer.core.config`, which holds the *scientific*
pipeline configuration vendored from the Burst Identifier project. This module
only decides where the app keeps its own files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from callisto_trainer.core.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "configs" / "trainer.yaml"


@dataclass
class AppSettings:
    """Where the app stores data, and how hard it works to stay responsive."""

    project_root: Path = PROJECT_ROOT
    database_path: Path = PROJECT_ROOT / "data" / "annotations.db"
    display_cache_dir: Path = PROJECT_ROOT / "data" / "display_cache"
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    pipeline_config_path: Path | None = None

    # Responsiveness knobs (objective: the GUI must stay fast and smooth).
    decode_workers: int = 3
    prefetch_ahead: int = 4
    memory_cache_mb: int = 512
    disk_cache_gb: int = 8

    pipeline: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline:
            self.pipeline = load_config(self.pipeline_config_path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppSettings":
        """Load settings from YAML, falling back to defaults for missing keys."""
        path = Path(path) if path else DEFAULT_SETTINGS_PATH
        data: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}

        paths = data.get("paths", {}) or {}
        performance = data.get("performance", {}) or {}
        root = Path(paths.get("project_root", PROJECT_ROOT))

        def resolve(key: str, default: Path) -> Path:
            value = paths.get(key)
            if not value:
                return default
            candidate = Path(value)
            return candidate if candidate.is_absolute() else root / candidate

        pipeline_config = paths.get("pipeline_config")
        return cls(
            project_root=root,
            database_path=resolve("database", root / "data" / "annotations.db"),
            display_cache_dir=resolve("display_cache", root / "data" / "display_cache"),
            datasets_dir=resolve("datasets", root / "datasets"),
            outputs_dir=resolve("outputs", root / "outputs"),
            pipeline_config_path=(
                Path(pipeline_config)
                if pipeline_config and Path(pipeline_config).is_absolute()
                else (root / pipeline_config if pipeline_config else None)
            ),
            decode_workers=int(performance.get("decode_workers", 3)),
            prefetch_ahead=int(performance.get("prefetch_ahead", 4)),
            memory_cache_mb=int(performance.get("memory_cache_mb", 512)),
            disk_cache_gb=int(performance.get("disk_cache_gb", 8)),
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.database_path.parent,
            self.display_cache_dir,
            self.datasets_dir,
            self.outputs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def memory_cache_bytes(self) -> int:
        return self.memory_cache_mb * 1024**2

    @property
    def disk_cache_bytes(self) -> int:
        return self.disk_cache_gb * 1024**3
