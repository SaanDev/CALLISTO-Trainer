"""Configuration loading helpers."""

# NOTE: Vendored from H:\Burst Identifier (src/utils/config.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "e_callisto_burst_ml"},
    "paths": {
        "raw_dir_candidates": ["data/raw", "."],
        "manifest_path": "data/manifest.csv",
        "processed_dir": "data/processed",
        "checkpoint_dir": "outputs/checkpoints",
        "figures_dir": "outputs/figures",
        "reports_dir": "outputs/reports",
    },
    "data": {
        "classes": {"No_Burst": 0, "Burst": 1},
        "target_shape": [224, 224],
        "drop_missing_processed": True,
        # When True, skip the per-file existence scan at startup (trust the
        # manifest). Set this after a clean preprocessing run to remove ~500k
        # filesystem stat() calls before each training run.
        "assume_processed_complete": False,
        "expected_counts": {},
        # When False (default), a mismatch between expected_counts and the actual
        # raw file counts only warns (the manifest is still written). Set True to
        # make it a hard error.
        "strict_expected_counts": False,
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "seed": 42,
            # Keep every recording of one solar event in a single split to
            # avoid train/test leakage across stations. Set false for the old
            # per-file behaviour.
            "group_by_event": True,
        },
    },
    "preprocessing": {
        "background_method": "plotutil_median_db",
        "normalization": "db_window",
        "db_vmin": -1.0,
        "db_vmax": 8.0,
        "normalization_clip": 8.0,
        "epsilon": 1.0e-6,
        "overwrite": False,
        "device": "auto",
        # "auto" uses all logical CPUs; preprocessing is CPU/NumPy + I/O bound,
        # so parallel workers give a near-linear speedup over the old serial (0).
        "num_workers": "auto",
        "save_compressed": False,
    },
    # Geometry for turning a drawn box into a training tensor. See core/crops.py.
    # These are all no-ops by default: the crop is exactly the region the operator
    # boxed, so the tensor previewed while labelling is the tensor the model gets.
    # Raising context_margin (fraction of the box added to every side) or the
    # minimums (floor size in pixels) makes every crop cover more than was drawn,
    # so only do it if the whole corpus is re-exported with the same values.
    "crops": {
        "context_margin": 0.0,
        "min_rows": 1,
        "min_cols": 1,
        "target_shape": [224, 224],
    },
    # Widths are pixels on the fixed 224x224 crop, so they read as percentages of
    # the frame. These are deliberately stronger than the upstream whole-file
    # values: a crop dataset has few distinct examples per class, and weak
    # augmentation lets the network memorize them within ~50 epochs.
    "augmentation": {
        "enabled": True,
        "probability": 0.90,
        "time_shift_max": 24,        # ~10% of the frame
        "intensity_scale_min": 0.85,
        "intensity_scale_max": 1.15,
        "noise_std": 0.05,
        "freq_mask_max": 24,         # ~10%, applied num_freq_masks times
        "time_mask_max": 32,         # ~14%, applied num_time_masks times
        "num_freq_masks": 2,
        "num_time_masks": 2,
        "mask_value": 0.0,
    },
    "performance": {
        "device": "auto",
        "mixed_precision": True,
        "amp_dtype": "float16",
        # Off: measured on an RTX 5060 (torch 2.12, CUDA 13), channels_last made
        # ResNet training 5-8x slower (ResNet18 243 vs 1402 images/s, ResNet50
        # 58 vs 473) and ConvNeXt no faster. Single-channel spectrogram inputs
        # do not get the tensor-core layout win it was meant for.
        "channels_last": False,
        "torch_compile": False,
        "cudnn_benchmark": True,
        "deterministic": False,
        "matmul_precision": "high",
    },
    "model": {
        "name": "simple_cnn",
        "in_channels": 1,
        "dropout": 0.25,
        # Load ImageNet-pretrained weights for torchvision backbones (ResNet/
        # EfficientNet/MobileNet). Ignored by simple_cnn. Big accuracy + faster
        # convergence; requires internet on first download.
        "pretrained": False,
        # Multi-input conditioning on station / frequency range / date.
        "use_metadata": True,
        "station_emb_dim": 8,
    },
    "training": {
        "batch_size": 64,
        # Concrete, safe default. Avoid "all cores": under Windows spawn each
        # worker re-imports torch and copies the split row list, which can
        # exhaust RAM and stall before the GPU starts. "auto" is capped (see
        # dataset._auto_worker_count); 0 loads in the main process.
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "epochs": 40,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        # Early-stop patience, in epochs without an improved validation score.
        # 0 disables it, so `epochs` above runs in full -- the default, because a
        # flat stretch in the monitored metric is common on imbalanced burst
        # classes and is not reliable evidence that training has converged.
        # best.pt tracks the best epoch either way, so the extra epochs can only
        # help. Set a positive number to stop on a plateau again.
        "patience": 0,
        "seed": 42,
        # Softens the one-hot targets. Cross-entropy on hard labels rewards ever
        # larger logits on samples already classified correctly, which is how a
        # run ends up with training loss at ~0 and a validation loss that climbs
        # while accuracy stays flat: the model is not getting more wrong, it is
        # getting more confident about the few it gets wrong. Smoothing removes
        # that incentive and keeps the softmax usable as a confidence.
        "label_smoothing": 0.05,
        "threshold": 0.5,
        "auto_threshold": True,
        "threshold_metric": "f1",
        "monitor": "pr_auc",
        # "plateau" (ReduceLROnPlateau on the monitored metric) or "cosine".
        "scheduler": "plateau",
        "keep_checkpoints": 3,
        "class_balance": {"strategy": "auto_pos_weight"},
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return a new dictionary."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and merge it with project defaults."""
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    merged = deep_update(DEFAULT_CONFIG, loaded)

    # ``data.classes`` is a mapping, so ``deep_update`` would MERGE it with the
    # default {No_Burst, Burst}. That is wrong for a config that defines its own
    # label set (e.g. the burst-type classes): the defaults would linger and
    # collide on label ids. When a config explicitly provides its own classes,
    # treat them as a full replacement of the default set.
    if isinstance(loaded.get("data"), dict) and "classes" in loaded["data"]:
        merged["data"]["classes"] = dict(loaded["data"]["classes"])

    return merged


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to ``base_dir`` or the current working directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is not None:
        return Path(base_dir).resolve() / path
    return path.resolve()
