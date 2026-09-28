"""The vendored pipeline must reproduce the original Burst Identifier output.

The Trainer holds its own copy of the preprocessing code. If that copy ever
drifts from `H:\\Burst Identifier\\src`, tensors exported here would no longer
match the ones existing checkpoints were trained on -- the exact failure mode
vendoring risks. These tests import the original modules directly from the H:
tree and compare, skipping when the drive is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from callisto_trainer.core import preprocess as vendored
from callisto_trainer.core.config import load_config
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.logging_utils import get_logger

ORIGINAL_ROOT = Path(r"H:\Burst Identifier")


def _load_original_preprocess():
    """Import the upstream preprocess module without its `src.` package deps.

    Loading it as a standalone module avoids needing the whole `src` package on
    sys.path (which would pull in torch-dependent siblings).
    """
    path = ORIGINAL_ROOT / "src" / "preprocessing" / "preprocess.py"
    if not path.exists():
        return None

    source = path.read_text(encoding="utf-8")
    # Strip the sibling imports; only the pure numeric functions are needed.
    kept = [
        line
        for line in source.split("\n")
        if not line.startswith("from src.") and not line.startswith("import src.")
    ]
    module = type(sys)("original_preprocess")
    module.__dict__["__file__"] = str(path)
    # Supply the names the stripped imports would have bound. The numeric
    # functions under test do not touch these, but they are referenced at module
    # scope (LOGGER) and by the CLI entry points.
    module.__dict__.update(
        {
            "get_logger": get_logger,
            "load_config": load_config,
            "read_fits_spectrum": read_fits_spectrum,
            "read_manifest": lambda *args, **kwargs: [],
        }
    )
    exec(compile("\n".join(kept), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def original():
    module = _load_original_preprocess()
    if module is None:
        pytest.skip(f"Original project not available at {ORIGINAL_ROOT}")
    return module


@pytest.fixture
def config() -> dict:
    return load_config()


def test_db_scale_constant_matches(original) -> None:
    assert vendored.PLOTUTIL_DB_SCALE == original.PLOTUTIL_DB_SCALE
    assert vendored.PLOTUTIL_DISPLAY_LIMITS == original.PLOTUTIL_DISPLAY_LIMITS


def test_preprocess_array_matches_on_synthetic_data(
    original, synthetic_spectrum: np.ndarray, config: dict
) -> None:
    assert np.array_equal(
        vendored.preprocess_array(synthetic_spectrum, config),
        original.preprocess_array(synthetic_spectrum, config),
    )


def test_preprocess_array_matches_on_real_files(original, axes_files, config: dict) -> None:
    for path in axes_files:
        spectrum, _ = read_fits_spectrum(path)
        assert np.array_equal(
            vendored.preprocess_array(spectrum, config),
            original.preprocess_array(spectrum, config),
        ), f"vendored preprocessing diverged on {path.name}"


def test_individual_stages_match(original, synthetic_spectrum: np.ndarray, config: dict) -> None:
    cleaned_v = vendored.clean_invalid_values(synthetic_spectrum)
    cleaned_o = original.clean_invalid_values(synthetic_spectrum)
    assert np.array_equal(cleaned_v, cleaned_o)

    bg_v = vendored.subtract_background(cleaned_v, method="plotutil_median_db")
    bg_o = original.subtract_background(cleaned_o, method="plotutil_median_db")
    assert np.array_equal(bg_v, bg_o)

    norm_v = vendored.normalize_spectrum(bg_v, config["preprocessing"])
    norm_o = original.normalize_spectrum(bg_o, config["preprocessing"])
    assert np.array_equal(norm_v, norm_o)

    assert np.array_equal(
        vendored.resize_spectrum(norm_v, (224, 224)),
        original.resize_spectrum(norm_o, (224, 224)),
    )


def test_nan_and_inf_are_handled(config: dict) -> None:
    spectrum = np.full((32, 64), 10.0, dtype=np.float32)
    spectrum[0, 0] = np.nan
    spectrum[1, 1] = np.inf
    spectrum[2, 2] = -np.inf

    tensor = vendored.preprocess_array(spectrum, config)
    assert np.isfinite(tensor).all()
    assert tensor.shape == (1, 224, 224)
