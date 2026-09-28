"""The crop path must agree exactly with the original whole-file pipeline.

This is the load-bearing test of the crop design. If a full-extent crop ever
stops matching ``preprocess_array`` bit-for-bit, then crops and whole-file
tensors are on different scales and any model trained across both is invalid.
"""

from __future__ import annotations

import numpy as np
import pytest

from callisto_trainer.core.config import load_config
from callisto_trainer.core.crops import (
    CropConfig,
    PixelBox,
    crop_from_normalized,
    expand_box,
    normalize_full_spectrum,
    whole_file_box,
)
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.preprocess import preprocess_array


@pytest.fixture
def config() -> dict:
    return load_config()


@pytest.fixture
def crop_cfg(config: dict) -> CropConfig:
    return CropConfig.from_config(config)


def test_full_extent_crop_is_bit_identical_to_preprocess_array(
    synthetic_spectrum: np.ndarray, config: dict, crop_cfg: CropConfig
) -> None:
    expected = preprocess_array(synthetic_spectrum, config)

    normalized = normalize_full_spectrum(synthetic_spectrum, config)
    actual = crop_from_normalized(
        normalized, whole_file_box(synthetic_spectrum.shape), crop_cfg, apply_margin=False
    )

    assert actual.shape == expected.shape == (1, 224, 224)
    assert actual.dtype == expected.dtype == np.float32
    assert np.array_equal(actual, expected), "full-extent crop diverged from preprocess_array"


def test_full_extent_crop_matches_on_real_files(
    axes_files, config: dict, crop_cfg: CropConfig
) -> None:
    for path in axes_files:
        spectrum, _ = read_fits_spectrum(path)
        expected = preprocess_array(spectrum, config)
        normalized = normalize_full_spectrum(spectrum, config)
        actual = crop_from_normalized(
            normalized, whole_file_box(spectrum.shape), crop_cfg, apply_margin=False
        )
        assert np.array_equal(actual, expected), f"mismatch on {path.name}"


def test_margin_on_full_extent_box_is_a_no_op(
    synthetic_spectrum: np.ndarray, config: dict, crop_cfg: CropConfig
) -> None:
    """A box already covering the file cannot grow, so the margin changes nothing."""
    normalized = normalize_full_spectrum(synthetic_spectrum, config)
    box = whole_file_box(synthetic_spectrum.shape)
    with_margin = crop_from_normalized(normalized, box, crop_cfg, apply_margin=True)
    without = crop_from_normalized(normalized, box, crop_cfg, apply_margin=False)
    assert np.array_equal(with_margin, without)


def test_crop_output_contract(
    synthetic_spectrum: np.ndarray, config: dict, crop_cfg: CropConfig
) -> None:
    """Every crop is [1,224,224] float32 confined to the normalized [0,1] range."""
    normalized = normalize_full_spectrum(synthetic_spectrum, config)
    tensor = crop_from_normalized(normalized, PixelBox(20, 60, 300, 480), crop_cfg)

    assert tensor.shape == (1, 224, 224)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_crop_is_taken_after_whole_file_background_subtraction(
    config: dict, crop_cfg: CropConfig
) -> None:
    """The burst must survive cropping; naive crop-then-preprocess destroys it.

    The background step subtracts the per-frequency median *along time*. When the
    burst fills the box's whole time span, cropping first makes that median equal
    the burst level, so the feature is subtracted into its own background and the
    crop comes out uniformly blank. Computing the median over the full file first
    keeps it, because the burst is then a small fraction of the time axis.
    """
    spectrum = np.full((181, 1200), 100.0, dtype=np.float32)
    # A band that spans the entire width of the box but only 12% of the file.
    spectrum[60:90, 420:560] += 80.0
    box = PixelBox(60, 90, 420, 560)

    normalized = normalize_full_spectrum(spectrum, config)
    correct = crop_from_normalized(normalized, box, crop_cfg, apply_margin=False)

    raw_patch = spectrum[box.row0 : box.row1, box.col0 : box.col1]
    wrong = preprocess_array(raw_patch, config)

    # Correct order: +80 digits is far above the display window, so it saturates.
    assert float(correct.min()) == pytest.approx(1.0), "burst should survive as bright"
    # Wrong order: the band becomes its own baseline, leaving a flat 0 dB crop.
    assert float(wrong.max()) < 0.2, "crop-then-preprocess should erase the burst"
    assert float(correct.mean()) > 5.0 * float(wrong.mean())


def test_default_geometry_crops_exactly_the_drawn_box(crop_cfg: CropConfig) -> None:
    """The tensor must cover the labelled region and nothing else.

    A crop that quietly includes surrounding pixels cannot be checked by the
    operator: the preview beside the canvas would no longer be a picture of the
    training sample, and every exported tensor would carry context nobody
    labelled. Boxes of every size and position must come back untouched.
    """
    shape = (200, 1000)
    for box in [
        PixelBox(100, 140, 500, 600),   # ordinary burst-sized box
        PixelBox(50, 52, 300, 303),     # far below any plausible size floor
        PixelBox(0, 5, 0, 5),           # flush against the origin
        PixelBox(195, 200, 995, 1000),  # flush against the far corner
        PixelBox(0, 200, 0, 1000),      # the whole file
    ]:
        assert expand_box(box, shape, crop_cfg).as_tuple() == box.as_tuple()


def test_exact_crop_slices_the_drawn_pixels(
    synthetic_spectrum: np.ndarray, config: dict, crop_cfg: CropConfig
) -> None:
    """End to end: the resized crop comes from the boxed patch, not a larger one."""
    normalized = normalize_full_spectrum(synthetic_spectrum, config)
    box = PixelBox(20, 60, 300, 480)

    from callisto_trainer.core.preprocess import resize_spectrum

    patch = normalized[box.row0 : box.row1, box.col0 : box.col1]
    expected = resize_spectrum(patch, target_shape=crop_cfg.target_shape)[np.newaxis]

    assert np.array_equal(crop_from_normalized(normalized, box, crop_cfg), expected)


def test_expand_box_never_leaves_the_array(crop_cfg: CropConfig) -> None:
    shape = (200, 1000)
    for box in [PixelBox(0, 5, 0, 5), PixelBox(195, 200, 995, 1000), PixelBox(0, 200, 0, 1000)]:
        grown = expand_box(box, shape, crop_cfg)
        assert 0 <= grown.row0 < grown.row1 <= shape[0]
        assert 0 <= grown.col0 < grown.col1 <= shape[1]


# A snapshot exported before crops became exact records its own margin and
# minimums, and ``CropConfig.from_config`` reads them back, so the growth
# machinery still has to work when a config asks for it.


def test_configured_margin_still_expands_and_clamps() -> None:
    padded = CropConfig(context_margin=0.15, min_rows=8, min_cols=8)
    shape = (200, 1000)
    grown = expand_box(PixelBox(100, 140, 500, 600), shape, padded)

    # 15% of 40 rows = 6, 15% of 100 cols = 15.
    assert grown.as_tuple() == (94, 146, 485, 615)

    for box in [PixelBox(0, 5, 0, 5), PixelBox(195, 200, 995, 1000)]:
        clamped = expand_box(box, shape, padded)
        assert 0 <= clamped.row0 < clamped.row1 <= shape[0]
        assert 0 <= clamped.col0 < clamped.col1 <= shape[1]


def test_configured_minimum_still_grows_a_tiny_box() -> None:
    padded = CropConfig(context_margin=0.15, min_rows=8, min_cols=8)
    grown = expand_box(PixelBox(50, 52, 300, 303), (200, 1000), padded)
    assert grown.n_rows >= padded.min_rows
    assert grown.n_cols >= padded.min_cols


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        PixelBox(10, 10, 0, 5)
    with pytest.raises(ValueError):
        PixelBox(0, 5, 10, 3)
