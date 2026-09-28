"""Turn a drawn box into a training tensor.

## The rule that makes this correct

The background step (``plotutil_median_db``) subtracts the **per-frequency median
along the time axis**. If a box were cropped from the raw array and preprocessed
afterwards, that median would be computed over only the few seconds inside the
box -- and a burst that fills the box would be subtracted into its own
background, erasing the very signal being labelled.

So the order is fixed and non-negotiable:

    read -> clean -> background-subtract -> normalize   (whole file)
                                                |
                                                +-> slice the box
                                                        |
                                                        +-> resize to 224x224

Every step except the slice calls the vendored functions in
:mod:`callisto_trainer.core.preprocess` unchanged. The direct consequence, which
:mod:`tests.test_crop_equivalence` asserts, is that a full-extent crop with no
margin is **bit-identical** to ``preprocess_array()`` on the same file.

## What the slice covers

Exactly the box, and nothing more. The crop geometry in :class:`CropConfig` can
pad a box with surrounding context or grow it to a floor size, but both are zero
by default, because a tensor that covers more than the operator labelled is a
tensor they cannot verify by looking at it. The preview in the labelling panel
and the exporters run the identical code path, so what is drawn, what is
previewed and what is trained on are the same pixels.

## The context view

The exact crop cannot show what surrounds a region, and that is where most
interference gives itself away: a carrier continues far beyond the box, an
impulse runs the full height of the band, periodic interference repeats. So the
unified model also receives a second, separate view (:func:`crop_context`) -- the
full frequency band over several times the region's duration. It is its own
tensor, previewed beside the crop, and never alters the crop itself.

Its resize max-pools first. Squeezing a few thousand columns into 224 by linear
interpolation samples between columns, and a one-sample interference spike --
the very thing this view exists to show -- can fall between samples and vanish.

## The quiet-background view

The background step takes each channel's *median* over the file. A continuum
that lasts more than about half the file -- a Type IV -- is then its own
background: its channels come out near zero and the quiet part of the file
turns dark instead. Measured on the archive's long "Other" boxes (over 80% of
the file), 0.7% of their pixels cleared the region finder's level, against 3-9%
for every other kind of box. :func:`quiet_normalized_spectrum` takes each
channel's background from its quietest part instead (the 10th percentile), on
the same dB scale and window, and the third view (``quiet_context``) is the
context strip of that array. It only helps where some of the file is quiet: a
continuum filling the whole recording has no quiet part in any per-file
background. In an A/B on the archive, at equal false alarms the three-view model
found as many or more burst files and more of the long boxes at every level
(e.g. 11 vs 9 of 13 at 4 false alarms in 192 quiet files).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.preprocess import (
    PLOTUTIL_DB_SCALE,
    clean_invalid_values,
    normalize_spectrum,
    resize_spectrum,
    subtract_background,
)

# Each channel's background for the quiet-background view: this percentile of
# its samples over the file, i.e. its quietest tenth.
QUIET_PERCENTILE = 10.0


@dataclass(frozen=True)
class PixelBox:
    """A half-open box in spectrum pixel coordinates: rows ``[row0, row1)``.

    Rows are frequency channels, columns are time samples, matching the
    ``[frequency, time]`` array orientation.
    """

    row0: int
    row1: int
    col0: int
    col1: int

    def __post_init__(self) -> None:
        if self.row1 <= self.row0 or self.col1 <= self.col0:
            raise ValueError(
                f"PixelBox must have positive extent, got "
                f"rows [{self.row0}, {self.row1}) cols [{self.col0}, {self.col1})"
            )

    @property
    def n_rows(self) -> int:
        return self.row1 - self.row0

    @property
    def n_cols(self) -> int:
        return self.col1 - self.col0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.row0, self.row1, self.col0, self.col1


@dataclass(frozen=True)
class CropConfig:
    """Crop geometry settings, resolved from the ``crops`` config section.

    The defaults are deliberately inert -- zero margin, no size floor -- so a crop
    is exactly the drawn box. A dataset snapshot that recorded non-zero values
    still reproduces its own geometry, because ``from_config`` reads them back.
    """

    context_margin: float = 0.0
    min_rows: int = 1
    min_cols: int = 1
    target_shape: tuple[int, int] = (224, 224)
    # The context view extends a region by its own width on each side, and by at
    # least this many samples (about a minute at the usual 0.25 s cadence).
    context_min_pad_cols: int = 240

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CropConfig":
        crop_cfg = config.get("crops", {}) or {}
        target = crop_cfg.get("target_shape") or config["data"]["target_shape"]
        return cls(
            context_margin=float(crop_cfg.get("context_margin", 0.0)),
            min_rows=int(crop_cfg.get("min_rows", 1)),
            min_cols=int(crop_cfg.get("min_cols", 1)),
            target_shape=(int(target[0]), int(target[1])),
            context_min_pad_cols=int(crop_cfg.get("context_min_pad_cols", 240)),
        )


def normalize_full_spectrum(spectrum: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Run clean -> background -> normalize on the **whole** file, without resizing.

    This is both the array crops are taken from and the array the labelling canvas
    displays, so what the user sees is what the model receives.
    """
    prep_cfg = config["preprocessing"]
    data = clean_invalid_values(spectrum)
    data = subtract_background(data, method=prep_cfg["background_method"])
    return normalize_spectrum(data, prep_cfg)


def quiet_normalized_spectrum(
    spectrum: np.ndarray, config: dict[str, Any], percentile: float = QUIET_PERCENTILE
) -> np.ndarray:
    """The whole file with each channel's background taken from its quiet part.

    Identical to :func:`normalize_full_spectrum` except for the baseline: the
    ``percentile``-th sample of each channel instead of its median, so emission
    that lasts most of the recording stays above background. Same dB scale, same
    display window, same ``[0, 1]`` range.
    """
    prep_cfg = config["preprocessing"]
    method = str(prep_cfg.get("background_method") or "").strip().lower().replace("-", "_")
    data = clean_invalid_values(spectrum)
    baseline = np.percentile(data, float(percentile), axis=1, keepdims=True).astype(np.float32)
    data = (data - baseline).astype(np.float32)
    if method in {"plotutil_median_db", "plotutil", "plotutil_median", "ecallisto_db"}:
        data = (data * np.float32(PLOTUTIL_DB_SCALE)).astype(np.float32)
    elif method != "per_frequency_median":
        raise ValueError(
            f"The quiet-background view needs a per-channel background; "
            f"background_method is {method!r}"
        )
    return normalize_spectrum(data, prep_cfg)


def whole_file_box(shape: tuple[int, ...]) -> PixelBox:
    """The box covering an entire spectrum of the given ``[frequency, time]`` shape."""
    return PixelBox(0, int(shape[0]), 0, int(shape[1]))


def expand_box(box: PixelBox, shape: tuple[int, ...], crop_cfg: CropConfig) -> PixelBox:
    """Grow a box by the context margin and up to the minimum size, clamped to bounds.

    With the default configuration this returns ``box`` unchanged: the crop is
    exactly the region the operator drew, which is what makes the labelling
    preview a faithful picture of the training tensor. Growth only happens for a
    config that asks for it -- ``context_margin`` adds surrounding context on
    every side, and ``min_rows``/``min_cols`` impose a floor so a 3x4 patch is not
    upsampled to 224x224 as pure interpolation artifacts. Both are off by default
    because either one silently makes the tensor cover more than was labelled.

    Clamping to the array bounds always runs, so the returned box is a valid slice
    of ``shape`` whatever the configuration.
    """
    n_freq, n_time = int(shape[0]), int(shape[1])

    row_pad = int(round(box.n_rows * crop_cfg.context_margin))
    col_pad = int(round(box.n_cols * crop_cfg.context_margin))
    row0, row1 = box.row0 - row_pad, box.row1 + row_pad
    col0, col1 = box.col0 - col_pad, box.col1 + col_pad

    row0, row1 = _grow_to_minimum(row0, row1, crop_cfg.min_rows, n_freq)
    col0, col1 = _grow_to_minimum(col0, col1, crop_cfg.min_cols, n_time)

    row0, row1 = _clamp_span(row0, row1, n_freq)
    col0, col1 = _clamp_span(col0, col1, n_time)
    return PixelBox(row0, row1, col0, col1)


def _grow_to_minimum(low: int, high: int, minimum: int, limit: int) -> tuple[int, int]:
    """Expand ``[low, high)`` symmetrically to at least ``minimum`` samples."""
    target = min(int(minimum), limit)
    deficit = target - (high - low)
    if deficit <= 0:
        return low, high
    return low - deficit // 2, high + (deficit - deficit // 2)


def _clamp_span(low: int, high: int, limit: int) -> tuple[int, int]:
    """Shift then clip ``[low, high)`` into ``[0, limit)``, preserving width if possible."""
    width = min(high - low, limit)
    if low < 0:
        low, high = 0, width
    if high > limit:
        high, low = limit, limit - width
    return max(0, low), min(limit, max(high, low + 1))


def crop_from_normalized(
    normalized: np.ndarray,
    box: PixelBox,
    crop_cfg: CropConfig,
    apply_margin: bool = True,
) -> np.ndarray:
    """Slice ``normalized`` and resize the patch to the model input shape.

    ``normalized`` must already be the output of :func:`normalize_full_spectrum`
    for the whole file. Returns a ``[1, H, W]`` float32 tensor.
    """
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D normalized spectrum, got shape {normalized.shape}")

    effective = expand_box(box, normalized.shape, crop_cfg) if apply_margin else box
    row0, row1, col0, col1 = effective.as_tuple()
    patch = normalized[row0:row1, col0:col1]
    resized = resize_spectrum(patch, target_shape=crop_cfg.target_shape)
    return resized[np.newaxis, :, :].astype(np.float32)


def context_box(box: PixelBox, shape: tuple[int, ...], crop_cfg: CropConfig) -> PixelBox:
    """The region the context view covers: the full band, and wider in time."""
    n_freq, n_time = int(shape[0]), int(shape[1])
    pad = max(box.n_cols, int(crop_cfg.context_min_pad_cols))
    col0, col1 = _clamp_span(box.col0 - pad, box.col1 + pad, n_time)
    return PixelBox(0, n_freq, col0, col1)


def _max_pool_axis(data: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Block-max along ``axis`` down to no fewer than ``target`` samples."""
    size = data.shape[axis]
    factor = size // max(1, int(target))
    if factor < 2:
        return data
    blocks = int(np.ceil(size / factor))
    padding = blocks * factor - size
    if padding:
        widths = [(0, 0), (0, 0)]
        widths[axis] = (0, padding)
        data = np.pad(data, widths, mode="edge")
    if axis == 1:
        return data.reshape(data.shape[0], blocks, factor).max(axis=2)
    return data.reshape(blocks, factor, data.shape[1]).max(axis=1)


def resize_keeping_peaks(patch: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize with a max-pool first, so thin bright features survive downsampling."""
    data = np.asarray(patch, dtype=np.float32)
    data = _max_pool_axis(data, int(target_shape[1]), axis=1)
    data = _max_pool_axis(data, int(target_shape[0]), axis=0)
    return resize_spectrum(data, target_shape=target_shape)


def crop_context(normalized: np.ndarray, box: PixelBox, crop_cfg: CropConfig) -> np.ndarray:
    """The context view of a region as a ``[1, H, W]`` tensor (see module docs)."""
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D normalized spectrum, got shape {normalized.shape}")
    region = context_box(box, normalized.shape, crop_cfg)
    patch = normalized[region.row0:region.row1, region.col0:region.col1]
    return resize_keeping_peaks(patch, crop_cfg.target_shape)[np.newaxis].astype(np.float32)


VIEW_CROP = "crop"
VIEW_CONTEXT = "context"
VIEW_QUIET_CONTEXT = "quiet_context"
VIEWS = (VIEW_CROP, VIEW_CONTEXT, VIEW_QUIET_CONTEXT)


def region_views(
    normalized: np.ndarray,
    box: PixelBox,
    crop_cfg: CropConfig,
    views: tuple[str, ...] = (VIEW_CROP,),
    quiet: np.ndarray | None = None,
) -> np.ndarray:
    """Stack the requested views of one region into a ``[V, H, W]`` tensor.

    View 0 is always the exact crop, so a one-view tensor is bit-identical to
    :func:`crop_from_normalized`. ``quiet`` is the file's
    :func:`quiet_normalized_spectrum`, needed only by the ``quiet_context`` view.
    """
    layers = []
    for view in views:
        if view == VIEW_CROP:
            layers.append(crop_from_normalized(normalized, box, crop_cfg)[0])
        elif view == VIEW_CONTEXT:
            layers.append(crop_context(normalized, box, crop_cfg)[0])
        elif view == VIEW_QUIET_CONTEXT:
            if quiet is None:
                raise ValueError(
                    "The quiet_context view needs the quiet-background spectrum "
                    "(crops.quiet_normalized_spectrum of the raw file)"
                )
            layers.append(crop_context(quiet, box, crop_cfg)[0])
        else:
            raise ValueError(f"Unknown view {view!r}; expected one of {VIEWS}")
    return np.stack(layers).astype(np.float32)


def crop_from_spectrum(
    spectrum: np.ndarray,
    box: PixelBox,
    config: dict[str, Any],
    crop_cfg: CropConfig | None = None,
    apply_margin: bool = True,
) -> np.ndarray:
    """Full path from a raw spectrum to one crop tensor.

    Prefer :func:`normalize_full_spectrum` + :func:`crop_from_normalized` when a
    file has several boxes, so the expensive whole-file normalization runs once.
    """
    crop_cfg = crop_cfg or CropConfig.from_config(config)
    normalized = normalize_full_spectrum(spectrum, config)
    return crop_from_normalized(normalized, box, crop_cfg, apply_margin=apply_margin)


def crop_from_file(
    path: str | Path,
    box: PixelBox,
    config: dict[str, Any],
    crop_cfg: CropConfig | None = None,
    apply_margin: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a FITS file and return ``(crop_tensor, metadata)`` for one box."""
    spectrum, metadata = read_fits_spectrum(path)
    tensor = crop_from_spectrum(
        spectrum, box, config, crop_cfg=crop_cfg, apply_margin=apply_margin
    )
    return tensor, metadata


def crops_from_file(
    path: str | Path,
    boxes: list[PixelBox],
    config: dict[str, Any],
    crop_cfg: CropConfig | None = None,
    apply_margin: bool = True,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Read a FITS file once and produce one crop tensor per box."""
    crop_cfg = crop_cfg or CropConfig.from_config(config)
    spectrum, metadata = read_fits_spectrum(path)
    normalized = normalize_full_spectrum(spectrum, config)
    tensors = [
        crop_from_normalized(normalized, box, crop_cfg, apply_margin=apply_margin)
        for box in boxes
    ]
    return tensors, metadata
