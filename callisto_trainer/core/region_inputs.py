"""Everything a region model is given about one region, built one way.

The exporter, the post-training calibration, the Predict tab and the labelling
suggestions all turn a candidate region into model inputs. If any of them did it
differently the model would be evaluated -- or used -- on inputs unlike the ones
it learned from, and it would still answer confidently. So all of them go
through :class:`RegionEncoder`, configured from the model's own config.

What a model takes is described by :class:`RegionInputSpec`:

* ``views`` -- the image tensors, stacked as channels: ``("crop",)`` for the
  exact crop alone, ``("crop", "context")`` to add the wide context strip, and
  ``"quiet_context"`` for that strip on the quiet-part background, which keeps
  long continua (Type IV) visible;
* ``feature_set`` -- the vector beside the image: ``None`` for image-only
  models, ``"physics_v1"`` for the original drift/extent measurements, or
  ``"region_v2"`` which adds the interference features of
  :mod:`callisto_trainer.core.region_features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from callisto_trainer.core.burst_physics import BurstPhysics, measure_burst
from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.crops import (
    VIEW_CONTEXT,
    VIEW_CROP,
    VIEW_QUIET_CONTEXT,
    CropConfig,
    PixelBox,
    quiet_normalized_spectrum,
    region_views,
)
from callisto_trainer.core.region_features import (
    FEATURE_SET_PHYSICS_V1,
    FEATURE_SET_REGION_V2,
    FileContext,
    feature_count,
    feature_vector,
    file_context,
    measure_region,
)

# What a new unified export trains on: V3 adds the quiet-background context.
V2_VIEWS: tuple[str, ...] = (VIEW_CROP, VIEW_CONTEXT)
V3_VIEWS: tuple[str, ...] = (VIEW_CROP, VIEW_CONTEXT, VIEW_QUIET_CONTEXT)
V2_FEATURE_SET = FEATURE_SET_REGION_V2


@dataclass(frozen=True)
class RegionInputSpec:
    """The inputs a region model expects. Read from its config, never assumed."""

    views: tuple[str, ...] = (VIEW_CROP,)
    feature_set: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RegionInputSpec":
        model = config.get("model", {}) or {}
        views = tuple(model.get("views") or (VIEW_CROP,))
        feature_set = None
        if bool(model.get("use_physics", False)):
            # Checkpoints from before feature sets existed used the eight
            # physics features, so that is what an unnamed set means.
            feature_set = str(model.get("feature_set") or FEATURE_SET_PHYSICS_V1)
        return cls(views=views, feature_set=feature_set)

    @property
    def num_views(self) -> int:
        return len(self.views)

    @property
    def num_features(self) -> int:
        return feature_count(self.feature_set) if self.feature_set else 0

    @property
    def needs_context(self) -> bool:
        return self.feature_set == FEATURE_SET_REGION_V2

    @property
    def needs_quiet(self) -> bool:
        return VIEW_QUIET_CONTEXT in self.views


@dataclass
class EncodedRegion:
    """One region, ready for the model."""

    image: np.ndarray                 # [V, H, W] float32
    features: np.ndarray | None       # [N] float32, or None for image-only models
    physics: BurstPhysics | None = None
    region: dict[str, float] | None = None


class RegionEncoder:
    """Builds :class:`EncodedRegion` inputs for one model configuration."""

    def __init__(self, config: dict[str, Any], spec: RegionInputSpec | None = None) -> None:
        self.spec = spec or RegionInputSpec.from_config(config)
        self.crop_config = CropConfig.from_config(config)
        self.config = config

    def quiet(self, spectrum: np.ndarray | None) -> np.ndarray | None:
        """The quiet-background array of a raw spectrum, when this model uses it."""
        if not self.spec.needs_quiet or spectrum is None:
            return None
        return quiet_normalized_spectrum(spectrum, self.config)

    def context(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None = None,
        rfi_channels: Any = None,
    ) -> FileContext | None:
        """Whole-file statistics, computed once per file when the model needs them."""
        if not self.spec.needs_context:
            return None
        return file_context(normalized, axes, rfi_channels)

    def encode(
        self,
        normalized: np.ndarray,
        box: PixelBox,
        axes: SpectrumAxes | None = None,
        context: FileContext | None = None,
        quiet: np.ndarray | None = None,
    ) -> EncodedRegion:
        image = region_views(normalized, box, self.crop_config, self.spec.views, quiet=quiet)
        if not self.spec.feature_set:
            return EncodedRegion(image=image, features=None)

        # Physics is measured on the same pixels for every class, background
        # included, so the model cannot tell classes apart by whether a
        # measurement exists.
        physics = (
            measure_burst(normalized, axes, box.row0, box.row1, box.col0, box.col1)
            if axes is not None
            else BurstPhysics()
        )
        region = None
        if self.spec.feature_set == FEATURE_SET_REGION_V2:
            if context is None:
                context = file_context(normalized, axes)
            region = measure_region(context, box.row0, box.row1, box.col0, box.col1, physics)
        features = feature_vector(self.spec.feature_set, physics, region)
        return EncodedRegion(image=image, features=features, physics=physics, region=region)
