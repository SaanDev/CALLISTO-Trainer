"""Scientifically conservative tensor augmentations for dynamic spectra.

## Why these are sized the way they are

The crop handed to the model is 224x224 whatever the burst's real extent, so the
augmentation widths below are fractions of that fixed grid, not of the original
spectrum. The upstream defaults were tuned for whole-file spectrograms, where a
12-column shift is a meaningful slice of time; on a tight crop it is 5% of the
frame and barely perturbs the sample. Values here are scaled for the crop.

This matters more than usual for this dataset: the rare burst classes are backed
by only a few dozen distinct hand-drawn examples, so a network with millions of
parameters will memorize them unless each epoch presents a visibly different
view.
"""

# NOTE: Vendored from H:\\Burst Identifier (src/data/augmentations.py), then
# adapted. These transforms run at training time only and never touch the
# preprocessing pipeline, so tensor parity (tests/test_preprocess_parity.py) is
# unaffected by anything in this module.

from __future__ import annotations

from typing import Any

import torch


class SpectrumAugmenter:
    """Apply safe training-only augmentations to ``[1, frequency, time]`` tensors.

    These transforms avoid image-style flips and rotations. They preserve the
    basic physical structure of solar radio dynamic spectra while improving
    robustness to timing offsets, intensity calibration differences, weak noise,
    and local missing/RFI-contaminated bands.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        aug_cfg = config.get("augmentation", {})
        self.enabled = bool(aug_cfg.get("enabled", False))
        self.time_shift_max = int(aug_cfg.get("time_shift_max", 0))
        self.intensity_scale_min = float(aug_cfg.get("intensity_scale_min", 1.0))
        self.intensity_scale_max = float(aug_cfg.get("intensity_scale_max", 1.0))
        self.noise_std = float(aug_cfg.get("noise_std", 0.0))
        self.freq_mask_max = int(aug_cfg.get("freq_mask_max", 0))
        self.time_mask_max = int(aug_cfg.get("time_mask_max", 0))
        self.mask_value = float(aug_cfg.get("mask_value", 0.0))
        self.probability = float(aug_cfg.get("probability", 1.0))
        # SpecAugment-style: several narrow masks per axis teach the model to
        # survive a missing band without letting one wide mask swallow the burst.
        self.num_freq_masks = int(aug_cfg.get("num_freq_masks", 1))
        self.num_time_masks = int(aug_cfg.get("num_time_masks", 1))

    def _shift_time(self, tensor: torch.Tensor) -> torch.Tensor:
        """Translate along time, filling the vacated edge rather than wrapping.

        ``torch.roll`` would wrap pixels off one edge back onto the other. On a
        whole-file spectrogram that is harmless; on a tight crop it takes part of
        the burst and pastes a copy of it against the opposite edge, teaching the
        model a morphology that cannot occur in real data.
        """
        shift = int(torch.randint(-self.time_shift_max, self.time_shift_max + 1, ()).item())
        if not shift:
            return tensor

        shifted = torch.full_like(tensor, self.mask_value)
        if shift > 0:
            shifted[:, :, shift:] = tensor[:, :, :-shift]
        else:
            shifted[:, :, :shift] = tensor[:, :, -shift:]
        return shifted

    def _apply_masks(self, tensor: torch.Tensor, axis: int, max_width: int, count: int) -> None:
        """Zero ``count`` random bands of up to ``max_width`` along ``axis``, in place."""
        limit = min(max_width, tensor.shape[axis])
        if limit <= 0:
            return
        for _ in range(max(0, count)):
            width = int(torch.randint(0, limit + 1, ()).item())
            if not width:
                continue
            start = int(torch.randint(0, tensor.shape[axis] - width + 1, ()).item())
            if axis == 1:
                tensor[:, start : start + width, :] = self.mask_value
            else:
                tensor[:, :, start : start + width] = self.mask_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled or float(torch.rand(()).item()) > self.probability:
            return tensor
        if tensor.shape[0] > 1:
            return self._augment_views(tensor)

        augmented = tensor.clone()

        if self.time_shift_max > 0:
            augmented = self._shift_time(augmented)

        if self.intensity_scale_min != 1.0 or self.intensity_scale_max != 1.0:
            scale = torch.empty(()).uniform_(self.intensity_scale_min, self.intensity_scale_max)
            augmented = augmented * scale

        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std

        if self.freq_mask_max > 0:
            self._apply_masks(augmented, 1, self.freq_mask_max, self.num_freq_masks)
        if self.time_mask_max > 0:
            self._apply_masks(augmented, 2, self.time_mask_max, self.num_time_masks)

        # Intensity scaling and noise can push samples outside the [0, 1] range
        # the network is trained to expect; the crop pipeline clips there too.
        return augmented.clamp_(0.0, 1.0)

    def _augment_views(self, tensor: torch.Tensor) -> torch.Tensor:
        """Augment a ``[V, frequency, time]`` stack of views of one region.

        The views are different windows onto the same recording, so a
        calibration change (the intensity scale) is shared, while shifts, noise
        and masks are drawn per view: the same pixel offset means a different
        span of time in each, so sharing them would not describe anything real.
        """
        scale = torch.empty(()).uniform_(self.intensity_scale_min, self.intensity_scale_max)
        views = []
        for view in tensor.unbind(0):
            augmented = view.unsqueeze(0).clone()
            if self.time_shift_max > 0:
                augmented = self._shift_time(augmented)
            augmented = augmented * scale
            if self.noise_std > 0:
                augmented = augmented + torch.randn_like(augmented) * self.noise_std
            if self.freq_mask_max > 0:
                self._apply_masks(augmented, 1, self.freq_mask_max, self.num_freq_masks)
            if self.time_mask_max > 0:
                self._apply_masks(augmented, 2, self.time_mask_max, self.num_time_masks)
            views.append(augmented)
        return torch.cat(views, dim=0).clamp_(0.0, 1.0)
