"""Candidate regions: where in a spectrum the model is asked to look.

A plain signal-processing pass -- threshold the normalized spectrum, group
connected pixels, keep blobs of a plausible size. It is **not** a detector: it
finds *bright things*, and interference, carrier lines and calibration artifacts
qualify as readily as solar bursts. Deciding which candidates are bursts is the
unified model's job, which is why the same finder, with the same settings, must
produce both the training regions and the inference regions.

Qt-free and torch-free so the exporter and the feature code can import it
without pulling in a deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from callisto_trainer.core.crops import PixelBox

# 0.35 in normalized units is about +2.2 dB above background. Calibrated against
# 20 confirmed burst files: it locates a region in 19 of them (0.45 managed only
# 17, and missed 5 of 8 in a quieter ALASKA subset), while keeping the candidate
# count low enough to review by eye.
DEFAULT_REGION_THRESHOLD = 0.35
DEFAULT_MIN_AREA = 60
# Candidates are ranked largest-first, and large interference crowds real bursts
# down the list: on 107 real validation burst files, 13 of the 28 that no region
# reached had their burst ranked below the twelfth region. With eight or twelve,
# the cap was protecting a model that could not reject interference; the unified
# model can, and at 24 the share of burst files reachable rose from 74% to 83%
# with false alarms held by the calibrated threshold, which is tuned with this
# same cap. Training mines from twice this many (negatives.MINING_POOL).
DEFAULT_MAX_REGIONS = 24

# Adaptive thresholding. Stations differ enormously in gain, so the same solar
# event can peak at 1.0 in one recording and 0.43 in another: a single absolute
# threshold either floods the bright files or misses the faint ones entirely
# (ALASKA-COHOE_20230617_2243_2248 is a real example - a clearly visible Type III
# whose whole burst sits below 0.43). Taking a high percentile of the file itself
# adapts to its own noise floor. The absolute setting acts as a ceiling and
# ADAPTIVE_FLOOR as a lower bound, so this can only relax the bar for faint
# files, never raise it for bright ones or chase pure noise.
ADAPTIVE_PERCENTILE = 99.9
ADAPTIVE_FLOOR = 0.15

# Hysteresis. A faint burst -- a drifting lane at +1.3 to +1.8 dB -- only crosses
# the threshold in scattered patches, and each patch alone is under the minimum
# area; the adaptive threshold cannot relax to help, because bright interference
# elsewhere in the same file pins it at its ceiling. Measured over 710 labelled
# burst files, 95 (13%) were reachable only as such fragments. So the pixels at
# the threshold act as *seeds*, and each seed's component is grown through the
# connected pixels of a lightly smoothed copy that reach HYSTERESIS x the
# threshold. Smoothing keeps independent noise pixels from linking up; the seed
# requirement means faint structure with no bright core never becomes a
# candidate. Against the alternatives, on all 2,103 labelled files, hysteresis
# reached 88% of burst files where lowering the minimum area to 20 reached 83%
# and dilating the mask 77-79% (it glued bursts to nearby interference: 9-13% of
# burst files lost to merging, against 1.5%). The settings were then swept on
# the train and validation files only (ratio 0.6-0.8, seeds 3-10):
#
#     plain thresholding    reached 82.5%   merged away 1.5%   18.5 candidates/quiet file
#     ratio 0.65, 5 seeds   reached 90.1%   merged away 1.5%   20.8 candidates/quiet file
#
# 0.6 starts merging (3.3%); from 0.75 reach falls back. None restores plain
# thresholding.
DEFAULT_HYSTERESIS = 0.65
HYSTERESIS_SMOOTH = 3
MIN_SEED_PIXELS = 5


def resolve_threshold(
    normalized: np.ndarray,
    absolute: float = DEFAULT_REGION_THRESHOLD,
    adaptive: bool = True,
    percentile: float = ADAPTIVE_PERCENTILE,
    floor: float = ADAPTIVE_FLOOR,
) -> float:
    """The brightness threshold to use for one spectrum."""
    if not adaptive:
        return float(absolute)
    value = float(np.percentile(normalized, percentile))
    return float(min(float(absolute), max(float(floor), value)))


@dataclass
class Proposal:
    """A candidate region with an optional predicted type."""

    row0: int
    row1: int
    col0: int
    col1: int
    area: int
    peak: float
    burst_type: str | None = None
    probability: float | None = None

    def as_box(self) -> PixelBox:
        return PixelBox(self.row0, self.row1, self.col0, self.col1)


def _label_connected(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components, preferring SciPy when it is available."""
    try:
        from scipy import ndimage

        return ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    except ImportError:  # pragma: no cover - SciPy is in requirements
        return _label_connected_fallback(mask)


def _label_connected_fallback(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Iterative flood fill, used only when SciPy is missing."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        current += 1
        stack = [start]
        labels[start] = current
        while stack:
            row, col = stack.pop()
            for d_row, d_col in neighbours:
                r, c = row + d_row, col + d_col
                if (
                    0 <= r < mask.shape[0]
                    and 0 <= c < mask.shape[1]
                    and mask[r, c]
                    and not labels[r, c]
                ):
                    labels[r, c] = current
                    stack.append((r, c))
    return labels, current


def find_candidate_regions(
    normalized: np.ndarray,
    threshold: float = DEFAULT_REGION_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_candidates: int = DEFAULT_MAX_REGIONS,
    hysteresis: float | None = DEFAULT_HYSTERESIS,
    min_seed: int = MIN_SEED_PIXELS,
) -> list[Proposal]:
    """Propose bright connected regions in a normalized spectrum, largest first.

    With ``hysteresis`` (see the module constants) a region is a component of the
    smoothed spectrum above ``hysteresis * threshold`` holding at least
    ``min_seed`` pixels above ``threshold``; its box, area and peak are taken over
    its own pixels above the lower level, so smoothing never widens it. With
    ``hysteresis=None`` it is a component above ``threshold``, as it always was.
    """
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D normalized spectrum, got {normalized.shape}")

    seeds = normalized >= float(threshold)
    if not seeds.any():
        return []

    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - SciPy is in requirements
        ndimage = None

    if hysteresis is not None and ndimage is not None:
        return _hysteresis_regions(
            normalized, seeds, float(threshold) * float(hysteresis),
            int(min_area), int(max_candidates), int(min_seed), ndimage,
        )

    mask = seeds
    labels, count = _label_connected(mask)
    if count == 0:
        return []

    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(areas[1:] >= min_area) + 1
    if keep.size == 0:
        return []

    proposals: list[Proposal] = []
    if ndimage is not None:
        # One pass for every bounding box and every peak. Slicing the label image
        # once per component instead costs a full-array scan each time, which on
        # a noisy 40,000-column recording with hundreds of blobs takes seconds.
        slices = ndimage.find_objects(labels)
        peaks = ndimage.maximum(normalized, labels, index=keep)
        for component, peak in zip(keep, np.atleast_1d(peaks)):
            rows, cols = slices[component - 1]
            proposals.append(
                Proposal(
                    row0=int(rows.start),
                    row1=int(rows.stop),
                    col0=int(cols.start),
                    col1=int(cols.stop),
                    area=int(areas[component]),
                    peak=float(peak),
                )
            )
    else:  # pragma: no cover
        for component in keep:
            rows, cols = np.nonzero(labels == component)
            proposals.append(
                Proposal(
                    row0=int(rows.min()),
                    row1=int(rows.max()) + 1,
                    col0=int(cols.min()),
                    col1=int(cols.max()) + 1,
                    area=int(areas[component]),
                    peak=float(normalized[rows, cols].max()),
                )
            )

    proposals.sort(key=lambda item: item.area, reverse=True)
    return proposals[:max_candidates]


def _hysteresis_regions(
    normalized: np.ndarray,
    seeds: np.ndarray,
    low_level: float,
    min_area: int,
    max_candidates: int,
    min_seed: int,
    ndimage,
) -> list[Proposal]:
    """Seeded regions grown through a smoothed low-level mask (see module notes)."""
    smoothed = ndimage.uniform_filter(np.asarray(normalized, dtype=np.float32),
                                      size=HYSTERESIS_SMOOTH)
    grown = (smoothed >= low_level) | seeds
    labels, count = ndimage.label(grown, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return []

    # A region is measured over its own pixels at the low level, not over the
    # smoothing halo, so a sharp-edged blob keeps exactly its own box.
    own = np.where(normalized >= low_level, labels, 0)
    areas = np.bincount(own.ravel(), minlength=count + 1)
    seed_counts = np.bincount(labels[seeds], minlength=count + 1)
    keep = np.flatnonzero((areas[1:] >= min_area) & (seed_counts[1:] >= min_seed)) + 1
    if keep.size == 0:
        return []

    slices = ndimage.find_objects(own)
    peaks = np.atleast_1d(ndimage.maximum(normalized, own, index=keep))
    proposals = []
    for component, peak in zip(keep, peaks):
        rows, cols = slices[component - 1]
        proposals.append(
            Proposal(
                row0=int(rows.start),
                row1=int(rows.stop),
                col0=int(cols.start),
                col1=int(cols.stop),
                area=int(areas[component]),
                peak=float(peak),
            )
        )
    proposals.sort(key=lambda item: item.area, reverse=True)
    return proposals[:max_candidates]
