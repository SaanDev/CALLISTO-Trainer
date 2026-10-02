"""The physical parameters of a burst: from its box, and from its pixels.

Frequency drift rate is the quantity that physically separates the burst types.
A Type III burst is an electron beam travelling at 0.1-0.5c and drifts at roughly
``-0.01 x f^1.84`` MHz/s (Alvarez & Haddock 1973) -- about -10 MHz/s at 40 MHz. A
Type II burst is a shock front moving at ~1000 km/s and drifts a hundred times
more slowly, order -0.1 MHz/s. Nothing else discriminates them so cleanly.

There are two measurements here, for two different consumers.

## The burst parameters of a drawn box (:func:`box_parameters`)

What the Label tab shows and stores for a box, calculated **from the box** and
only for **Type II and Type III**: the operator draws the box from the burst's
start to its end, so its height is the frequency range and its width the
duration. A Type II or III drifts from high to low frequency, so the burst
starts at the top of the box and ends at the bottom:

    f_start = highest frequency of the box,  f_end = lowest
    t_start = first time of the box,          t_end = last
    df/dt   = (f_start - f_end) / (t_start - t_end)      (negative)

On the archive's boxes this gives a median |df/dt| of 1.9 MHz/s for Type III
(86% inside the published 1-200 MHz/s) and 0.14 MHz/s for Type II (94% inside
0.05-1 MHz/s), and a value for every box -- the pixel fit below gave a usable
value for only 801 of 1,513 Type III boxes. The result is only as good as the
box: drawn wider in time than the burst, it reads slower. Type IIIG (a group:
the box spans several bursts, not one), Type IV and Other get no parameters.

## The measurement given to the model (:func:`measure_burst`)

The model's physics features describe every *candidate region* the finder
proposes, background and interference included, and at prediction there is no
drawn box and no type -- so they cannot come from a box, and must not depend on
the label (a value present only for Type II/III samples would teach the model
the label itself). They are measured from the region's own pixels, the same way
for every region:

## Method

1. Threshold inside the box, adaptively, relative to that box's own content.
2. Keep the largest connected component -- the burst, rather than whatever else
   is bright nearby.
3. Track its ridge. Along time when the burst spans many columns (slow drift,
   Type II-like), along frequency when it spans many rows (fast drift, Type
   III-like, where a per-column fit would have almost no points to work with).
   The axis with more distinct samples wins.
4. Fit with Theil-Sen rather than least squares: the median of pairwise slopes
   ignores the outliers that interference and edge effects inject, where a
   squared-error fit would be dragged by them.

Every result carries a quality flag. A drift rate from four noisy columns is not
the same measurement as one from eighty, and the model and the operator both need
to be able to tell the difference.

## Counting bursts

A box can hold one Type III or a *group* of them (Type IIIG). The drift fit above
deliberately keeps only the largest component, so it describes one lane of a
group; the count of separate bursts is measured separately, over every bright
pixel in the box (see :func:`count_bursts`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.taxonomy import IIIG_MIN_BURSTS, TYPE_II, TYPE_III, TYPE_IIIG

# Fraction of the box's own bright tail used as the burst threshold. Relative
# rather than absolute because station gain varies by orders of magnitude.
#
# 93 was chosen by sweeping 88-99 over 198 real annotations: it maximises the
# separation between the two burst types (41x, versus 35x at 96 and 4.8x at 98,
# where so few pixels survive that the track fragments) while keeping the
# fraction of regions that yield a usable fit at its peak.
BURST_PERCENTILE = 93.0
# Never go below this in normalized units (~+1.7 dB) or noise becomes "burst".
MIN_BURST_LEVEL = 0.30
# Fewer usable samples than this and a fitted slope is not a measurement.
MIN_TRACK_SAMPLES = 5
# Published single-burst drift rates at metric wavelengths, for reference and for
# the UI to quote. |df/dt| in MHz/s.
LITERATURE_DRIFT_RANGES: dict[str, tuple[float, float]] = {
    "Type II": (0.05, 1.0),
    "Type III": (1.0, 200.0),
    # Each burst of a group drifts like a single Type III.
    "Type IIIG": (1.0, 200.0),
}

# Bounds used to flag a drift that contradicts its label. Deliberately far wider
# than the literature values above: a check that fires on ordinary boxes gets
# ignored. They catch gross errors -- a "Type II" drifting like a beam, a
# "Type III" that is flat. With the box drift (box_parameters) they flag 9 of the
# archive's 382 Type II boxes, all faster than 2 MHz/s, and none of its 1,513
# Type III boxes. (Calibrated first against the pixel fit, where the literature
# range flagged half of all correctly-labelled Type III boxes.) Type IIIG boxes
# get no drift, so their entries only matter to an older stored measurement.
EXPECTED_DRIFT_RANGES: dict[str, tuple[float, float]] = {
    "Type II": (0.001, 2.0),
    "Type III": (0.02, 500.0),
    # The drift is fitted to the largest lane of the group, so it reads like a
    # single Type III.
    "Type IIIG": (0.02, 500.0),
}

# Burst counting. A gap shorter than BURST_GAP_S inside one lane is closed before
# counting, and a run shorter than MIN_BURST_RUN_S is not a burst. Both are in
# seconds so the rule does not change with a station's cadence; the shortest run
# also keeps single-sample interference spikes from being counted as bursts.
BURST_GAP_S = 0.5
MIN_BURST_RUN_S = 0.5

# The burst types whose parameters are calculated from the drawn box, and the
# confidence recorded for a box-derived result.
BOX_DRIFT_TYPES: tuple[str, ...] = (TYPE_II, TYPE_III)
BOX_CONFIDENCE = "box"
# Types whose box also gets the count of separate bursts (the IIIG hint).
COUNTED_TYPES: tuple[str, ...] = (TYPE_III, TYPE_IIIG)
NO_BOX_PARAMETERS = "burst parameters are calculated for Type II and Type III boxes only"


@dataclass
class BurstPhysics:
    """Physical parameters measured from one burst region."""

    # Extent of the burst itself, not of the box drawn around it.
    freq_start_mhz: float | None = None   # frequency at the burst's first moment
    freq_end_mhz: float | None = None     # frequency at its last moment
    freq_high_mhz: float | None = None
    freq_low_mhz: float | None = None
    time_start_s: float | None = None
    time_end_s: float | None = None
    duration_s: float | None = None
    bandwidth_mhz: float | None = None

    # Drift. Negative is the normal high-to-low progression.
    drift_mhz_per_s: float | None = None
    relative_drift_per_s: float | None = None  # (1/f)(df/dt), comparable across bands

    # How much to trust the above.
    fit_quality: float | None = None      # |Spearman rho| of the tracked ridge
    track_samples: int = 0
    track_axis: str = ""                  # "time" or "frequency"
    edge_clipped: bool = False            # burst touches the box edge
    confidence: str = "none"              # good | fair | poor | none
    note: str = ""

    # Separate bursts in the region: 1 for a single Type III, 3+ for a group.
    burst_count: int = 0

    @property
    def measured(self) -> bool:
        return self.drift_mhz_per_s is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def from_box(self) -> bool:
        """Whether these are a drawn box's parameters (see :func:`box_parameters`)."""
        return self.confidence == BOX_CONFIDENCE

    def describe(self) -> str:
        """One-line human summary, for the label panel and reports."""
        if not self.measured:
            return f"Not measurable ({self.note or 'no clean burst track in this region'})"
        source = (
            "from the box" if self.from_box
            else f"{self.confidence} fit, {self.track_samples} samples"
        )
        return (
            f"{self.freq_start_mhz:.1f} → {self.freq_end_mhz:.1f} MHz over "
            f"{self.duration_s:.1f} s · drift {self.drift_mhz_per_s:+.3f} MHz/s ({source})"
        )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 8-connected blob, i.e. the burst itself."""
    if not mask.any():
        return mask
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        if count <= 1:
            return mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(sizes.argmax())
    except ImportError:  # pragma: no cover - scipy is a dependency
        return mask


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float | None:
    """Median of pairwise slopes: robust to the outliers interference produces.

    The slope ``scipy.stats.theilslopes`` returns, computed the same way (every
    pair with a positive x step), without the confidence interval scipy also
    computes and this never uses: with the many repeated times of a track
    followed along frequency, that interval takes the square root of a negative
    number and warns. None when every ``x`` is the same -- a track confined to
    one time sample, a one-sample spike, has no slope (scipy warns four times
    over and returns NaN).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(np.ptp(x)) == 0.0:
        return None
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    rising = dx > 0
    return float(np.median(dy[rising] / dx[rising]))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation: how monotonic the tracked ridge is."""
    if x.size < 3:
        return 0.0
    # A constant track has no defined correlation, and scipy warns about it. That
    # is a real outcome here (a perfectly horizontal ridge), not an error.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    try:
        from scipy import stats

        value = float(stats.spearmanr(x, y).statistic)
    except Exception:
        return 0.0
    return 0.0 if math.isnan(value) else abs(value)


def _grade(quality: float, samples: int) -> str:
    if samples >= 12 and quality >= 0.8:
        return "good"
    if samples >= MIN_TRACK_SAMPLES and quality >= 0.5:
        return "fair"
    return "poor"


def box_parameters(
    axes: SpectrumAxes | None,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    burst_type: str | None,
    normalized: np.ndarray | None = None,
) -> BurstPhysics:
    """The burst parameters of a drawn box, from its geometry (see module docs).

    Only a Type II or Type III box gets frequencies, times and a drift rate:
    ``df/dt = (f_start - f_end) / (t_start - t_end)`` with the burst starting at
    the top of the box. A Type III or IIIG box also gets the count of separate
    bursts inside it when ``normalized`` is given -- the labelling hint between
    the two. Every other box returns an unmeasured result saying why.
    """
    from callisto_trainer.core.coords import box_to_physical

    physics = BurstPhysics()
    if normalized is not None and burst_type in COUNTED_TYPES:
        physics.burst_count = _count_in_box(normalized, axes, row0, row1, col0, col1)
    if burst_type not in BOX_DRIFT_TYPES:
        physics.note = NO_BOX_PARAMETERS
        return physics
    if axes is None:
        physics.note = "no frequency and time axes for this file"
        return physics

    bounds = box_to_physical(axes, row0, row1, col0, col1)
    f_start, f_end = bounds["freq_hi_mhz"], bounds["freq_lo_mhz"]
    t_start, t_end = bounds["t_start_s"], bounds["t_end_s"]
    physics.freq_start_mhz, physics.freq_end_mhz = f_start, f_end
    physics.freq_high_mhz, physics.freq_low_mhz = f_start, f_end
    physics.time_start_s, physics.time_end_s = t_start, t_end
    physics.duration_s = t_end - t_start
    physics.bandwidth_mhz = f_start - f_end
    physics.track_axis = BOX_CONFIDENCE
    if not (physics.duration_s > 0 and physics.bandwidth_mhz > 0):
        physics.note = "the box must span more than one channel and more than one sample"
        return physics

    physics.drift_mhz_per_s = float((f_start - f_end) / (t_start - t_end))
    physics.confidence = BOX_CONFIDENCE
    mid_freq = 0.5 * (f_start + f_end)
    if mid_freq > 0:
        physics.relative_drift_per_s = float(physics.drift_mhz_per_s / mid_freq)
    return physics


def box_parameters_current(stored: dict[str, Any] | None, burst_type: str | None) -> bool:
    """Whether a box's stored parameters follow :func:`box_parameters` for its type.

    False for a box measured by an earlier version (a pixel fit) or retyped
    since, so it can be recalculated.
    """
    stored = stored or {}
    if burst_type in BOX_DRIFT_TYPES:
        return stored.get("physics_confidence") == BOX_CONFIDENCE
    return stored.get("drift_mhz_per_s") is None and stored.get("physics_confidence") is not None


def _count_in_box(
    normalized: np.ndarray, axes: SpectrumAxes | None, row0: int, row1: int, col0: int, col1: int
) -> int:
    """Separate bursts in a box, at the same local level :func:`measure_burst` uses."""
    row0, col0 = max(0, int(row0)), max(0, int(col0))
    row1, col1 = min(int(normalized.shape[0]), int(row1)), min(int(normalized.shape[1]), int(col1))
    if row1 - row0 < 2 or col1 - col0 < 2:
        return 0
    window = normalized[row0:row1, col0:col1]
    if not np.isfinite(window).any():
        return 0
    level = max(float(MIN_BURST_LEVEL), float(np.percentile(window, BURST_PERCENTILE)))
    return count_bursts(
        window, level, cadence_seconds(axes),
        skip_rows=carrier_rows(normalized, row0, row1, col0, col1, level),
    )


def measure_burst(
    normalized: np.ndarray,
    axes: SpectrumAxes,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    percentile: float = BURST_PERCENTILE,
    min_level: float = MIN_BURST_LEVEL,
) -> BurstPhysics:
    """Measure a burst's extent and drift rate from the pixels of a region.

    What the model is given for every candidate region, whatever it is (see
    module docs); a drawn box's own parameters come from :func:`box_parameters`.
    ``normalized`` is the whole-file array from
    :func:`callisto_trainer.core.crops.normalize_full_spectrum`; the region is in
    the same pixel coordinates as the stored annotation.
    """
    physics = BurstPhysics()

    row0 = max(0, int(row0))
    col0 = max(0, int(col0))
    row1 = min(int(normalized.shape[0]), int(row1))
    col1 = min(int(normalized.shape[1]), int(col1))
    if row1 - row0 < 2 or col1 - col0 < 2:
        physics.note = "region too small to measure"
        return physics

    window = normalized[row0:row1, col0:col1]
    if not np.isfinite(window).any():
        physics.note = "no finite samples in region"
        return physics

    level = max(float(min_level), float(np.percentile(window, percentile)))
    # Counted over every bright pixel, before the largest-component step below
    # throws away all but one lane of a group -- but not over carriers.
    physics.burst_count = count_bursts(
        window, level, cadence_seconds(axes),
        skip_rows=carrier_rows(normalized, row0, row1, col0, col1, level),
    )
    mask = _largest_component(window >= level)
    if mask.sum() < MIN_TRACK_SAMPLES:
        physics.note = "no burst above the local threshold"
        return physics

    rows, cols = np.nonzero(mask)
    # Absolute pixel coordinates, then physical units via the real axes.
    abs_rows, abs_cols = rows + row0, cols + col0
    freqs = np.asarray(axes.freq_mhz, dtype=float)[np.clip(abs_rows, 0, axes.n_freq - 1)]
    times = np.asarray(axes.time_s, dtype=float)[np.clip(abs_cols, 0, axes.n_time - 1)]

    physics.freq_high_mhz = float(freqs.max())
    physics.freq_low_mhz = float(freqs.min())
    physics.time_start_s = float(times.min())
    physics.time_end_s = float(times.max())
    physics.duration_s = physics.time_end_s - physics.time_start_s
    physics.bandwidth_mhz = physics.freq_high_mhz - physics.freq_low_mhz
    physics.edge_clipped = bool(
        abs_rows.min() <= row0 or abs_rows.max() >= row1 - 1
        or abs_cols.min() <= col0 or abs_cols.max() >= col1 - 1
    )

    # Track along whichever axis the burst actually spans. A near-vertical Type
    # III occupies many rows but only a handful of columns, so a per-column fit
    # would have almost nothing to fit; the transpose is well conditioned.
    unique_cols = np.unique(abs_cols)
    unique_rows = np.unique(abs_rows)
    weights = window[rows, cols]

    if unique_cols.size >= unique_rows.size:
        physics.track_axis = "time"
        track_t, track_f = _ridge(abs_cols, freqs, weights, unique_cols, axes.time_s)
    else:
        physics.track_axis = "frequency"
        track_f, track_t = _ridge(abs_rows, times, weights, unique_rows, axes.freq_mhz)

    physics.track_samples = int(track_t.size)
    if track_t.size < MIN_TRACK_SAMPLES:
        physics.note = f"only {track_t.size} usable sample(s) along the {physics.track_axis} axis"
        physics.confidence = "none"
        return physics

    if float(np.ptp(track_t)) == 0.0:
        physics.note = "the whole track lies in one time sample, so it has no drift to fit"
        return physics
    slope = _theil_sen(track_t, track_f)
    if slope is None or not math.isfinite(slope):
        physics.note = "drift fit did not converge"
        return physics

    physics.drift_mhz_per_s = float(slope)
    physics.fit_quality = _spearman(track_t, track_f)
    physics.confidence = _grade(physics.fit_quality, physics.track_samples)

    # Start/end frequency from the fitted track at the burst's own start and end,
    # clamped to what was actually observed so the fit cannot extrapolate beyond
    # the data.
    order = np.argsort(track_t)
    first_t, last_t = float(track_t[order[0]]), float(track_t[order[-1]])
    intercept = float(np.median(track_f - slope * track_t))
    span = (physics.freq_low_mhz, physics.freq_high_mhz)
    physics.freq_start_mhz = float(np.clip(slope * first_t + intercept, *span))
    physics.freq_end_mhz = float(np.clip(slope * last_t + intercept, *span))

    mid_freq = 0.5 * (physics.freq_high_mhz + physics.freq_low_mhz)
    if mid_freq > 0:
        # Type III drift scales as ~f^1.84, so raw MHz/s is not comparable between
        # a 400 MHz and a 40 MHz observation; the relative rate largely is.
        physics.relative_drift_per_s = float(slope / mid_freq)
    return physics


def cadence_seconds(axes: SpectrumAxes | None, default: float = 0.25) -> float:
    """Median time step of a spectrum, in seconds."""
    if axes is None or axes.n_time < 2:
        return default
    step = float(np.median(np.diff(np.asarray(axes.time_s, dtype=float))))
    return step if math.isfinite(step) and step > 0 else default


# A channel bright for more than this fraction of the time outside a region is a
# carrier, not part of a burst.
CARRIER_PERSISTENCE = 0.3


def carrier_rows(
    normalized: np.ndarray, row0: int, row1: int, col0: int, col1: int, level: float
) -> np.ndarray:
    """Channels of a region that stay bright across the rest of the recording."""
    rows = np.asarray(normalized[row0:row1]) >= float(level)
    outside = rows.sum(axis=1) - rows[:, col0:col1].sum(axis=1)
    span = max(1, normalized.shape[1] - (col1 - col0))
    return outside / span > CARRIER_PERSISTENCE


def count_bursts(
    window: np.ndarray,
    level: float,
    cadence_s: float = 0.25,
    skip_rows: np.ndarray | None = None,
) -> int:
    """Number of separate bursts in a region, counted along time.

    A group of Type III bursts is several bright lanes one after another, so any
    frequency channel crossing the group sees one run of bright samples per lane.
    Counting runs per channel and taking the median over the channels that have
    any is indifferent to drift (the lanes are slanted, so a single column would
    see them overlap) and to a lane that fades out of a few channels. Short gaps
    inside a lane are closed first so noise does not split one burst in two.

    ``skip_rows`` marks channels to leave out -- carriers: a keyed transmitter
    crossing the box adds one "run" per on-period to its channel.

    Checked on 80 real Type III boxes: 55 count 1 and one counts 3 or more,
    which matches what they hold -- almost all are single bursts. Separate bursts
    are counted correctly; dense groups whose lanes saturate and merge are
    undercounted, which only keeps the IIIG hint quiet. The count means little
    for a Type II, whose fragmented and harmonic lanes give several runs per
    channel (60% of 80 real Type II boxes count 3 or more), so the hint is only
    ever offered for Type III and Type IIIG.
    """
    mask = np.asarray(window) >= float(level)
    if skip_rows is not None:
        mask = mask & ~np.asarray(skip_rows, dtype=bool)[:, None]
    if mask.ndim != 2 or not mask.any():
        return 0
    cadence = cadence_s if cadence_s and cadence_s > 0 else 0.25
    gap = max(1, int(round(BURST_GAP_S / cadence)))
    min_run = max(2, int(round(MIN_BURST_RUN_S / cadence)))

    counts: list[int] = []
    for row in mask:
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        starts, ends = edges[0::2], edges[1::2]
        if starts.size > 1:
            # A boundary between two runs survives only when the gap is long.
            separate = (starts[1:] - ends[:-1]) > gap
            starts = np.concatenate((starts[:1], starts[1:][separate]))
            ends = np.concatenate((ends[:-1][separate], ends[-1:]))
        runs = int(((ends - starts) >= min_run).sum())
        if runs:
            counts.append(runs)
    if not counts:
        return 0
    return int(math.floor(float(np.median(counts)) + 0.5))


def group_hint(physics: BurstPhysics | None, burst_type: str | None) -> str | None:
    """Suggest Type IIIG for a Type III box holding a group, and the reverse.

    Advisory only, like :func:`consistency_warning`: the count is a heuristic and
    the operator's judgement of what is one burst stays authoritative.
    """
    if physics is None:
        return None
    count = int(physics.burst_count or 0)
    if burst_type == TYPE_III and count >= IIIG_MIN_BURSTS:
        return (
            f"{count} separate bursts in this box. A group of {IIIG_MIN_BURSTS} or more "
            "Type III bursts is Type IIIG (key 3)."
        )
    if burst_type == TYPE_IIIG and 0 < count < 2:
        return (
            f"Only {count} burst found in this box; Type IIIG is a group of "
            f"{IIIG_MIN_BURSTS} or more Type III bursts."
        )
    return None


def _ridge(
    positions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    unique_positions: np.ndarray,
    axis_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted centroid of the burst at each position along one axis.

    Returns ``(axis_coordinate, centroid_value)`` in physical units. Weighting by
    intensity rather than taking the plain midpoint keeps the track on the bright
    core when a component is ragged.
    """
    axis_values = np.asarray(axis_values, dtype=float)
    coordinates: list[float] = []
    centroids: list[float] = []

    for position in unique_positions:
        selected = positions == position
        weight = weights[selected]
        total = float(weight.sum())
        if total <= 0:
            continue
        coordinates.append(float(axis_values[min(int(position), axis_values.size - 1)]))
        centroids.append(float((values[selected] * weight).sum() / total))

    return np.asarray(coordinates, dtype=float), np.asarray(centroids, dtype=float)


def expected_drift_range(burst_type: str) -> tuple[float, float] | None:
    """Published |df/dt| range for a burst type, or None when unconstrained."""
    return EXPECTED_DRIFT_RANGES.get(burst_type)


def consistency_warning(physics: BurstPhysics, burst_type: str) -> str | None:
    """Flag a measured drift that contradicts the assigned type.

    Advisory only. A poor fit is reported as such rather than as a contradiction,
    because an unreliable measurement is not evidence against a label.
    """
    if not physics.measured or physics.confidence in ("poor", "none"):
        return None
    expected = expected_drift_range(burst_type)
    if expected is None:
        return None

    magnitude = abs(physics.drift_mhz_per_s or 0.0)
    low, high = expected
    if magnitude < low:
        other = _closest_type(magnitude, exclude=burst_type)
        return (
            f"Measured drift {magnitude:.3f} MHz/s is slower than the {low}-{high} MHz/s "
            f"expected for {burst_type}"
            + (f"; it is in the {other} range." if other else ".")
        )
    if magnitude > high:
        other = _closest_type(magnitude, exclude=burst_type)
        return (
            f"Measured drift {magnitude:.3f} MHz/s is faster than the {low}-{high} MHz/s "
            f"expected for {burst_type}"
            + (f"; it is in the {other} range." if other else ".")
        )
    return None


def _closest_type(magnitude: float, exclude: str) -> str | None:
    for name, (low, high) in EXPECTED_DRIFT_RANGES.items():
        if name != exclude and low <= magnitude <= high:
            return name
    return None


# -- model features --------------------------------------------------------

# Order is fixed: it defines the physics branch's input layout and is recorded in
# exported model cards so a consumer can reproduce it.
PHYSICS_FEATURES = (
    "log_freq_start",
    "log_freq_end",
    "log_bandwidth",
    "log_duration",
    "signed_log_drift",
    "signed_log_relative_drift",
    "fit_quality",
    "measured",
)
NUM_PHYSICS_FEATURES = len(PHYSICS_FEATURES)


def _signed_log(value: float, scale: float = 1.0) -> float:
    """Compress a signed quantity spanning orders of magnitude.

    Drift rates run from ~0.005 to ~200 MHz/s across the burst types -- more than
    four decades -- so the raw number would let the largest values dominate a
    linear layer. The sign carries the physics (bursts normally drift downward)
    and must survive, hence signed log rather than log of the magnitude.
    """
    return float(math.copysign(math.log10(1.0 + abs(value) / scale), value))


def physics_to_vector(physics: BurstPhysics | None) -> np.ndarray:
    """Fixed-length float32 features for the model's physics branch.

    An unmeasurable region yields zeros with the ``measured`` flag off, so the
    network can learn to ignore the rest of the vector rather than treating a
    missing measurement as a real value of zero.
    """
    if physics is None or not physics.measured:
        return np.zeros(NUM_PHYSICS_FEATURES, dtype=np.float32)

    return np.array(
        [
            math.log10(1.0 + max(0.0, physics.freq_start_mhz or 0.0)),
            math.log10(1.0 + max(0.0, physics.freq_end_mhz or 0.0)),
            math.log10(1.0 + max(0.0, physics.bandwidth_mhz or 0.0)),
            math.log10(1.0 + max(0.0, physics.duration_s or 0.0)),
            _signed_log(physics.drift_mhz_per_s or 0.0, scale=0.01),
            _signed_log(physics.relative_drift_per_s or 0.0, scale=0.0001),
            float(physics.fit_quality or 0.0),
            1.0,
        ],
        dtype=np.float32,
    )


def physics_from_row(row: dict[str, Any]) -> BurstPhysics:
    """Rebuild a :class:`BurstPhysics` from a manifest row or database record."""

    def number(key: str) -> float | None:
        value = row.get(key)
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(parsed) else parsed

    return BurstPhysics(
        freq_start_mhz=number("freq_start_mhz"),
        freq_end_mhz=number("freq_end_mhz"),
        freq_high_mhz=number("freq_high_mhz"),
        freq_low_mhz=number("freq_low_mhz"),
        time_start_s=number("time_start_s"),
        time_end_s=number("time_end_s"),
        duration_s=number("duration_s"),
        bandwidth_mhz=number("bandwidth_mhz"),
        drift_mhz_per_s=number("drift_mhz_per_s"),
        relative_drift_per_s=number("relative_drift_per_s"),
        fit_quality=number("fit_quality"),
        track_samples=int(number("track_samples") or 0),
        track_axis=str(row.get("track_axis") or ""),
        edge_clipped=bool(int(number("edge_clipped") or 0)),
        confidence=str(row.get("physics_confidence") or "none"),
        burst_count=int(number("burst_count") or 0),
    )
