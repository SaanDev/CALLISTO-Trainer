"""Synthetic interference injected into quiet recordings, as extra RFI examples.

Real interference in the labelled set is whatever happened to be recorded and
marked. Some of the shapes that most often fool a burst classifier -- periodic
pulse trains, swept signals, broadband impulses, gain steps -- can be rare in a
given archive, so the model sees too few of them to learn they are not bursts.

This module paints those shapes onto a confirmed quiet file's normalized
spectrum. The exporter then runs the ordinary region finder over the result and
keeps the candidates that fall on the injected pattern as RFI samples, so every
synthetic example goes through exactly the path a real one does: found by the
same finder, cropped the same way, measured by the same features.

Every pattern is kept deliberately *unlike* a solar burst on the axis that
separates them: impulses and sweeps are one sample thin with no decay, carriers
do not drift, periodic trains are exactly regular. A synthetic shape that
resembled a real burst would teach the model to reject real bursts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATTERNS = ("carrier", "impulses", "periodic", "sweep", "band_step")


@dataclass
class InjectedRFI:
    """The spectrum with interference painted on, and where it was painted."""

    normalized: np.ndarray   # float32 [freq, time], clipped to [0, 1]
    mask: np.ndarray         # bool [freq, time], True where the pattern was drawn
    pattern: str


def _level(rng: np.random.Generator) -> float:
    return float(rng.uniform(0.55, 1.0))


def _carrier(shape, rng, cadence_s) -> np.ndarray:
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        row = int(rng.integers(0, n_freq))
        height = int(rng.integers(1, 4))
        span = int(n_time * rng.uniform(0.4, 1.0))
        start = int(rng.integers(0, max(1, n_time - span)))
        line = np.zeros(n_time, dtype=bool)
        line[start:start + span] = True
        if rng.random() < 0.4:
            # Intermittent: keyed on and off, as many transmitters are.
            duty = int(max(2, rng.uniform(5.0, 40.0) / cadence_s))
            line &= (np.arange(n_time) // duty) % 2 == 0
        mask[row:row + height] |= line
    return mask


def _impulses(shape, rng, cadence_s) -> np.ndarray:
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(int(rng.integers(1, 7))):
        col = int(rng.integers(0, n_time))
        band = int(n_freq * rng.uniform(0.6, 1.0))
        top = int(rng.integers(0, max(1, n_freq - band + 1)))
        mask[top:top + band, col:col + int(rng.integers(1, 3))] = True
    return mask


def _periodic(shape, rng, cadence_s) -> np.ndarray:
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    period = int(max(3, rng.uniform(1.0, 20.0) / cadence_s))
    width = int(rng.integers(1, 3))
    if rng.random() < 0.5:
        top, bottom = 0, n_freq                      # broadband pulses
    else:
        height = int(rng.integers(5, max(6, n_freq // 4)))
        top = int(rng.integers(0, max(1, n_freq - height)))
        bottom = top + height                        # narrowband pulses
    span = int(n_time * rng.uniform(0.3, 1.0))
    start = int(rng.integers(0, max(1, n_time - span)))
    for col in range(start, start + span, period):
        mask[top:bottom, col:col + width] = True
    return mask


def _sweep(shape, rng, cadence_s) -> np.ndarray:
    """Thin straight tracks across the band, like an ionosonde or a chirp."""
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    duration = int(max(4, rng.uniform(2.0, 40.0) / cadence_s))
    repeats = int(rng.integers(1, 4))
    gap = int(max(duration + 4, rng.uniform(30.0, 120.0) / cadence_s))
    start = int(rng.integers(0, max(1, n_time - repeats * gap)))
    rising = rng.random() < 0.5
    rows = np.arange(n_freq)
    for index in range(repeats):
        offset = start + index * gap
        cols = offset + np.round(rows / max(1, n_freq - 1) * duration).astype(int)
        if rising:
            cols = cols[::-1]
        valid = (cols >= 0) & (cols < n_time)
        mask[rows[valid], cols[valid]] = True
    return mask


def _band_step(shape, rng, cadence_s) -> np.ndarray:
    """A flat elevation across most of the band: gain change, switching, saturation."""
    n_freq, n_time = shape
    mask = np.zeros(shape, dtype=bool)
    band = int(n_freq * rng.uniform(0.6, 1.0))
    top = int(rng.integers(0, max(1, n_freq - band + 1)))
    span = int(n_time * rng.uniform(0.1, 0.5))
    start = int(rng.integers(0, max(1, n_time - span)))
    mask[top:top + band, start:start + span] = True
    return mask


_BUILDERS = {
    "carrier": _carrier,
    "impulses": _impulses,
    "periodic": _periodic,
    "sweep": _sweep,
    "band_step": _band_step,
}


def inject_rfi(
    normalized: np.ndarray,
    rng: np.random.Generator,
    pattern: str | None = None,
    cadence_s: float = 0.25,
) -> InjectedRFI:
    """Paint one interference pattern onto a copy of ``normalized``."""
    data = np.asarray(normalized, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D normalized spectrum, got {data.shape}")
    pattern = pattern or str(rng.choice(PATTERNS))
    if pattern not in _BUILDERS:
        raise ValueError(f"Unknown RFI pattern {pattern!r}; expected one of {PATTERNS}")

    cadence = cadence_s if cadence_s > 0 else 0.25
    mask = _BUILDERS[pattern](data.shape, rng, cadence)
    if pattern == "band_step":
        # A step raises the existing background rather than replacing it.
        painted = data + np.float32(rng.uniform(0.3, 0.5)) * mask
    else:
        level = _level(rng)
        jitter = rng.normal(0.0, 0.03, size=data.shape).astype(np.float32)
        painted = np.where(mask, np.maximum(data, level + jitter), data)
    return InjectedRFI(
        normalized=np.clip(painted, 0.0, 1.0).astype(np.float32),
        mask=mask,
        pattern=pattern,
    )
