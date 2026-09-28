"""Name the interference among the regions that are not bursts, automatically.

Interference is not drawn. Every region the finder proposes in a no-burst file,
and every region of a burst file lying outside the drawn burst boxes, is already
known not to be a burst -- the operator's verdict and boxes say so. What is not
known is *which* of those rejections are interference and which are just
background. This module decides that from the region features (see
``region_features.py``), so the model has an RFI class to learn and the Predict
tab can report interference by name, without the operator marking any.

A region is RFI when it has one of these signatures; otherwise it stays
No_Burst. Checked in this order, so the first match names it:

* **impulse** -- lights most of the band in the same instant, one or two
  samples thick;
* **sweep** -- a thin, sparse track across much of the band (an ionosonde, a
  chirp), never lit all at once;
* **periodic** -- repeats with a regular period;
* **gain step** -- a flat plateau across most of the band, with no peak;
* **carrier** -- its channels stay bright outside the region for much of the
  recording, or it is a thin line lasting much of the file;
* **flagged channels** -- the station itself lists these channels as RFI.

## How the thresholds were set

Measured on 22,693 regions from 250 quiet and 305 burst files of the archive:

=====================================  =========  ==================
regions                                 labelled   note
=====================================  =========  ==================
matched to real Type III boxes              0.0%   must stay near 0
matched to real Type II boxes               0.9%
matched to real Other boxes                 0.3%
synthetic impulses / sweeps / steps    85 / 100 / 93%
synthetic periodic / carriers             98 / 87%
quiet-file candidates                      23.6%
unboxed burst-file candidates              27.0%
=====================================  =========  ==================

Inspected by eye, the quiet-file regions labelled periodic or carrier were
interference: the periodic calibration blocks in the lowest channels of many
stations, keyed carriers, repeating sweeps. The ones left No_Burst were mostly
weaker carrier fragments and noisy bands -- still rejections, just not named.
The rules are strict on purpose: calling a burst RFI would teach the model to
reject bursts, while leaving interference as No_Burst costs nothing, because
both are rejections and the burst decision uses them together.

Retrained on the archive with nothing changed but these labels (the same 22,544
samples; 10,256 of the 18,669 negatives named RFI) and judged on the same 310
held-out files: false alarms fell from 6 to 1 of 207 quiet files, burst files
found went from 77 to 79 of 103, and 7 of 440 test burst regions were called RFI.
"""

from __future__ import annotations

from typing import Sequence

IMPULSE = "impulse"
SWEEP = "sweep"
PERIODIC = "periodic"
GAIN_STEP = "gain step"
CARRIER = "carrier"
FLAGGED = "flagged channels"

KINDS: tuple[str, ...] = (IMPULSE, SWEEP, PERIODIC, GAIN_STEP, CARRIER, FLAGGED)

# A thin line lasting at least this much of the file is a carrier even when the
# region itself spans the recording (so nothing lies "outside" it to persist).
LONG_LINE_DURATION = 0.4
LONG_LINE_MAX_ROWS = 22


def interference_kind(values: dict[str, float] | None, box: Sequence[int]) -> str | None:
    """The interference signature of one region, or ``None`` for plain background.

    ``values`` are the region's :data:`~callisto_trainer.core.region_features.REGION_FEATURES`
    and ``box`` its ``(row0, row1, col0, col1)``.
    """
    if not values:
        return None
    rows = int(box[1]) - int(box[0])
    get = values.get
    if (
        get("band_fraction", 0.0) > 0.6
        and get("log_onset_spread", 1.0) < 0.1
        and get("log_thickness_time", 1.0) < 0.3
    ):
        return IMPULSE
    if (
        get("band_fraction", 0.0) > 0.4
        and get("fill_fraction", 1.0) < 0.08
        and get("log_thickness_time", 1.0) < 0.2
        and get("simultaneity", 1.0) < 0.3
    ):
        return SWEEP
    if get("periodicity", 0.0) > 0.5:
        return PERIODIC
    if get("peakiness", 1.0) < 0.1 and get("band_fraction", 0.0) > 0.6:
        return GAIN_STEP
    if get("outside_persistence", 0.0) > 0.4 or (
        get("duration_fraction", 0.0) > LONG_LINE_DURATION and rows <= LONG_LINE_MAX_ROWS
    ):
        return CARRIER
    if get("rfi_flag_fraction", 0.0) > 0.5:
        return FLAGGED
    return None
