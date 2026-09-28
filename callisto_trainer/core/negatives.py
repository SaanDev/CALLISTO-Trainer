"""Mining the not-a-burst regions for the unified model.

The unified model classifies a *region*, so its No_Burst class has to be made of
regions too. Which regions matters enormously.

At inference the model only ever sees candidate regions produced by the
brightness finder, and those are not empty background -- they are the brightest
things in the file: carrier lines, receiver interference, calibration artifacts.
Measured across this archive, quiet files yield a median of 12 such regions, more
than burst files do. If the negatives were random background patches the model
would learn a trivially easy task and still be fooled by everything the finder
actually proposes.

So negatives are mined with the same finder used at inference, from two places:

* **confirmed no-burst files** -- whatever the finder proposes there is, by the
  operator's own judgement, not a burst;
* **unboxed regions of burst files** -- interference sitting alongside a genuine
  burst, which is the hardest and most valuable negative of all. Anything
  overlapping a drawn box is excluded, so a real burst is never mislabelled.

Each mined region is a rejection. Whether it is named RFI or left No_Burst is
decided afterwards from its measured features (see ``core/rfi_labels.py``).

The regions inference will actually examine -- the largest few, see
``region_finder.DEFAULT_MAX_REGIONS`` -- are always mined first. A negative the
model is never shown at inference teaches it little about the false positives
that matter; the ones it *is* shown are exactly where false positives come from.

When a previously trained model is supplied, candidates are ranked by how
strongly *it* called them a burst (hard-negative mining), so each retraining
concentrates on the previous model's own false positives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from callisto_trainer.core.crops import PixelBox
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.region_finder import DEFAULT_MAX_REGIONS, Proposal, resolve_threshold
from callisto_trainer.core.taxonomy import NO_BURST

LOGGER = get_logger(__name__)

# A candidate with more than this fraction of itself inside a drawn box is
# assumed to be that burst, and is dropped rather than used as a negative.
DEFAULT_CONTAINMENT_REJECT = 0.10
# Per-file caps keep one busy recording from dominating the negative set. Sized
# so the inference-visible regions of a file fit within them.
DEFAULT_MAX_PER_QUIET_FILE = 16
DEFAULT_MAX_PER_BURST_FILE = 8
# How many candidates to consider per file before choosing negatives.
MINING_POOL = 2 * DEFAULT_MAX_REGIONS
# A candidate inside a burst box whose channels are bright for more than this
# fraction of the time *outside* the box is a carrier fragment, not the burst.
DEFAULT_MAX_OUTSIDE_PERSISTENCE = 0.5
# Line-shaped: at most this many channels tall and at least this many times wider
# than tall. Such a candidate inside a burst box is never used as a burst sample.
LINE_MAX_ROWS = 22
LINE_MIN_ASPECT = 3.0


def is_line_shaped(row0: int, row1: int, col0: int, col1: int) -> bool:
    """A thin horizontal candidate: the shape of a carrier segment (or a thin lane)."""
    rows, cols = row1 - row0, col1 - col0
    return rows <= LINE_MAX_ROWS and cols >= LINE_MIN_ASPECT * rows

# Scores candidates by burst evidence under a previously trained model.
CandidateScorer = Callable[[Sequence[Proposal]], Sequence[float]]


@dataclass(frozen=True)
class MinedNegative:
    """One background region selected as a No_Burst training crop."""

    row0: int
    row1: int
    col0: int
    col1: int
    area: int
    peak: float
    source: str  # "quiet_file" | "burst_file_background"

    def as_box(self) -> PixelBox:
        return PixelBox(self.row0, self.row1, self.col0, self.col1)


def _intersection(a: Sequence[int], b: Sequence[int]) -> int:
    rows = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    cols = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    return rows * cols


def box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    """Intersection-over-union of two ``(row0, row1, col0, col1)`` boxes."""
    intersection = _intersection(a, b)
    if intersection == 0:
        return 0.0
    area_a = max(0, a[1] - a[0]) * max(0, a[3] - a[2])
    area_b = max(0, b[1] - b[0]) * max(0, b[3] - b[2])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def containment(candidate: Sequence[int], reference: Sequence[int]) -> float:
    """Fraction of ``candidate`` lying inside ``reference``.

    The right measure for matching finder regions to drawn boxes, because the two
    are wildly different sizes: measured on this archive a drawn box has a median
    area of ~46,900 px while a candidate region has ~388 px, 121x smaller. IoU
    between them sits around 0.003 even when the candidate is *entirely* inside
    the burst, so an IoU rule discards almost every genuine match. Containment
    asks the question that actually matters -- is this region inside a burst --
    and is 1.0 for that same case.
    """
    intersection = _intersection(candidate, reference)
    if intersection == 0:
        return 0.0
    area = max(0, candidate[1] - candidate[0]) * max(0, candidate[3] - candidate[2])
    return float(intersection / area) if area else 0.0


def union_containment(candidate: Sequence[int], references: Iterable[Sequence[int]]) -> float:
    """Fraction of ``candidate`` lying inside *any* of ``references``.

    :func:`containment` against one box at a time undercounts a region that
    straddles two overlapping boxes: 40% in each is 80% inside burst boxes, yet
    neither alone reaches a positive or a rejection threshold. Pixels shared by
    two boxes count once.
    """
    row0, row1, col0, col1 = (int(value) for value in candidate[:4])
    height, width = row1 - row0, col1 - col0
    if height <= 0 or width <= 0:
        return 0.0
    covered = np.zeros((height, width), dtype=bool)
    for reference in references:
        r0, r1 = max(int(reference[0]), row0) - row0, min(int(reference[1]), row1) - row0
        c0, c1 = max(int(reference[2]), col0) - col0, min(int(reference[3]), col1) - col0
        if r1 > r0 and c1 > c0:
            covered[r0:r1, c0:c1] = True
    return float(covered.mean())


def box_owner_map(shape: tuple[int, ...], boxes: Sequence[Any]) -> np.ndarray:
    """Which box owns each pixel: its index + 1, or 0 outside every box.

    Where boxes overlap the **smallest** box owns the pixel -- the operator's
    decision: a Type III drawn inside a Type II or Type IV box is that Type III,
    and the rest of the big box keeps its own type. Boxes are painted largest
    first so smaller ones land on top; equal areas go to the one listed last.
    """
    owner = np.zeros((int(shape[0]), int(shape[1])), dtype=np.int32)

    def area(box) -> int:
        return max(0, box.row1 - box.row0) * max(0, box.col1 - box.col0)

    for index in sorted(range(len(boxes)), key=lambda i: (-area(boxes[i]), i)):
        box = boxes[index]
        owner[max(0, box.row0):max(0, box.row1), max(0, box.col0):max(0, box.col1)] = index + 1
    return owner


def mine_negatives(
    normalized: np.ndarray,
    exclude_boxes: Iterable[Sequence[int]] = (),
    max_negatives: int = DEFAULT_MAX_PER_QUIET_FILE,
    source: str = "quiet_file",
    threshold: float | None = None,
    min_area: int | None = None,
    containment_reject: float = DEFAULT_CONTAINMENT_REJECT,
    seed: int = 0,
    scorer: CandidateScorer | None = None,
) -> list[MinedNegative]:
    """Propose No_Burst crops from one normalized spectrum.

    Uses the inference-time region finder so the negatives match the distribution
    the model will actually be asked about. Regions overlapping ``exclude_boxes``
    are dropped, which is what makes mining inside burst files safe.

    Selection order: the regions inference examines come first, hardest first;
    the remaining budget goes half to the hardest of the rest and half to a
    random sample of it, for variety. "Hardest" is the ``scorer``'s burst
    evidence when one is given, brightness otherwise.
    """
    from callisto_trainer.core.region_finder import DEFAULT_MIN_AREA, find_candidate_regions

    if max_negatives <= 0:
        return []

    effective_threshold = (
        resolve_threshold(normalized) if threshold is None else float(threshold)
    )
    proposals = find_candidate_regions(
        normalized,
        threshold=effective_threshold,
        min_area=DEFAULT_MIN_AREA if min_area is None else int(min_area),
        # Look at more than needed, so the choice below has something to choose
        # from after the overlapping ones are removed.
        max_candidates=max(MINING_POOL, max_negatives + 4),
    )
    if not proposals:
        return []

    excluded = [tuple(box) for box in exclude_boxes]
    kept: list[Proposal] = []
    for proposal in proposals:
        candidate = (proposal.row0, proposal.row1, proposal.col0, proposal.col1)
        # Containment, not IoU: a candidate sitting wholly inside a large drawn
        # box scores an IoU of ~0.003 and would sail through an IoU filter, which
        # would label genuine burst signal as background. Measured over all the
        # boxes at once, so straddling two overlapping boxes counts in full.
        if excluded and union_containment(candidate, excluded) > containment_reject:
            continue
        kept.append(proposal)

    hardness = (
        [float(value) for value in scorer(kept)] if scorer is not None and kept
        else [float(item.peak) for item in kept]
    )
    return [
        MinedNegative(
            row0=item.row0,
            row1=item.row1,
            col0=item.col0,
            col1=item.col1,
            area=item.area,
            peak=item.peak,
            source=source,
        )
        for item in select_negatives(kept, hardness, max_negatives, seed)
    ]


def select_negatives(
    candidates: Sequence[Any], hardness: Sequence[float], limit: int, seed: int = 0
) -> list[Any]:
    """Choose up to ``limit`` negatives from ``candidates`` (largest first).

    The first :data:`DEFAULT_MAX_REGIONS` are what inference examines, so they
    are taken first, hardest first. Any remaining budget goes half to the hardest
    of the rest and half to a seeded random sample of it, for variety without
    over-fitting to one file's quirks.
    """
    if limit <= 0 or not candidates:
        return []
    indexed = list(range(len(candidates)))
    visible = sorted(indexed[:DEFAULT_MAX_REGIONS], key=lambda i: -hardness[i])
    chosen = visible[:limit]
    budget = limit - len(chosen)
    if budget > 0:
        rest = sorted(indexed[DEFAULT_MAX_REGIONS:], key=lambda i: -hardness[i])
        hard = rest[: max(1, budget // 2)] if rest else []
        remainder = rest[len(hard):]
        rng = random.Random(seed)
        chosen += hard + rng.sample(remainder, min(len(remainder), budget - len(hard)))
    return [candidates[i] for i in chosen]


@dataclass(frozen=True)
class AssignedRegion:
    """A finder-proposed region and the label matching it to the drawn boxes."""

    row0: int
    row1: int
    col0: int
    col1: int
    area: int
    peak: float
    label: str | None  # a box type, "No_Burst", or None when ambiguous
    containment: float
    matched_box_id: int | None = None
    # Why an ambiguous region was dropped, for the snapshot's accounting.
    reason: str = ""
    # How overlapping boxes decided a labelled region, when they did:
    # "smaller box" (inside boxes of two types; the smaller one's type) or
    # "joined boxes" (inside same-type boxes together, but no one of them).
    overlap: str = ""

    def as_box(self) -> PixelBox:
        return PixelBox(self.row0, self.row1, self.col0, self.col1)


def _outside_persistence(context: Any, row0: int, row1: int, col0: int, col1: int) -> float:
    """Mean fraction of time a region's bright channels stay bright elsewhere."""
    bright = context.bright[row0:row1, col0:col1]
    rows = np.flatnonzero(bright.any(axis=1))
    if rows.size == 0:
        return 0.0
    inside = bright[rows].sum(axis=1)
    outside = context.row_bright[row0 + rows] - inside
    return float(np.mean(outside / float(max(1, context.n_time - (col1 - col0)))))


def assign_regions(
    normalized: np.ndarray,
    boxes: Sequence[Any],
    positive_containment: float = 0.60,
    negative_containment: float = 0.10,
    max_candidates: int = 24,
    threshold: float | None = None,
    min_area: int | None = None,
    context: Any = None,
    max_outside_persistence: float = DEFAULT_MAX_OUTSIDE_PERSISTENCE,
    drop_lines: bool = True,
) -> list[AssignedRegion]:
    """Label finder-proposed regions by their overlap with the drawn boxes.

    This exists to close a train/serve gap that is easy to create and hard to
    see. If positives are the operator's hand-drawn boxes while negatives are
    finder output, the two classes differ in *how the region was produced* as
    well as in what is inside it -- and a model will happily learn that shortcut.
    Measured on a model trained that way: 0.95 burst recall on held-out crops, and
    1 detection in 12 burst files at inference, because every finder region looked
    like a negative.

    Labelling finder regions by containment in the boxes puts both classes on the
    same footing, the way an object detector assigns anchors. In order:

    1. inside a **burst** box -> that burst type, *unless* its channels stay
       bright for much of the recording outside the box (needs ``context``), or
       it is line-shaped (:func:`is_line_shaped`). Either is dropped, not
       labelled a burst. Boxes here average ~1,000 columns, so they routinely
       contain interference, and a carrier crossing the box inherited its type.
       Measured on 2,103 real files: 36% of the finder regions labelled Type II
       this way were line-shaped, against 1.3% of the drawn Type II boxes, and
       about a third of those lines were interference -- which is how a trained
       model learned "thin horizontal segment = Type II" and flagged carrier
       segments at up to 0.99. No measurement of the segment itself (flatness,
       slant, continuation along its channels) told the interference from the
       genuinely thin Type II lanes, so line-shaped regions are left out of the
       burst classes altogether (``drop_lines``); the drawn box still teaches its
       burst. Retrained and judged on 310 held-out files, this took line-shaped
       false alarms and strays from 7 regions to 0, with false-alarm files
       unchanged (6) and 5 fewer burst files found -- on inspection one a real
       faint Type II, one an "Other" box drawn around a dotted carrier, three
       doubtful detections in interference-heavy files;
    2. outside every box -> No_Burst (the exporter then names the interference
       among these RFI, see ``core/rfi_labels.py``);
    3. anything in between is dropped rather than guessed at: a half-overlapping
       region is a clean example of nothing.

    **Overlapping boxes.** "Inside" is measured on a pixel map in which every
    pixel belongs to the smallest box covering it (:func:`box_owner_map`), and a
    region takes the type owning at least ``positive_containment`` of it. So:

    * where boxes of two types overlap, the **smaller box's type** wins -- a
      Type III drawn on a Type IV continuum is Type III there, Type IV elsewhere;
    * boxes of the **same type** count together: a region straddling two
      overlapping Type III boxes is Type III even if neither holds 60% of it;
    * a region inside the boxes but split between two types, none of them
      owning 60% of it, is dropped ("mixed types").

    Measured on the 45 readable archive files with overlapping boxes (1,928
    candidate regions), judging one box at a time had labelled regions inside
    two boxes of different types by whichever was listed first, and dropped
    regions straddling boxes. Under these rules 43 labels change: 32 regions
    dropped before are now Type III (straddling Type III boxes), 5 change from
    Other to the Type III drawn inside the Other box, 3 split between two types
    are left out, and 3 that were No_Burst though over 10% on bursts are
    dropped. The drawn boxes themselves are still cropped exactly as drawn,
    overlap included: at prediction bursts sit together too.

    Every box passed is a burst box; interference is never drawn.
    """
    from callisto_trainer.core.region_finder import DEFAULT_MIN_AREA, find_candidate_regions

    effective_threshold = (
        resolve_threshold(normalized) if threshold is None else float(threshold)
    )
    proposals = find_candidate_regions(
        normalized,
        threshold=effective_threshold,
        min_area=DEFAULT_MIN_AREA if min_area is None else int(min_area),
        max_candidates=max_candidates,
    )

    owner = box_owner_map(normalized.shape, boxes) if boxes else None

    assigned: list[AssignedRegion] = []
    for proposal in proposals:
        candidate = (proposal.row0, proposal.row1, proposal.col0, proposal.col1)
        reason, how = "", ""
        if owner is None:
            union, top_type, share, matched = 0.0, None, 0.0, None
        else:
            window = owner[proposal.row0:proposal.row1, proposal.col0:proposal.col1]
            counts = np.bincount(window.ravel(), minlength=len(boxes) + 1)
            total = float(max(1, window.size))
            union = 1.0 - counts[0] / total
            by_type: dict[str, float] = {}
            for index, box in enumerate(boxes, start=1):
                if counts[index]:
                    by_type[box.burst_type] = by_type.get(box.burst_type, 0.0) + counts[index] / total
            top_type, share = max(by_type.items(), key=lambda item: item[1], default=(None, 0.0))
            owned = [i for i, box in enumerate(boxes, start=1) if box.burst_type == top_type]
            matched = boxes[max(owned, key=lambda i: counts[i]) - 1].id if owned else None

        if top_type is not None and share >= positive_containment:
            label, overlap = top_type, share
            single = [
                (containment(candidate, (b.row0, b.row1, b.col0, b.col1)), b.burst_type)
                for b in boxes
            ]
            if any(c >= positive_containment and t != top_type for c, t in single):
                how = "smaller box"
            elif max(c for c, _ in single) < positive_containment:
                how = "joined boxes"
            if (
                context is not None
                and _outside_persistence(context, *candidate) > max_outside_persistence
            ):
                label, reason = None, "carrier inside a burst box"
            elif drop_lines and is_line_shaped(*candidate):
                label, reason = None, "line inside a burst box"
        elif union <= negative_containment:
            label, matched, overlap = NO_BURST, None, union
        elif union >= positive_containment:
            label, overlap, reason = None, union, "mixed types"
        else:
            label, overlap, reason = None, union, "partial overlap"

        assigned.append(
            AssignedRegion(
                row0=proposal.row0,
                row1=proposal.row1,
                col0=proposal.col0,
                col1=proposal.col1,
                area=proposal.area,
                peak=proposal.peak,
                label=label,
                containment=overlap,
                matched_box_id=matched,
                reason=reason,
                overlap=how if label is not None else "",
            )
        )
    return assigned


def negative_budget(
    positive_count: int,
    quiet_files: int,
    burst_files: int,
    target_ratio: float = 1.0,
) -> tuple[int, int]:
    """Per-file caps that aim for roughly ``target_ratio`` negatives per positive.

    Returns ``(per_quiet_file, per_burst_file)``. Most negatives should come from
    confirmed quiet files; burst files contribute a smaller number of hard ones.
    Both are clamped to sane bounds so a tiny quiet set does not force absurd
    per-file counts.

    The ratio is a *target*, not a guarantee: a quiet file can only give as many
    negatives as its finder proposes. Background outnumbers bursts by far in real
    use, so the unified export asks for several negatives per positive.
    """
    wanted = max(1, int(round(positive_count * target_ratio)))
    from_burst = min(wanted // 3, burst_files * DEFAULT_MAX_PER_BURST_FILE)
    from_quiet = max(0, wanted - from_burst)

    per_quiet = (
        int(np.ceil(from_quiet / quiet_files)) if quiet_files else 0
    )
    per_burst = int(np.ceil(from_burst / burst_files)) if burst_files else 0
    return (
        int(np.clip(per_quiet, 1, 48)) if quiet_files else 0,
        int(np.clip(per_burst, 0, DEFAULT_MAX_PER_BURST_FILE)) if burst_files else 0,
    )
