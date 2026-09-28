"""Overlapping burst boxes: how the regions in and around them are labelled.

The operator's rules: where boxes of two types overlap, the smaller box's type
wins; boxes of the same type count together; a region split between two types
is left out; the drawn boxes themselves are cropped exactly as drawn.
"""

from __future__ import annotations

import numpy as np

from callisto_trainer.core.negatives import (
    assign_regions,
    box_owner_map,
    mine_negatives,
    union_containment,
)
from callisto_trainer.core.taxonomy import NO_BURST, TYPE_II, TYPE_III, TYPE_IV

SHAPE = (200, 3600)


class _Box:
    def __init__(self, row0, row1, col0, col1, burst_type, identifier):
        self.row0, self.row1, self.col0, self.col1 = row0, row1, col0, col1
        self.burst_type, self.id = burst_type, identifier


def _quiet(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)


def _label_at(assigned, row0: int, col0: int):
    """The assigned region whose box starts at (row0, col0)."""
    matches = [r for r in assigned if (r.row0, r.col0) == (row0, col0)]
    assert len(matches) == 1, f"expected one region at {(row0, col0)}, got {matches}"
    return matches[0]


# -- the pixel map ---------------------------------------------------------------


def test_the_smallest_box_owns_a_shared_pixel() -> None:
    big = _Box(0, 100, 0, 100, TYPE_IV, 1)
    small = _Box(20, 40, 20, 40, TYPE_III, 2)
    owner = box_owner_map((120, 120), [small, big])   # order must not matter
    assert owner[30, 30] == 1, "inside both: the smaller box (index 0 -> 1)"
    assert owner[80, 80] == 2, "only the big box"
    assert owner[110, 110] == 0, "no box"


def test_equal_boxes_resolve_deterministically() -> None:
    first, second = _Box(0, 10, 0, 10, TYPE_II, 1), _Box(0, 10, 0, 10, TYPE_III, 2)
    assert box_owner_map((10, 10), [first, second])[5, 5] == 2, "the one listed last"
    assert box_owner_map((10, 10), [first, second])[5, 5] == box_owner_map(
        (10, 10), [first, second]
    )[5, 5]


def test_boxes_beyond_the_array_are_clipped() -> None:
    owner = box_owner_map((50, 50), [_Box(-5, 20, 40, 80, TYPE_III, 1)])
    assert owner[:20, 40:].all() and not owner[20:, :].any()


def test_union_containment_counts_shared_pixels_once() -> None:
    region = (0, 10, 0, 100)
    halves = [(0, 10, 0, 50), (0, 10, 50, 100)]
    assert union_containment(region, halves) == 1.0
    overlapping = [(0, 10, 0, 60), (0, 10, 40, 60)]
    assert union_containment(region, overlapping) == 0.6
    assert union_containment(region, []) == 0.0


# -- region assignment -----------------------------------------------------------


def test_a_burst_drawn_inside_a_bigger_box_takes_the_smaller_type() -> None:
    """A Type III on a Type IV continuum: Type III on it, Type IV elsewhere."""
    array = _quiet()
    array[50:110, 1000:1030] = 0.9      # on the Type III, inside the Type IV
    array[50:110, 2000:2030] = 0.9      # elsewhere in the Type IV
    boxes = [_Box(10, 190, 100, 3000, TYPE_IV, 1), _Box(40, 120, 990, 1040, TYPE_III, 2)]
    assigned = assign_regions(array, boxes)

    on_iii = _label_at(assigned, 50, 1000)
    assert on_iii.label == TYPE_III and on_iii.overlap == "smaller box"
    assert on_iii.matched_box_id == 2
    elsewhere = _label_at(assigned, 50, 2000)
    assert elsewhere.label == TYPE_IV and elsewhere.overlap == ""


def test_a_region_where_two_bursts_cross_takes_the_smaller_box() -> None:
    array = _quiet()
    array[80:120, 500:530] = 0.9
    boxes = [_Box(60, 140, 200, 900, TYPE_II, 1), _Box(0, 200, 480, 560, TYPE_III, 2)]
    region = _label_at(assign_regions(array, boxes), 80, 500)
    small = min(boxes, key=lambda b: (b.row1 - b.row0) * (b.col1 - b.col0))
    assert region.label == small.burst_type == TYPE_III


def test_a_region_straddling_same_type_boxes_is_kept() -> None:
    """Neither box holds 60% of it, together they hold all of it."""
    array = _quiet()
    array[60:120, 1000:1040] = 0.9
    boxes = [_Box(50, 130, 900, 1022, TYPE_III, 1), _Box(50, 130, 1018, 1100, TYPE_III, 2)]
    region = _label_at(assign_regions(array, boxes), 60, 1000)
    assert region.label == TYPE_III and region.overlap == "joined boxes"
    assert region.containment == 1.0


def test_a_region_split_between_two_types_is_left_out() -> None:
    array = _quiet()
    array[60:120, 1000:1040] = 0.9
    boxes = [_Box(50, 130, 900, 1020, TYPE_II, 1), _Box(50, 130, 1020, 1100, TYPE_III, 2)]
    region = _label_at(assign_regions(array, boxes), 60, 1000)
    assert region.label is None and region.reason == "mixed types"


def test_a_single_box_behaves_as_before() -> None:
    array = _quiet()
    array[60:120, 1000:1040] = 0.9
    array[150:190, 3000:3040] = 0.9
    assigned = assign_regions(array, [_Box(40, 140, 980, 1060, TYPE_III, 1)])
    assert _label_at(assigned, 60, 1000).label == TYPE_III
    assert _label_at(assigned, 60, 1000).overlap == ""
    assert _label_at(assigned, 150, 3000).label == NO_BURST


def test_negatives_skip_a_region_straddling_two_boxes() -> None:
    """6% in each of two boxes is 12% on bursts: over the 10% rejection limit."""
    array = _quiet()
    array[60:110, 1000:1100] = 0.9
    boxes = [(0, 200, 994, 1000 + 6), (0, 200, 1094, 1200)]
    chosen = mine_negatives(array, exclude_boxes=boxes, max_negatives=5)
    assert not any((c.row0, c.col0) == (60, 1000) for c in chosen)
    alone = mine_negatives(array, exclude_boxes=boxes[:1], max_negatives=5)
    assert any((c.row0, c.col0) == (60, 1000) for c in alone), "sanity: one box alone allows it"


def test_a_detection_straddling_two_boxes_lands_on_the_bursts() -> None:
    from callisto_trainer.core.file_eval import lands_on_burst

    bursts = [(0, 100, 0, 110, TYPE_III), (0, 100, 110, 200, TYPE_III)]
    assert lands_on_burst((10, 20, 90, 130), bursts)
    assert not lands_on_burst((10, 20, 500, 600), bursts)
    assert not lands_on_burst((10, 20, 90, 130), [])
