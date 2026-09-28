"""The label taxonomy: what a box can be, and how the labels relate.

Kept in one Qt-free module because the class list reaches almost everywhere --
the keyboard shortcuts, the box colours, the exported snapshot, the checkpoint
config, the file-level decision -- and a second copy anywhere is how a class
silently goes missing from one of them.

Two relationships matter beyond the flat list:

* **Burst versus not-a-burst.** ``No_Burst`` and ``RFI`` are both rejections. A
  file is a burst only if some region is a burst *type*; a region classified RFI
  is reported as RFI, never as a detection. Neither is drawn by the operator:
  every region the finder proposes outside the drawn burst boxes, and every
  region of a no-burst file, is a rejection, and the exporter itself names the
  ones that measure like interference RFI (see ``core/rfi_labels.py``).
* **Fallbacks.** ``Type IIIG`` is a group of Type III bursts, ``Type IV`` was
  labelled ``Other`` before it had a class of its own, and ``RFI`` is a named
  kind of background. Each has a class it is folded into when there are too few
  examples to learn it on its own, so a fresh dataset can train from the first
  day instead of being blocked until every class is populated.
"""

from __future__ import annotations

NO_BURST = "No_Burst"
TYPE_II = "Type II"
TYPE_III = "Type III"
TYPE_IIIG = "Type IIIG"
TYPE_IV = "Type IV"
OTHER = "Other"
RFI = "RFI"

# Kinds of solar radio burst. Order is the one shown in the label panel.
BURST_TYPES: tuple[str, ...] = (TYPE_II, TYPE_III, TYPE_IIIG, TYPE_IV, OTHER)

# Everything a drawn box can be labelled. Position + 1 is the keyboard shortcut:
# 1 Type II, 2 Type III, 3 Type IIIG, 4 Type IV, 5 Other. Interference is not
# drawn; it is found automatically.
BOX_TYPES: tuple[str, ...] = BURST_TYPES

# Labels that mean "this region is not a burst".
NON_BURST_LABELS: tuple[str, ...] = (NO_BURST, RFI)

# A class and the class it is folded into when it has too few examples.
PARENT_LABEL: dict[str, str] = {TYPE_IIIG: TYPE_III, TYPE_IV: OTHER, RFI: NO_BURST}

# Class-id order of the unified model. No_Burst stays at index 0 so that older
# code reading "argmax != 0" keeps meaning "a burst"; the snapshot records the
# actual mapping, so a folded or absent class simply drops out of this order.
UNIFIED_LABEL_ORDER: tuple[str, ...] = (
    NO_BURST, RFI, TYPE_II, TYPE_III, TYPE_IIIG, TYPE_IV, OTHER,
)

# A Type III box holding at least this many separate bursts is a group (IIIG).
# Drives the labelling hint only; the model learns the class from the labels.
IIIG_MIN_BURSTS = 3

# A class with fewer drawn boxes (or, for RFI, samples) than this is folded into
# its fallback at export.
MIN_SUBCLASS_BOXES = 20


def is_burst_type(label: str | None) -> bool:
    return label in BURST_TYPES


def is_non_burst(label: str | None) -> bool:
    return label in NON_BURST_LABELS


def family(label: str) -> str:
    """The top-level class a label belongs to: Type IIIG is a Type III."""
    return TYPE_III if label == TYPE_IIIG else label


def fold_rare_subclasses(
    box_counts: dict[str, int], minimum: int = MIN_SUBCLASS_BOXES
) -> tuple[dict[str, str], dict[str, str]]:
    """Which class every label trains as, folding subclasses with too few boxes.

    Returns ``(mapping, folded)``: ``mapping`` sends every known label to its
    training class, and ``folded`` lists the classes that were merged into their
    fallback because fewer than ``minimum`` boxes of them were drawn. A class
    learned from a handful of examples is memorised, not learned, and it would
    also block training outright by leaving a split with none of it.

    ``RFI`` is only folded when its count is given: it is labelled automatically
    after the boxes are counted, so the exporter decides it separately.
    """
    mapping = {label: label for label in (NO_BURST, RFI, *BOX_TYPES)}
    folded: dict[str, str] = {}
    for child, parent in PARENT_LABEL.items():
        if child == RFI and RFI not in box_counts:
            continue
        if int(box_counts.get(child, 0)) < int(minimum):
            mapping[child] = parent
            folded[child] = parent
    return mapping, folded


def ordered_classes(present: set[str] | list[str]) -> dict[str, int]:
    """Class-id mapping for the labels present, in the unified order.

    Labels outside the known taxonomy are appended alphabetically rather than
    dropped, so an unexpected name surfaces in the snapshot instead of vanishing.
    """
    present = set(present)
    names = [name for name in UNIFIED_LABEL_ORDER if name in present]
    names += sorted(present - set(UNIFIED_LABEL_ORDER))
    return {name: index for index, name in enumerate(names)}
