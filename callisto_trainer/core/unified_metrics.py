"""Region-level metrics for the unified model, beyond macro-F1.

Macro-F1 over every class answers "is the argmax right", weighting a rare burst
type exactly as heavily as rejecting interference, and it never looks at how
the model *ranks* regions. But at inference the decision is a threshold on burst
evidence, tuned afterwards to a false-alarm budget (see ``file_eval``), so what
training should select for is:

* **detection** -- does burst evidence rank real bursts above background and
  RFI? Average precision measures exactly that, at every threshold at once;
* **typing** -- among real bursts, is the type right? Scored with the argmax
  over the burst classes only, because typing is only ever asked of a region
  that has already been called a burst.

``unified_score`` is their mean, and is what the unified trainer monitors.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from callisto_trainer.core.taxonomy import NO_BURST, NON_BURST_LABELS, RFI, family


def merge_rejections(
    labels: Sequence[int] | np.ndarray, class_names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Report RFI and No_Burst as one "not a burst" class.

    The unified model is trained with RFI as its own rejection class, because
    that split measurably cut false alarms, but the operator reads one outcome:
    not a burst. Returns the labels re-indexed onto the class list without RFI,
    and that list. A class list without both classes is returned unchanged.
    """
    names = list(class_names)
    labels = np.asarray(labels, dtype=int)
    if RFI not in names or NO_BURST not in names:
        return labels, names
    merged = [name for name in names if name != RFI]
    lookup = np.array([merged.index(NO_BURST if name == RFI else name) for name in names])
    return (lookup[labels] if labels.size else labels), merged


def class_groups(class_names: Sequence[str]) -> tuple[list[int], list[int]]:
    """``(non_burst_indices, burst_indices)`` for a class list."""
    non_burst = [i for i, name in enumerate(class_names) if name in NON_BURST_LABELS]
    burst = [i for i, name in enumerate(class_names) if name not in NON_BURST_LABELS]
    return non_burst, burst


def burst_evidence(probabilities: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """``1 - P(not a burst)`` per region: the quantity the file decision thresholds."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    non_burst, _ = class_groups(class_names)
    if not non_burst:
        return np.ones(probabilities.shape[0])
    return np.clip(1.0 - probabilities[:, non_burst].sum(axis=1), 0.0, 1.0)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true.tolist()))
    if not labels:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def unified_region_metrics(
    y_true: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    """Detection and typing quality for one split. Empty when there is no background."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    non_burst, burst = class_groups(class_names)
    if not non_burst or not burst or y_true.size == 0:
        return {}

    is_burst = np.isin(y_true, burst)
    evidence = burst_evidence(probabilities, class_names)
    metrics: dict[str, Any] = {}
    if is_burst.any() and (~is_burst).any():
        metrics["detection_ap"] = float(average_precision_score(is_burst, evidence))
        metrics["detection_auc"] = float(roc_auc_score(is_burst, evidence))
    else:
        metrics["detection_ap"] = float("nan")
        metrics["detection_auc"] = float("nan")

    if is_burst.any():
        typed = np.asarray(burst)[probabilities[is_burst][:, burst].argmax(axis=1)]
        truth = y_true[is_burst]
        metrics["type_macro_f1"] = _macro_f1(truth, typed)
        # Type IIIG is a Type III: confusing the two is a subclass slip, not a
        # typing failure, and this is the number that says so.
        to_family = np.vectorize(lambda index: family(class_names[int(index)]))
        metrics["family_type_macro_f1"] = _macro_f1(to_family(truth), to_family(typed))
    else:
        metrics["type_macro_f1"] = float("nan")
        metrics["family_type_macro_f1"] = float("nan")

    if RFI in class_names:
        rfi_index = list(class_names).index(RFI)
        rfi_rows = y_true == rfi_index
        if rfi_rows.any():
            rejected = np.isin(probabilities[rfi_rows].argmax(axis=1), non_burst)
            metrics["rfi_rejected"] = float(rejected.mean())

    parts = [metrics["detection_ap"], metrics["type_macro_f1"]]
    finite = [value for value in parts if np.isfinite(value)]
    metrics["unified_score"] = float(np.mean(finite)) if finite else float("nan")
    return metrics
