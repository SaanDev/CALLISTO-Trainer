"""Multiclass metrics for the burst-type classifier (Type II / III / Other)."""

# NOTE: Vendored from H:\Burst Identifier (src/evaluation/type_metrics.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, per-class scores and a confusion matrix.

    ``class_names`` is ordered by label id (index 0 -> class id 0). ``y_prob``
    (softmax probabilities, shape ``[N, K]``) is accepted for API symmetry with
    :func:`compute_binary_metrics` but is not required for the reported metrics.
    Mirrors the return-shape spirit of the binary metrics helper: a flat dict of
    floats/ints plus nested per-class and confusion-matrix structures.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    num_classes = len(class_names)
    labels = list(range(num_classes))

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=num_classes)

    per_class = {
        class_names[i]: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": per_class,
        "class_names": list(class_names),
        "confusion_matrix": cm.tolist(),
        "num_samples": int(y_true.shape[0]),
    }
