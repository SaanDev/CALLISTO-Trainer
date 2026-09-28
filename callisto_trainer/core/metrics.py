"""Binary classification metrics for burst detection."""

# NOTE: Vendored from H:\Burst Identifier (src/evaluation/metrics.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(metric_fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        value = float(metric_fn(y_true, y_prob))
    except ValueError:
        return math.nan
    return value


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute thresholded and ranking metrics for binary classification."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else math.nan
    fnr = fn / (fn + tp) if (fn + tp) else math.nan

    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else math.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_prob),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        # Trainer additions. On a heavily imbalanced set, accuracy and F1 are both
        # dominated by the majority class -- a model that always says "Burst"
        # scores well on a 86%-Burst set. These two weight both classes equally,
        # so they collapse to ~0.5 for a constant predictor and expose it.
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2.0)
        if not math.isnan(specificity)
        else math.nan,
    }


def majority_baseline_metrics(y_true: np.ndarray) -> dict[str, Any]:
    """How a constant "always predict the majority class" model would score.

    The number every headline metric must beat before a model has demonstrated
    it learned anything. Reported alongside real metrics because F1 = 0.91 on an
    86%-positive set sounds excellent and can still be worse than a coin that is
    glued to one side.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_true.size == 0:
        return {"f1": math.nan, "accuracy": math.nan, "balanced_accuracy": math.nan}

    positives = int(y_true.sum())
    total = int(y_true.size)
    majority = 1 if positives * 2 >= total else 0
    constant = np.full_like(y_true, majority)
    return {
        "majority_class": "Burst" if majority == 1 else "No_Burst",
        "positive_fraction": float(positives / total),
        "accuracy": float(accuracy_score(y_true, constant)),
        "f1": float(f1_score(y_true, constant, zero_division=0)),
        # A constant predictor gets exactly 0.5 here, by construction.
        "balanced_accuracy": 0.5,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 181,
) -> tuple[float, dict[str, Any]]:
    """Find the best validation threshold for a selected metric.

    Thresholds are searched between 0.05 and 0.95 to avoid unstable edge
    choices that classify almost everything into one class.

    Trainer addition: ``balanced_accuracy`` is accepted as a metric. On an
    imbalanced validation set, maximising ``f1`` degenerates -- with 85%
    positives the optimum slides to the 0.05 floor, because calling nearly
    everything positive maximises F1. ``balanced_accuracy`` weights both classes
    equally and does not have that failure mode, which is why it is now the
    default for exported binary configs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    metric = metric.lower()
    if metric not in {"f1", "accuracy", "precision", "recall", "balanced_accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_prob, threshold=best_threshold)
    best_score = float(best_metrics[metric])

    for threshold in _candidate_thresholds(y_true, y_prob, num_thresholds):
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics[metric])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _candidate_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, num_thresholds: int
) -> np.ndarray:
    """Thresholds worth evaluating, derived from the scores themselves.

    The original implementation swept a fixed ``linspace(0.05, 0.95)``. That floor
    silently caps the search: a well-separated but poorly calibrated model can put
    every negative at exactly 0.0 and still score real bursts at 0.0004, so the
    entire useful range lies *below* 0.05 and every metric returns the same
    clipped answer. Observed on a real run -- the tuned threshold came back as
    0.05 for f1, accuracy, recall and balanced accuracy alike, and the model then
    missed 7 of 43 test bursts that a threshold of 0.001 would have caught.

    Predictions only change as the threshold crosses an observed probability, so
    the midpoints between consecutive distinct scores are the complete set of
    meaningful candidates. The legacy grid is still included, so any threshold the
    old code could have chosen remains reachable.

    Candidates that would put every sample in one class are dropped, which is what
    the 0.05/0.95 bounds were really guarding against -- but by checking the
    outcome rather than assuming a numeric range.
    """
    grid = np.linspace(0.05, 0.95, max(2, num_thresholds))
    unique = np.unique(y_prob[np.isfinite(y_prob)])
    midpoints = (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else np.empty(0)

    candidates = np.unique(np.concatenate([grid, midpoints, [0.5]]))
    # Keep the search bounded on very large validation sets.
    if candidates.size > 4096:
        step = int(np.ceil(candidates.size / 4096))
        candidates = candidates[::step]

    # Only meaningful when both classes are present to separate.
    if np.unique(y_true).size < 2:
        return candidates

    predicted_positive = (y_prob[None, :] >= candidates[:, None]).sum(axis=1)
    usable = (predicted_positive > 0) & (predicted_positive < y_prob.size)
    return candidates[usable] if usable.any() else candidates


# The legacy fixed grid, retained inside the candidate set for compatibility.
THRESHOLD_SEARCH_BOUNDS = (0.05, 0.95)
# Outside this range a decision threshold says more about calibration than about
# any real operating point.
EXTREME_THRESHOLD_BOUNDS = (0.01, 0.99)


def threshold_is_degenerate(threshold: float, y_prob: np.ndarray | None = None) -> bool:
    """True when this threshold puts every sample in one class.

    Checked against the actual scores when they are available, which is the real
    condition the old fixed 0.05/0.95 search bounds were approximating.
    """
    if y_prob is None:
        return False
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.size == 0:
        return False
    positives = int((y_prob >= threshold).sum())
    return positives == 0 or positives == y_prob.size


def threshold_is_extreme(threshold: float) -> bool:
    """True for a threshold so close to 0 or 1 that calibration is suspect.

    Not an error: a well-separated model can legitimately need 0.0001 because it
    pushes every negative to exactly 0. But it means the probabilities cannot be
    read as confidences, and a small shift in the data could move a lot of
    predictions, so it is worth saying out loud.
    """
    low, high = EXTREME_THRESHOLD_BOUNDS
    return not (low <= float(threshold) <= high)
