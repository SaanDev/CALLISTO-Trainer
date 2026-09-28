"""How often each burst type really occurs, and how the type decision uses it.

## The problem

Among the bursts CALLISTO records, Type III is by far the commonest (about 86%),
then Type IV (about 10.5%), Type II (2-3%) and everything else (about 1%). A
labelled dataset is nothing like that: the rare types are the interesting ones,
so they get labelled out of proportion (the previous archive was 18% Type II),
and the loss is then weighted to balance the types further, so that the rare
ones are learned at all. Both are right for *learning*. But a model trained that
way has learned to expect a Type II almost as often as a Type III, and when a
region is ambiguous it calls the rare type far more often than the sky produces
it. With Type III outnumbering Type II thirty to one, even a small share of
Type III called Type II swamps the real Type II detections.

## What is done about it

Training is left balanced, so the rare types are still learned. Afterwards the
type probabilities of a region are corrected by the ratio of how common each
type really is to how common it was in training (the standard prior-shift
correction):

    p'(type) ~ p(type) * (observed share / training share) ** strength

and renormalised over the burst types only, so the total burst probability --
and with it the calibrated burst/no-burst decision -- is untouched. Only *which
type* a burst is called changes. The training share is the effective one: each
class's training samples times its loss weight, which is what the model's
probabilities were fitted to. Regions are counted as events, which is an
approximation (a long Type II yields more regions than a single Type III).

``strength`` runs from 0 (the balanced model as trained) to 1 (fully real-world
odds). It is chosen after training on the validation regions by the
**real-world macro-F1**: per-type F1 with precision computed as it would be if
the types occurred at the observed frequencies. That rewards calling the common
type when unsure without letting a rare type vanish, which plain accuracy under
the real frequencies would allow (always answering Type III is 86% accurate).
The curve is flat near its top, so the middle of the strengths within one
bootstrap standard deviation of the best is taken (see :func:`choose_strength`).

Type IIIG is a kind of Type III, so the Type III share is split between the two
by how many boxes of each were drawn. A type folded into another class (Type IV
into Other when too few were drawn) adds its share to that class.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from callisto_trainer.core.taxonomy import (
    NON_BURST_LABELS,
    OTHER,
    PARENT_LABEL,
    TYPE_II,
    TYPE_III,
    TYPE_IIIG,
    TYPE_IV,
)

# Share of each burst type among the bursts CALLISTO observes (Type IIIG counted
# within Type III). Editable in the Dataset tab and in the training config.
OBSERVED_TYPE_SHARES: dict[str, float] = {
    TYPE_III: 0.86,
    TYPE_IV: 0.105,
    TYPE_II: 0.025,
    OTHER: 0.01,
}
# Types that can be given a share, in display order.
SHARE_TYPES: tuple[str, ...] = (TYPE_II, TYPE_III, TYPE_IV, OTHER)

# Strengths tried when choosing one on the validation regions, and how many
# bootstrap resamples measure the noise of that choice.
STRENGTH_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_RESAMPLES = 200
# A class never gets less than this share, so a zero typed in the settings
# makes the type rare rather than impossible.
MIN_SHARE = 1e-3


def normalise_shares(shares: Mapping[str, float] | None) -> dict[str, float]:
    """``shares`` (percent or fractions) over :data:`SHARE_TYPES`, summing to 1."""
    raw = dict(OBSERVED_TYPE_SHARES if not shares else shares)
    values = {name: max(0.0, float(raw.get(name, 0.0))) for name in SHARE_TYPES}
    total = sum(values.values())
    if total <= 0:
        return dict(OBSERVED_TYPE_SHARES)
    return {name: value / total for name, value in values.items()}


def _class_for(label: str, class_names: Sequence[str]) -> str | None:
    """The model class a label trains as, following fallbacks; None if absent."""
    seen = set()
    while label not in class_names:
        if label in seen or label not in PARENT_LABEL:
            return None
        seen.add(label)
        label = PARENT_LABEL[label]
    return label


def class_shares(
    shares: Mapping[str, float] | None,
    class_names: Sequence[str],
    box_counts: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """The observed share of every burst class of a model, summing to 1.

    ``box_counts`` (drawn boxes per class) split the Type III share between
    Type III and Type IIIG; without them the split is even.
    """
    shares = normalise_shares(shares)
    burst_classes = [name for name in class_names if name not in NON_BURST_LABELS]
    counts = dict(box_counts or {})
    result = {name: 0.0 for name in burst_classes}
    for label, share in shares.items():
        if label == TYPE_III and TYPE_IIIG in burst_classes and TYPE_III in burst_classes:
            single, group = float(counts.get(TYPE_III, 0)), float(counts.get(TYPE_IIIG, 0))
            part = group / (single + group) if single + group > 0 else 0.5
            result[TYPE_III] += share * (1.0 - part)
            result[TYPE_IIIG] += share * part
            continue
        target = _class_for(label, burst_classes)
        if target is not None:
            result[target] += share
    result = {name: max(value, MIN_SHARE) for name, value in result.items()}
    total = sum(result.values())
    return {name: value / total for name, value in result.items()} if total else result


def training_shares(
    sample_counts: Mapping[str, int], class_weights: Mapping[str, float], class_names: Sequence[str]
) -> dict[str, float]:
    """Each burst class's share of the *weighted* training set: count x loss weight."""
    burst_classes = [name for name in class_names if name not in NON_BURST_LABELS]
    effective = {
        name: float(sample_counts.get(name, 0)) * float(class_weights.get(name, 1.0))
        for name in burst_classes
    }
    total = sum(effective.values())
    if total <= 0:
        return {name: 1.0 / len(burst_classes) for name in burst_classes} if burst_classes else {}
    return {name: max(value / total, MIN_SHARE) for name, value in effective.items()}


def log_adjustment(observed: Mapping[str, float], trained: Mapping[str, float]) -> dict[str, float]:
    """``log(observed / trained)`` per burst class, centred on zero."""
    names = [name for name in observed if name in trained]
    if not names:
        return {}
    values = {name: math.log(observed[name] / trained[name]) for name in names}
    mean = sum(values.values()) / len(values)
    return {name: value - mean for name, value in values.items()}


def adjust_probabilities(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    adjustment: Mapping[str, float] | None,
    strength: float,
) -> np.ndarray:
    """Shift the burst-type probabilities toward the observed frequencies.

    ``probabilities`` is ``[N, K]`` over ``class_names``. Non-burst classes are
    unchanged and the burst classes keep their total, so burst evidence -- and
    every threshold calibrated on it -- is exactly what it was.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not adjustment or strength <= 0 or probabilities.size == 0:
        return probabilities
    burst = [i for i, name in enumerate(class_names) if name in adjustment]
    if not burst:
        return probabilities
    factors = np.exp(float(strength) * np.array([adjustment[class_names[i]] for i in burst]))
    part = probabilities[:, burst]
    mass = part.sum(axis=1, keepdims=True)
    shifted = part * factors[None, :]
    shifted_mass = shifted.sum(axis=1, keepdims=True)
    scale = np.divide(mass, shifted_mass, out=np.zeros_like(mass), where=shifted_mass > 0)
    adjusted = probabilities.copy()
    adjusted[:, burst] = shifted * scale
    return adjusted


def real_world_type_report(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    observed: Mapping[str, float],
    adjustment: Mapping[str, float] | None = None,
    strength: float = 0.0,
) -> dict[str, Any]:
    """Typing quality on real bursts, as it would be at the observed frequencies.

    Only regions whose true class is a burst type are scored, and the type is
    the argmax over the burst classes (typing is only asked of a region already
    called a burst). Recall per type does not depend on how common the types
    are; precision does, so it is computed from the per-type confusion rates
    weighted by ``observed`` rather than by how many of each were labelled.
    """
    y_true = np.asarray(y_true, dtype=int)
    burst = [i for i, name in enumerate(class_names) if name in observed]
    rows = np.isin(y_true, burst)
    if not burst or not rows.any():
        return {}
    adjusted = adjust_probabilities(probabilities, class_names, adjustment, strength)
    predicted = np.asarray(burst)[adjusted[rows][:, burst].argmax(axis=1)]
    truth = y_true[rows]

    present = [i for i in burst if (truth == i).any()]
    # rate[i][j] = P(called j | truly i), per type actually present.
    rate = {
        i: {j: float(np.mean(predicted[truth == i] == j)) for j in burst} for i in present
    }
    weight = {i: float(observed[class_names[i]]) for i in present}
    total = sum(weight.values()) or 1.0
    weight = {i: value / total for i, value in weight.items()}

    per_type: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for i in present:
        recall = rate[i][i]
        called = sum(weight[k] * rate[k][i] for k in present)
        precision = weight[i] * recall / called if called > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_type[class_names[i]] = {
            "recall": recall, "real_world_precision": precision, "real_world_f1": f1,
            "samples": int((truth == i).sum()),
        }
        f1s.append(f1)
    return {
        "strength": float(strength),
        "real_world_accuracy": float(sum(weight[i] * rate[i][i] for i in present)),
        "real_world_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "balanced_recall": float(np.mean([rate[i][i] for i in present])),
        "per_type": per_type,
    }


def _macro_f1(report: dict[str, Any]) -> float:
    value = report.get("real_world_macro_f1", float("nan")) if report else float("nan")
    return float(value) if math.isfinite(value) else float("nan")


def choose_strength(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    observed: Mapping[str, float],
    adjustment: Mapping[str, float],
    grid: Sequence[float] = STRENGTH_GRID,
    groups: Sequence[Any] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """The middle of the strengths that score as well as the best, and each report.

    The real-world macro-F1 curve over strength is flat near its top: on the
    archive's validation regions it read 0.756, 0.761, 0.766, 0.768 at 0.25, 0.5,
    0.75, 1.0, differences far inside the noise of a few dozen rare-type files.
    Its arg-max is then a coin toss, and so is its gentlest point. So, as the
    burst threshold is (``file_eval.choose_threshold``), the choice is the middle
    of the plateau: the paired shortfall of every strength against the best is
    bootstrapped over ``groups`` (the source file of each region, since regions
    of one file are not independent); the strengths whose mean shortfall is
    within one standard deviation form the plateau; and the grid value nearest
    its middle is taken, the gentler on a tie, because every step of strength
    costs rare-type recall. Without ``groups`` each region is its own group.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    reports = [
        real_world_type_report(y_true, probabilities, class_names, observed, adjustment, value)
        for value in grid
    ]
    scores = [_macro_f1(report) for report in reports]
    finite = [i for i, score in enumerate(scores) if math.isfinite(score)]
    if not finite:
        return 0.0, reports
    top = max(scores[i] for i in finite)
    best = max(i for i in finite if scores[i] >= top - 1e-12)

    keys = np.asarray(groups if groups is not None else np.arange(y_true.size), dtype=object)
    unique = list(dict.fromkeys(keys.tolist()))
    members = {key: np.flatnonzero(keys == key) for key in unique}
    rng = np.random.default_rng(seed)
    shortfall = {i: [] for i in finite}
    for _ in range(int(resamples)):
        drawn = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([members[unique[k]] for k in drawn])
        boot = [
            _macro_f1(real_world_type_report(
                y_true[rows], probabilities[rows], class_names, observed, adjustment, grid[i]
            )) if i in shortfall else float("nan")
            for i in range(len(grid))
        ]
        if not math.isfinite(boot[best]):
            continue
        for i in finite:
            if math.isfinite(boot[i]):
                shortfall[i].append(boot[best] - boot[i])

    for report, i in zip(reports, range(len(grid))):
        if report and shortfall.get(i):
            report["shortfall_to_best"] = float(np.mean(shortfall[i]))
            report["shortfall_sd"] = float(np.std(shortfall[i]))
    plateau = [
        float(grid[i]) for i in finite
        if i == best or (shortfall[i] and np.mean(shortfall[i]) <= np.std(shortfall[i]))
    ]
    return plateau_middle(plateau), reports


def plateau_middle(strengths: Sequence[float]) -> float:
    """The strength nearest the middle of ``strengths``, the gentler on a tie."""
    values = sorted(float(value) for value in strengths)
    middle = 0.5 * (values[0] + values[-1])
    return min(values, key=lambda value: (abs(value - middle), value))
