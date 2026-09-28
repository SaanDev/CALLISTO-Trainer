"""Guards against imbalanced data producing flattering but useless metrics.

Motivated by a real run: a 281-Burst / 46-No_Burst dataset trained to F1 0.911
on test, which *looks* strong but is below the 0.925 an always-say-Burst constant
scores — and it missed 7 of 43 real bursts. Nothing reported at the time made
that visible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from callisto_trainer.core.metrics import (  # noqa: E402
    THRESHOLD_SEARCH_BOUNDS,
    compute_binary_metrics,
    find_best_threshold,
    majority_baseline_metrics,
    threshold_is_degenerate,
    threshold_is_extreme,
)
from callisto_trainer.store.export import ExportResult  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _imbalanced(n_pos: int = 281, n_neg: int = 46, seed: int = 0):
    """Scores resembling the real run: mostly positives, a conservative model."""
    rng = np.random.RandomState(seed)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_prob = np.concatenate(
        [rng.beta(2.0, 1.2, n_pos), rng.beta(1.0, 8.0, n_neg)]
    )
    return y_true, y_prob


# -- balanced metrics ------------------------------------------------------


def test_balanced_accuracy_exposes_a_constant_predictor() -> None:
    y_true = np.array([1] * 86 + [0] * 14)
    always_burst = np.ones(100)

    metrics = compute_binary_metrics(y_true, always_burst, threshold=0.5)

    assert metrics["f1"] > 0.9, "F1 flatters a constant predictor on this split"
    assert metrics["accuracy"] > 0.85
    assert metrics["balanced_accuracy"] == pytest.approx(0.5), (
        "balanced accuracy must collapse to 0.5 for a constant predictor"
    )
    assert metrics["specificity"] == pytest.approx(0.0)


def test_specificity_and_recall_are_reported_separately() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.8])

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.625)


def test_majority_baseline_reports_what_must_be_beaten() -> None:
    y_true = np.array([1] * 43 + [0] * 7)
    baseline = majority_baseline_metrics(y_true)

    assert baseline["majority_class"] == "Burst"
    assert baseline["positive_fraction"] == pytest.approx(0.86)
    assert baseline["f1"] == pytest.approx(0.925, abs=0.005)
    assert baseline["balanced_accuracy"] == 0.5


def test_majority_baseline_handles_a_negative_majority() -> None:
    baseline = majority_baseline_metrics(np.array([0] * 90 + [1] * 10))
    assert baseline["majority_class"] == "No_Burst"
    assert baseline["f1"] == 0.0


def test_majority_baseline_on_empty_input() -> None:
    assert np.isnan(majority_baseline_metrics(np.array([]))["f1"])


# -- threshold tuning ------------------------------------------------------


def _uncalibrated_but_separable():
    """The real failure mode: negatives pinned at 0, bursts down to 1e-4.

    Reproduces the observed model. Every useful threshold lies below the old
    0.05 search floor, so the fixed grid could not reach the optimum and every
    metric returned the same clipped answer.
    """
    y_true = np.array([1] * 40 + [0] * 7)
    burst_scores = np.concatenate(
        [np.array([0.00015, 0.00437, 0.01687, 0.02435, 0.03743, 0.03964]), np.full(34, 0.99)]
    )
    y_prob = np.concatenate([burst_scores, np.zeros(7)])
    return y_true, y_prob


def test_search_reaches_below_the_legacy_floor() -> None:
    """The fix: the optimum here is at ~1e-4 and must actually be found."""
    y_true, y_prob = _uncalibrated_but_separable()
    threshold, metrics = find_best_threshold(y_true, y_prob, metric="f1")

    assert threshold < THRESHOLD_SEARCH_BOUNDS[0], (
        f"tuning stopped at the old floor instead of finding the optimum ({threshold})"
    )
    assert metrics["fn"] == 0, "every real burst should be recoverable here"
    assert metrics["fp"] == 0


def test_old_fixed_grid_would_have_missed_those_bursts() -> None:
    """Pins why the fix matters, by scoring the threshold the old code produced."""
    y_true, y_prob = _uncalibrated_but_separable()
    old = compute_binary_metrics(y_true, y_prob, threshold=THRESHOLD_SEARCH_BOUNDS[0])
    new_threshold, new = find_best_threshold(y_true, y_prob, metric="f1")

    assert old["fn"] == 6, "the 0.05 floor leaves the low-scoring bursts behind"
    assert new["fn"] < old["fn"]
    assert new["f1"] > old["f1"]


def test_all_metrics_agree_when_the_classes_are_separable() -> None:
    y_true, y_prob = _uncalibrated_but_separable()
    thresholds = {
        metric: find_best_threshold(y_true, y_prob, metric=metric)[0]
        for metric in ("f1", "balanced_accuracy", "accuracy")
    }
    assert len(set(round(t, 9) for t in thresholds.values())) == 1, thresholds


def test_candidate_search_never_collapses_to_one_class() -> None:
    y_true, y_prob = _imbalanced()
    threshold, _ = find_best_threshold(y_true, y_prob, metric="f1")
    assert not threshold_is_degenerate(threshold, y_prob)


def test_degenerate_detection_uses_the_actual_scores() -> None:
    y_prob = np.array([0.1, 0.4, 0.9])
    assert threshold_is_degenerate(0.99, y_prob), "nothing predicted positive"
    assert threshold_is_degenerate(0.05, y_prob), "everything predicted positive"
    assert not threshold_is_degenerate(0.5, y_prob)
    assert not threshold_is_degenerate(0.5), "no scores given, nothing to judge"


def test_extreme_threshold_detection() -> None:
    assert threshold_is_extreme(0.000077)
    assert threshold_is_extreme(0.995)
    assert not threshold_is_extreme(0.5)
    assert not threshold_is_extreme(0.05)


def test_tuning_still_works_on_a_single_class_split() -> None:
    y_true = np.ones(10, dtype=int)
    threshold, _ = find_best_threshold(y_true, np.linspace(0.1, 0.9, 10), metric="f1")
    assert 0.0 <= threshold <= 1.0


def test_unknown_threshold_metric_still_rejected() -> None:
    y_true, y_prob = _imbalanced(20, 20)
    with pytest.raises(ValueError):
        find_best_threshold(y_true, y_prob, metric="nonsense")


# -- export warnings -------------------------------------------------------


def _result(counts: dict[str, int]) -> ExportResult:
    result = ExportResult("binary", Path("/tmp/x"), Path("/tmp/x/manifest.csv"))
    result.class_counts = counts
    result.written = sum(counts.values())
    return result


def test_export_warns_about_the_real_imbalance() -> None:
    warnings = _result({"Burst": 281, "No_Burst": 46}).balance_warnings()

    assert warnings, "a 6:1 split must be flagged"
    assert "6.1:1" in warnings[0]
    assert "No_Burst" in warnings[0]
    assert any("very few" in warning for warning in warnings)


def test_export_is_quiet_on_a_balanced_set() -> None:
    assert _result({"Burst": 120, "No_Burst": 100}).balance_warnings() == []


def test_export_warns_on_a_tiny_minority_even_when_ratio_is_ok() -> None:
    warnings = _result({"Burst": 60, "No_Burst": 30}).balance_warnings()
    assert any("30" in warning and "very few" in warning for warning in warnings)


def test_balance_warnings_need_two_classes() -> None:
    assert _result({"Burst": 100}).balance_warnings() == []
    assert _result({}).balance_warnings() == []


def test_exported_binary_config_tunes_its_threshold(tmp_path: Path) -> None:
    import yaml

    from callisto_trainer.core.config import load_config
    from callisto_trainer.store.export import write_training_config

    result = ExportResult("binary", tmp_path, tmp_path / "manifest.csv")
    write_training_config(
        result, load_config(), tmp_path / "outputs", {"No_Burst": 0, "Burst": 1}, task="binary"
    )
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

    assert config["training"]["auto_threshold"] is True
    assert config["training"]["threshold_metric"] in ("f1", "balanced_accuracy")


# -- evaluate tab reporting ------------------------------------------------


@pytest.fixture
def evaluate_tab(qapp, tmp_path: Path):
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.evaluate_tab import EvaluateTab

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    settings.ensure_directories()
    tab = EvaluateTab(settings)
    yield tab
    tab.shutdown()


def test_evaluate_flags_a_model_below_the_baseline(evaluate_tab) -> None:
    """The exact numbers from the real run must be called out, not celebrated."""
    metrics = {
        "f1": 0.911, "precision": 1.0, "recall": 0.837, "roc_auc": 0.99, "pr_auc": 0.997,
        "tp": 36, "fn": 7, "fp": 0, "tn": 7, "threshold": 0.05,
        "balanced_accuracy": 0.918,
    }
    evaluate_tab._show_binary_metrics(metrics)

    headline = evaluate_tab.headline.text()
    assert "Missed <b>7</b> of 43 real bursts" in headline
    assert "16%" in headline
    assert "baseline" in headline.lower()
    assert "at or below" in headline

    warning = evaluate_tab.metric_warning.text()
    assert "does not beat a constant predictor" in warning


def test_evaluate_flags_an_uncalibrated_threshold(evaluate_tab) -> None:
    metrics = {
        "f1": 0.9767, "precision": 0.98, "recall": 0.977, "roc_auc": 0.99, "pr_auc": 0.99,
        "tp": 42, "fn": 1, "fp": 1, "tn": 6, "threshold": 0.000077,
        "balanced_accuracy": 0.917,
    }
    evaluate_tab._show_binary_metrics(metrics)
    assert "not calibrated" in evaluate_tab.metric_warning.text()


def test_evaluate_is_positive_about_a_genuinely_good_model(evaluate_tab) -> None:
    metrics = {
        "f1": 0.94, "precision": 0.95, "recall": 0.93, "roc_auc": 0.98, "pr_auc": 0.97,
        "tp": 93, "fn": 7, "fp": 5, "tn": 95, "threshold": 0.42,
        "balanced_accuracy": 0.94,
    }
    evaluate_tab._show_binary_metrics(metrics)

    assert "above" in evaluate_tab.headline.text()
    assert evaluate_tab.metric_warning.text() == ""


def test_evaluate_derives_balanced_accuracy_for_older_reports(evaluate_tab) -> None:
    """Reports written before balanced_accuracy existed must still be judged."""
    metrics = {
        "f1": 0.911, "precision": 1.0, "recall": 0.837, "roc_auc": 0.99, "pr_auc": 0.997,
        "tp": 36, "fn": 7, "fp": 0, "tn": 7, "threshold": 0.05,
    }
    evaluate_tab._show_binary_metrics(metrics)
    assert "Balanced acc" in evaluate_tab.headline.text()
