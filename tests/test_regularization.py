"""Guards on the anti-overfitting settings.

Motivated by a real 100-epoch unified run: training loss reached 0.001 while
validation loss climbed from 0.5 to 1.2 and macro-F1 sat flat at ~0.80 from
epoch 55 on. The model was not getting more wrong, it was getting more confident
about what it already got wrong, having memorized a training set whose rare
classes rest on ~105 and ~69 distinct hand-drawn bursts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from callisto_trainer.core.augmentations import SpectrumAugmenter
from callisto_trainer.core.config import load_config
from callisto_trainer.core.train_binary import (
    SmoothedBCEWithLogitsLoss,
    _make_criterion as _binary_make_criterion,
)
from callisto_trainer.core.train_type import _make_criterion as _type_make_criterion


def _augmenter(**overrides) -> SpectrumAugmenter:
    config = load_config()
    config["augmentation"].update({"probability": 1.0, **overrides})
    return SpectrumAugmenter(config)


# -- time shift must translate, not wrap ----------------------------------


def test_time_shift_does_not_wrap_the_burst_around_the_edge() -> None:
    """A wrapped shift pastes a copy of the burst against the opposite edge.

    On a whole-file spectrogram that is harmless. On a tight crop it invents a
    second burst that cannot occur in real data, which is worse than no
    augmentation at all.
    """
    tensor = torch.zeros(1, 224, 224)
    tensor[:, :, 210:] = 1.0  # a bright feature hard against the right edge

    for _ in range(60):
        shifted = _augmenter(
            time_shift_max=24, noise_std=0.0, freq_mask_max=0, time_mask_max=0,
            intensity_scale_min=1.0, intensity_scale_max=1.0,
        )(tensor)
        assert float(shifted[:, :, :100].max()) == 0.0, (
            "signal from the right edge reappeared on the left: the shift wrapped"
        )


def test_time_shift_preserves_the_feature_it_moves() -> None:
    tensor = torch.zeros(1, 224, 224)
    tensor[:, :, 100:120] = 1.0
    augment = _augmenter(
        time_shift_max=24, noise_std=0.0, freq_mask_max=0, time_mask_max=0,
        intensity_scale_min=1.0, intensity_scale_max=1.0,
    )

    widths = {int((augment(tensor)[0, 0] > 0.5).sum()) for _ in range(40)}
    assert widths == {20}, f"the shifted feature changed width: {sorted(widths)}"


# -- output contract -------------------------------------------------------


def test_augmented_tensors_stay_in_the_range_the_model_is_trained_on() -> None:
    """Noise and intensity scaling must not push samples outside [0, 1]."""
    rng = np.random.RandomState(0)
    tensor = torch.from_numpy(rng.uniform(0, 1, (1, 224, 224)).astype(np.float32))
    augment = _augmenter()

    for _ in range(40):
        out = augment(tensor)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_masks_are_applied_once_per_configured_count() -> None:
    """Two narrow masks beat one wide one: the burst is less likely to vanish."""
    tensor = torch.ones(1, 224, 224)
    augment = _augmenter(
        time_shift_max=0, noise_std=0.0, intensity_scale_min=1.0, intensity_scale_max=1.0,
        freq_mask_max=24, time_mask_max=0, num_freq_masks=2,
    )
    # Over many draws, two masks of <=24 rows must be able to blank more rows
    # than a single one ever could.
    widest = max(int((augment(tensor)[0, :, 0] < 0.5).sum()) for _ in range(200))
    assert 24 < widest <= 48, f"expected up to two masks, blanked {widest} rows"


def test_augmentation_is_off_when_disabled() -> None:
    tensor = torch.rand(1, 224, 224)
    config = load_config()
    config["augmentation"]["enabled"] = False
    assert torch.equal(SpectrumAugmenter(config)(tensor), tensor)


# -- label smoothing -------------------------------------------------------


def test_shipped_default_enables_label_smoothing() -> None:
    assert float(load_config()["training"]["label_smoothing"]) > 0


def test_multiclass_criterion_carries_the_configured_smoothing() -> None:
    config = load_config()
    config["training"]["class_balance"] = {"strategy": "none"}
    config["training"]["label_smoothing"] = 0.05

    criterion = _type_make_criterion(config, None, 4, torch.device("cpu"))

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == pytest.approx(0.05)


def test_smoothing_penalises_runaway_confidence() -> None:
    """The whole point: a hugely overconfident correct logit stops being free."""
    targets = torch.tensor([1.0, 0.0])
    confident = torch.tensor([20.0, -20.0])   # what epoch 100 of the real run looked like
    moderate = torch.tensor([3.0, -3.0])

    smoothed = SmoothedBCEWithLogitsLoss(0.05)
    plain = nn.BCEWithLogitsLoss()

    assert float(plain(confident, targets)) < float(plain(moderate, targets)), (
        "plain BCE rewards ever-larger logits, which is the failure mode"
    )
    assert float(smoothed(confident, targets)) > float(smoothed(moderate, targets)), (
        "smoothing must make runaway confidence cost more, not less"
    )


def test_binary_criterion_is_unwrapped_when_smoothing_is_off() -> None:
    config = load_config()
    config["training"]["class_balance"] = {"strategy": "none"}
    config["training"]["label_smoothing"] = 0.0

    criterion = _binary_make_criterion(config, None, torch.device("cpu"))

    assert isinstance(criterion, nn.BCEWithLogitsLoss)


def test_smoothing_rejects_an_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        SmoothedBCEWithLogitsLoss(1.0)


# -- class weighting -------------------------------------------------------


class _FakeLoader:
    """Minimal stand-in exposing the manifest rows the weighting reads."""

    def __init__(self, counts: dict[int, int]) -> None:
        self.dataset = type(
            "_DS", (), {"rows": [{"label_id": str(k)} for k, n in counts.items() for _ in range(n)]}
        )()


# The real unified split: No_Burst / Type III / Type II / Other.
REAL_COUNTS = {0: 2493, 1: 928, 2: 253, 3: 131}


def _weights(exponent: float) -> list[float]:
    config = load_config()
    config["training"]["class_balance"] = {
        "strategy": "auto_class_weights", "exponent": exponent
    }
    criterion = _type_make_criterion(
        config, _FakeLoader(REAL_COUNTS), 4, torch.device("cpu")
    )
    return criterion.weight.tolist()


def test_damped_weighting_cuts_the_spread_that_drives_memorization() -> None:
    """Inverse frequency puts 19x on a class backed by 69 distinct bursts."""
    plain = _weights(1.0)
    damped = _weights(0.5)

    assert max(plain) / min(plain) == pytest.approx(19.03, abs=0.1)
    assert max(damped) / min(damped) == pytest.approx(4.36, abs=0.1)


def test_weighting_still_favours_the_rare_classes() -> None:
    """Damping must not invert the correction it is damping."""
    damped = _weights(0.5)
    assert damped == sorted(damped), "weights must rise as classes get rarer"
    assert damped[3] > damped[0], "the rarest class must still outweigh the commonest"
    assert sum(damped) / len(damped) == pytest.approx(1.0), "weights are rescaled to mean 1"


def test_exponent_zero_is_uniform_weighting() -> None:
    assert _weights(0.0) == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_negative_exponent_is_rejected() -> None:
    with pytest.raises(ValueError):
        _weights(-1.0)


# -- dropout actually reaches the torchvision backbones --------------------


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "mobilenet_v3_small"])
def test_configured_dropout_is_applied_to_torchvision_backbones(name: str) -> None:
    """It used to be silently discarded: the head became a bare nn.Linear."""
    from callisto_trainer.core.models.model_factory import create_model

    model = create_model(name, in_channels=1, dropout=0.25, pretrained=False, num_classes=4)
    active = [m for m in model.modules() if isinstance(m, nn.Dropout) and m.p > 0]

    assert active, f"{name} trained with no dropout despite model.dropout being set"
    assert all(m.p == pytest.approx(0.25) for m in active)


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "mobilenet_v3_small"])
def test_dropout_does_not_change_checkpoint_keys(name: str) -> None:
    """Dropout must stay loadable into models trained before it was fixed.

    Dropout and AdaptiveAvgPool2d carry no parameters, so inserting them cannot
    rename anything -- which is what lets an already-exported bundle keep working.
    """
    from callisto_trainer.core.models.model_factory import create_model

    without = create_model(name, in_channels=1, dropout=0.0, pretrained=False, num_classes=4)
    with_dropout = create_model(name, in_channels=1, dropout=0.25, pretrained=False, num_classes=4)

    assert list(without.state_dict()) == list(with_dropout.state_dict())
    with_dropout.load_state_dict(without.state_dict(), strict=True)


def test_dropout_is_inactive_at_inference() -> None:
    from callisto_trainer.core.models.model_factory import create_model

    model = create_model("resnet18", in_channels=1, dropout=0.25, pretrained=False, num_classes=4)
    batch = torch.randn(4, 1, 224, 224)

    model.train()
    assert not torch.allclose(model(batch), model(batch)), "dropout is not active in training"
    model.eval()
    assert torch.allclose(model(batch), model(batch)), "dropout must be off at inference"
