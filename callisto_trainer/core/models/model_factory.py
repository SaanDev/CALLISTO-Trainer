"""Factory for baseline and comparison models, with optional metadata fusion."""

# NOTE: Vendored from H:\Burst Identifier (src/models/model_factory.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import math

import torch
from torch import nn

from callisto_trainer.core.models.metadata_model import MetadataConditionedModel
from callisto_trainer.core.models.simple_cnn import SmallBurstCNN
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _instantiate_tv(model_fn, pretrained: bool):
    """Build a torchvision model, returning ``(model, used_pretrained)``.

    Using the ``'DEFAULT'`` string keeps this robust across torchvision
    versions. If pretrained weights cannot be fetched (offline or blocked
    network), fall back to random init with a warning instead of crashing or
    hanging the run, and report that pretrained was not actually used so the
    stem conv is adapted correctly.
    """
    if not pretrained:
        return model_fn(weights=None), False
    try:
        return model_fn(weights="DEFAULT"), True
    except Exception as exc:  # network/download/version issues
        LOGGER.warning(
            "Could not load pretrained weights (%s). Falling back to random "
            "initialization. Set model.pretrained: false to skip the download.",
            exc,
        )
        return model_fn(weights=None), False


def _replace_first_conv(module: nn.Module, in_channels: int) -> bool:
    """Swap the first Conv2d for one accepting ``in_channels`` (random init)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            replacement = nn.Conv2d(
                in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            setattr(module, name, replacement)
            return True
        if _replace_first_conv(child, in_channels):
            return True
    return False


def _find_first_conv(module: nn.Module):
    """Return ``(parent_module, attr_name, conv)`` for the first Conv2d, or None."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        found = _find_first_conv(child)
        if found is not None:
            return found
    return None


def _adapt_first_conv(module: nn.Module, in_channels: int) -> None:
    """Adapt the first Conv2d to ``in_channels`` while keeping pretrained weights.

    For the common 3->1 case the RGB kernels are summed into a single input
    channel (the standard ImageNet->grayscale adaptation, as used by timm), so
    the pretrained low-level filters are preserved instead of reinitialized.
    """
    found = _find_first_conv(module)
    if found is None:
        return
    parent, name, old = found
    if old.in_channels == in_channels:
        return

    replacement = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        weight = old.weight.detach()
        if in_channels == 1:
            new_weight = weight.sum(dim=1, keepdim=True)
        else:
            repeat = int(math.ceil(in_channels / old.in_channels))
            new_weight = weight.repeat(1, repeat, 1, 1)[:, :in_channels]
            new_weight = new_weight * (old.in_channels / in_channels)
        replacement.weight.copy_(new_weight)
        if old.bias is not None:
            replacement.bias.copy_(old.bias.detach())
    setattr(parent, name, replacement)


def _prepare_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """Adapt the stem conv to ``in_channels`` (preserve weights if pretrained)."""
    if pretrained:
        _adapt_first_conv(model, in_channels)
    else:
        _replace_first_conv(model, in_channels)


def _apply_dropout(model: nn.Module, normalized_name: str, dropout: float) -> None:
    """Insert or retune dropout before the classifier head, in place.

    ``model.dropout`` was configurable but silently ignored for every torchvision
    backbone: the head was replaced with a bare ``nn.Linear``, so the binary and
    burst-type models trained with no dropout at all whatever the config said.

    Each backbone is handled where it already keeps its regularization, and none
    of these edits changes a parameter name -- ``AdaptiveAvgPool2d`` and
    ``Dropout`` both have zero parameters, so ``state_dict`` keys are byte-for-
    byte what they were. Checkpoints trained before this fix still load.
    """
    if dropout <= 0:
        return

    if normalized_name == "resnet18":
        # ResNet has no dropout anywhere; the pooled feature vector feeding fc is
        # the one place it belongs.
        model.avgpool = nn.Sequential(model.avgpool, nn.Dropout(dropout))
        return

    # EfficientNet and MobileNet already ship a Dropout in their classifier;
    # retune it rather than stacking a second one.
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return
    for layer in classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = float(dropout)


def _build_image_classifier(
    normalized_name: str,
    in_channels: int,
    dropout: float,
    pretrained: bool = False,
    num_classes: int = 1,
) -> nn.Module:
    """Single-input image classifier with a final ``Linear -> num_classes`` head.

    ``num_classes == 1`` (default) produces the binary logit head used by the
    burst/no-burst model; ``num_classes > 1`` produces a multiclass head (e.g.
    the 3-way Type II / Type III / Other burst-type model).
    """
    if normalized_name in {"simple_cnn", "small_cnn", "baseline"}:
        return SmallBurstCNN(in_channels=in_channels, dropout=dropout, num_classes=num_classes)

    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for ResNet/EfficientNet/MobileNet models. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if normalized_name == "resnet18":
        model, used_pretrained = _instantiate_tv(tv_models.resnet18, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        _apply_dropout(model, normalized_name, dropout)
        return model

    if normalized_name == "efficientnet_b0":
        model, used_pretrained = _instantiate_tv(tv_models.efficientnet_b0, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        _apply_dropout(model, normalized_name, dropout)
        return model

    if normalized_name == "mobilenet_v3_small":
        model, used_pretrained = _instantiate_tv(tv_models.mobilenet_v3_small, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        _apply_dropout(model, normalized_name, dropout)
        return model

    raise ValueError(f"Unsupported model name: {normalized_name}")


def create_backbone(
    name: str, in_channels: int = 1, dropout: float = 0.25, pretrained: bool = False
) -> tuple[nn.Module, int]:
    """Return an image feature extractor (no classifier) and its feature dim."""
    normalized_name = name.lower().replace("-", "_")

    if normalized_name in {"simple_cnn", "small_cnn", "baseline"}:
        base = SmallBurstCNN(in_channels=in_channels, dropout=dropout)
        backbone = nn.Sequential(base.features, nn.Flatten(1), nn.Dropout(dropout))
        return backbone, 128

    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for ResNet/EfficientNet/MobileNet models. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if normalized_name == "resnet18":
        model, used_pretrained = _instantiate_tv(tv_models.resnet18, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if normalized_name == "efficientnet_b0":
        model, used_pretrained = _instantiate_tv(tv_models.efficientnet_b0, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    if normalized_name == "mobilenet_v3_small":
        model, used_pretrained = _instantiate_tv(tv_models.mobilenet_v3_small, pretrained)
        _prepare_first_conv(model, in_channels, used_pretrained)
        feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported model name: {name}")


def model_kwargs_from_config(model_cfg: dict) -> dict:
    """``create_model`` keyword arguments that rebuild the branches a config names.

    Every loader goes through this, so a checkpoint is always rebuilt with the
    architecture it was trained with -- the physics branch, its feature count and
    the number of image views -- rather than whatever the defaults happen to be.
    A mismatch here does not fail quietly; it fails in ``load_state_dict`` with a
    wall of missing keys, which is what this exists to prevent.
    """
    if bool(model_cfg.get("use_physics", False)):
        from callisto_trainer.core.region_features import (
            FEATURE_SET_PHYSICS_V1,
            feature_count,
        )

        feature_set = str(model_cfg.get("feature_set") or FEATURE_SET_PHYSICS_V1)
        return dict(
            use_physics=True,
            num_physics=feature_count(feature_set),
            num_views=len(model_cfg.get("views") or ["crop"]),
        )
    if bool(model_cfg.get("use_metadata", False)):
        from callisto_trainer.core.metadata_features import NUM_NUMERIC

        vocab = model_cfg.get("station_vocab", {}) or {}
        return dict(
            use_metadata=True,
            num_stations=len(vocab) + 1,
            num_numeric=NUM_NUMERIC,
            station_emb_dim=int(model_cfg.get("station_emb_dim", 8)),
        )
    return {}


def create_model(
    name: str,
    in_channels: int = 1,
    dropout: float = 0.25,
    *,
    pretrained: bool = False,
    num_classes: int = 1,
    use_metadata: bool = False,
    num_stations: int = 1,
    num_numeric: int = 6,
    station_emb_dim: int = 8,
    use_physics: bool = False,
    num_physics: int = 8,
    num_views: int = 1,
) -> nn.Module:
    """Create an image classifier by name.

    ``num_classes == 1`` (default) builds the binary burst/no-burst head;
    ``num_classes > 1`` builds a multiclass head (the burst-type model uses 3).
    When ``use_metadata`` is true, the image backbone is fused with a metadata
    branch (station embedding + numeric frequency/date features). When
    ``pretrained`` is true, torchvision backbones load ImageNet weights and the
    stem conv is adapted to ``in_channels`` (weights preserved, not reset).
    """
    normalized_name = name.lower().replace("-", "_")

    if use_physics:
        # Trainer addition: fuse the image backbone with measured burst physics
        # (drift rate, extent, fit quality). See models/physics_model.py.
        from callisto_trainer.core.models.physics_model import (
            PhysicsConditionedModel,
            RegionContextModel,
        )

        if use_metadata:
            raise ValueError(
                "use_physics and use_metadata cannot be combined: the physics model is "
                "region-based and image-only apart from its physics branch."
            )
        backbone, feature_dim = create_backbone(
            name, in_channels=in_channels, dropout=dropout, pretrained=pretrained
        )
        if int(num_views) > 1:
            # Each view is a single-channel image through the shared backbone.
            return RegionContextModel(
                backbone,
                feature_dim=feature_dim,
                num_views=int(num_views),
                num_features=int(num_physics),
                num_classes=int(num_classes),
                dropout=dropout,
            )
        return PhysicsConditionedModel(
            backbone,
            feature_dim=feature_dim,
            num_physics=int(num_physics),
            num_classes=int(num_classes),
            dropout=dropout,
        )

    if not use_metadata:
        return _build_image_classifier(
            normalized_name, in_channels, dropout, pretrained, num_classes
        )

    if int(num_classes) != 1:
        # The metadata-fusion head is binary-only today; the burst-type model is
        # deliberately image-only, so this combination is never exercised.
        raise ValueError(
            "num_classes > 1 is not supported together with use_metadata=True "
            "(the multiclass burst-type model is image-only)."
        )

    backbone, feature_dim = create_backbone(
        name, in_channels=in_channels, dropout=dropout, pretrained=pretrained
    )
    return MetadataConditionedModel(
        backbone,
        feature_dim=feature_dim,
        num_stations=int(num_stations),
        num_numeric=int(num_numeric),
        station_emb_dim=int(station_emb_dim),
        dropout=dropout,
    )
