"""Image backbone fused with measured burst physics.

The spectrogram crop tells the network what a burst *looks* like. The physics
branch tells it how fast the emission drifts in frequency, which is the quantity
that physically separates the burst types: a Type III is an electron beam at
0.1-0.5c drifting tens of MHz per second, a Type II is a shock front two orders
of magnitude slower. Morphology alone has to infer that from a 224x224 image
whose axes have been resampled; the measurement states it directly.

Two design points worth keeping:

* The physics vector carries an explicit ``measured`` flag, and unmeasurable
  regions arrive as all-zeros. Without the flag the network could not tell "drift
  of zero" from "no drift could be measured", and roughly a seventh of regions
  are genuinely unmeasurable.
* The image path is never gated on the physics branch. If a measurement is
  missing or wrong the model degrades to the image-only case rather than
  failing, which matters because the measurement is a heuristic over real,
  noisy data.
"""

from __future__ import annotations

import torch
from torch import nn


class PhysicsConditionedModel(nn.Module):
    """Multiclass classifier fusing an image backbone with physics features."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_physics: int,
        num_classes: int,
        physics_hidden: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.num_physics = int(num_physics)
        self.num_classes = int(num_classes)

        self.physics_mlp = nn.Sequential(
            nn.Linear(self.num_physics, physics_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(physics_hidden, physics_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim + physics_hidden, self.num_classes),
        )

    def forward(self, image: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.dim() > 2:
            features = torch.flatten(features, 1)

        physics_features = self.physics_mlp(physics)
        return self.classifier(torch.cat([features, physics_features], dim=1))


class RegionContextModel(nn.Module):
    """One backbone over several views of a region, fused with region features.

    The views are the exact crop and a wide context strip (see core/crops.py).
    They are *not* pixel-aligned -- the context covers the full band and several
    times the duration -- so stacking them as input channels would ask the first
    convolution to combine unrelated pixels. Instead the same backbone embeds
    each view on its own and the embeddings are concatenated. Sharing the weights
    keeps the parameter count of a single view and keeps ImageNet-pretrained
    filters meaningful for both.

    The input is still one ``[B, V, H, W]`` tensor, so datasets, augmentation and
    TorchScript see a single image argument whatever the number of views.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_views: int,
        num_features: int,
        num_classes: int,
        hidden: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.num_views = int(num_views)
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)

        self.feature_mlp = nn.Sequential(
            nn.Linear(self.num_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * self.num_views + hidden, self.num_classes),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        views = image.reshape(batch * self.num_views, 1, image.shape[-2], image.shape[-1])
        embedded = self.backbone(views)
        if embedded.dim() > 2:
            embedded = torch.flatten(embedded, 1)
        embedded = embedded.reshape(batch, self.num_views * self.feature_dim)
        return self.classifier(torch.cat([embedded, self.feature_mlp(features)], dim=1))
