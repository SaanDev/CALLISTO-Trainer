"""Small CNN baseline trained from scratch."""

# NOTE: Vendored from H:\Burst Identifier (src/models/simple_cnn.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import torch
from torch import nn


class SmallBurstCNN(nn.Module):
    """Compact classifier for ``1 x 224 x 224`` dynamic spectra.

    With ``num_classes == 1`` (the default) the forward pass returns a squeezed
    ``[B]`` logit tensor for binary ``BCEWithLogitsLoss`` (unchanged behaviour).
    With ``num_classes > 1`` it returns ``[B, num_classes]`` logits for
    multiclass ``CrossEntropyLoss`` (used by the burst-type head).
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.25, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(x))
        if self.num_classes == 1:
            return logits.squeeze(1)
        return logits

