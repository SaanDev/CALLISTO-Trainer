"""Multi-input classifier: image backbone fused with tabular metadata.

The image backbone produces a feature vector; a metadata branch embeds the
station id and passes the numeric features (frequency range, cyclical date)
through a small MLP. The two are concatenated before the final classifier, so
the model can condition its burst decision on station/frequency/date context.
"""

# NOTE: Vendored from H:\Burst Identifier (src/models/metadata_model.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import torch
from torch import nn


class MetadataConditionedModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_stations: int,
        num_numeric: int,
        station_emb_dim: int = 8,
        meta_hidden: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.num_stations = int(num_stations)
        self.num_numeric = int(num_numeric)

        self.station_embedding = nn.Embedding(self.num_stations, int(station_emb_dim))
        self.meta_mlp = nn.Sequential(
            nn.Linear(int(station_emb_dim) + self.num_numeric, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(meta_hidden, meta_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim + meta_hidden, 1),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        if features.dim() > 2:
            features = torch.flatten(features, 1)

        # meta is float32 [B, 1 + num_numeric]: column 0 is the station index.
        station = meta[:, 0].long().clamp_(0, self.num_stations - 1)
        numeric = meta[:, 1 : 1 + self.num_numeric]

        meta_features = self.meta_mlp(torch.cat([self.station_embedding(station), numeric], dim=1))
        fused = torch.cat([features, meta_features], dim=1)
        return self.classifier(fused).squeeze(1)
