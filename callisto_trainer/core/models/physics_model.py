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


class StationDateCorrection(nn.Module):
    """A bounded correction to the class scores from the file's station and date.

    The image decides; station and date may only nudge. Both are real signal --
    each station has its own interference and receiver, and solar activity and
    the ionosphere change over the year -- but in labelled data they are also a
    shortcut: how often a station's files hold a burst says more about which of
    its files were picked for labelling than about the Sun. So the nudge is
    capped by construction:

    * every class score moves by at most ``cap / 2``, through a ``tanh``, so the
      log-odds between any two classes -- and between "burst" and "not a burst"
      -- move by at most ``cap``. At ``cap = 1`` a region the image puts at 50%
      can end up anywhere from 27% to 73%, and one at 95% no lower than 87%;
    * it reads the image representation as well, so it can learn interactions
      ("this narrowband line, at this station, is its known carrier"), still
      within the same cap;
    * its last layer starts at zero, so training begins from the image model;
    * during training station and date are hidden at random -- together, so the
      image path has to stand on its own, and separately, so either one can be
      missing at inference. With both unknown it contributes exactly nothing.
    """

    def __init__(
        self,
        image_dim: int,
        num_stations: int,
        num_classes: int,
        cap: float = 1.0,
        station_emb_dim: int = 8,
        hidden: int = 64,
        dropout: float = 0.25,
        drop_all: float = 0.25,
        drop_station: float = 0.15,
        drop_date: float = 0.15,
    ) -> None:
        super().__init__()
        if cap < 0:
            raise ValueError(f"station/date cap must be >= 0, got {cap}")
        self.num_stations = int(num_stations)
        self.cap = float(cap)
        self.drop_all = float(drop_all)
        self.drop_station = float(drop_station)
        self.drop_date = float(drop_date)

        self.station_embedding = nn.Embedding(self.num_stations, int(station_emb_dim))
        # month_sin, month_cos, year, date_known follow the station index.
        self.meta_mlp = nn.Sequential(
            nn.Linear(int(station_emb_dim) + 4, 32),
            nn.ReLU(inplace=True),
        )
        self.image_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(int(image_dim), hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(num_classes)),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image_repr: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        station = meta[:, 0].round().long().clamp(0, self.num_stations - 1)
        date = meta[:, 1:5]
        if self.training:
            batch = meta.shape[0]
            hide_all = torch.rand(batch, device=meta.device) < self.drop_all
            hide_station = hide_all | (torch.rand(batch, device=meta.device) < self.drop_station)
            hide_date = hide_all | (torch.rand(batch, device=meta.device) < self.drop_date)
            station = torch.where(hide_station, torch.zeros_like(station), station)
            date = date * (~hide_date).unsqueeze(1).to(date.dtype)
        known = ((station > 0) | (date[:, 3] > 0.5)).to(image_repr.dtype).unsqueeze(1)

        meta_repr = self.meta_mlp(torch.cat([self.station_embedding(station), date], dim=1))
        raw = self.head(torch.cat([self.image_proj(image_repr), meta_repr.to(image_repr.dtype)], dim=1))
        return (0.5 * self.cap) * torch.tanh(raw) * known


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

    With a :class:`StationDateCorrection` (``station_date``), the file's station
    and date vector follows the region features in the same ``features`` tensor
    -- callers still pass two inputs -- and its bounded correction is added to
    the image-and-region scores.
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
        station_date: dict | None = None,
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
        # Only built when enabled, so checkpoints without it keep their keys.
        self.station_date: StationDateCorrection | None = None
        if station_date:
            self.station_date = StationDateCorrection(
                image_dim=self.feature_dim * self.num_views + hidden,
                num_classes=self.num_classes,
                dropout=dropout,
                **station_date,
            )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        views = image.reshape(batch * self.num_views, 1, image.shape[-2], image.shape[-1])
        embedded = self.backbone(views)
        if embedded.dim() > 2:
            embedded = torch.flatten(embedded, 1)
        embedded = embedded.reshape(batch, self.num_views * self.feature_dim)
        region = features[:, : self.num_features]
        fused = torch.cat([embedded, self.feature_mlp(region)], dim=1)
        logits = self.classifier(fused)
        if self.station_date is not None:
            logits = logits + self.station_date(fused, features[:, self.num_features:])
        return logits
