"""Pre-labelling assistance.

Two capabilities, deliberately kept distinct because they carry very different
levels of trust:

**Triage ordering** runs a trained binary checkpoint over pending files and
stores a burst probability on each. The queue can then be sorted by it, so the
likely bursts are reviewed first. This is a real model prediction and the only
claim made about it is a ranking.

**Box proposals** are *not* a trained detector. The type model is a classifier:
given a region it names the burst type, but it has no notion of where a burst
is. So candidate regions come from a plain signal-processing pass -- threshold
the normalized spectrum, group connected pixels, keep blobs of a plausible size
-- and the type checkpoint is then asked to classify each candidate. Proposals
are stored unconfirmed and drawn dashed; they never count as labels until the
operator accepts them. The UI says all of this out loud, because a proposal that
looks like a prediction invites misplaced trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from callisto_trainer.core.crops import (
    CropConfig,
    crop_from_normalized,
    normalize_full_spectrum,
)
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.logging_utils import get_logger

# The finder lives in core/region_finder.py so the exporter and inference share
# it without importing this module; these names are re-exported for callers that
# have always found them here.
from callisto_trainer.core.region_finder import (  # noqa: F401
    DEFAULT_MIN_AREA,
    Proposal,
    _label_connected,
    _label_connected_fallback,
    find_candidate_regions,
)

LOGGER = get_logger(__name__)

# Defaults for labelling suggestions, in normalized [0,1] units where 0 maps to
# -1 dB and 1 to +8 dB above background.
DEFAULT_THRESHOLD = 0.45          # ~+3 dB, a conservative "clearly above noise"
DEFAULT_MAX_CANDIDATES = 12


# -- model-backed scoring --------------------------------------------------


class CheckpointScorer:
    """Loads a checkpoint once and scores tensors with it."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        import torch

        from callisto_trainer.core.models.model_factory import create_model

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config: dict[str, Any] = checkpoint.get("config", {})

        classes = self.config.get("data", {}).get("classes", {})
        self.class_names = [
            name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))
        ]
        model_cfg = self.config.get("model", {})
        self.uses_metadata = bool(model_cfg.get("use_metadata", False))
        self.uses_physics = bool(model_cfg.get("use_physics", False))
        self.num_classes = int(model_cfg.get("num_classes", 1))
        self.threshold = float(self.config.get("training", {}).get("threshold", 0.5))

        from callisto_trainer.core.models.model_factory import model_kwargs_from_config

        kwargs: dict[str, Any] = model_kwargs_from_config(model_cfg)
        self.station_vocab = (
            (model_cfg.get("station_vocab", {}) or {})
            if self.uses_metadata and not self.uses_physics
            else {}
        )

        self.model = create_model(
            model_cfg.get("name", "resnet18"),
            in_channels=int(model_cfg.get("in_channels", 1)),
            dropout=float(model_cfg.get("dropout", 0.25)),
            num_classes=self.num_classes,
            **kwargs,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def score(
        self,
        tensor: np.ndarray,
        metadata_row: dict[str, Any] | None = None,
        physics: Any = None,
    ) -> tuple[str | None, float]:
        """Return ``(predicted_class_or_None, probability)`` for one ``[1,H,W]`` tensor."""
        torch = self.torch
        inputs = [torch.from_numpy(tensor[np.newaxis]).float().to(self.device)]

        if self.uses_physics:
            from callisto_trainer.core.burst_physics import physics_to_vector

            vector = physics_to_vector(physics)
            inputs.append(torch.from_numpy(vector[np.newaxis]).float().to(self.device))
        elif self.uses_metadata:
            from callisto_trainer.core.metadata_features import row_to_meta_vector

            vector = row_to_meta_vector(metadata_row or {}, self.station_vocab)
            inputs.append(torch.from_numpy(vector[np.newaxis]).float().to(self.device))

        with torch.no_grad():
            logits = self.model(*inputs)
            if self.num_classes == 1:
                probability = float(torch.sigmoid(logits.reshape(-1))[0].item())
                return None, probability
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(probabilities.argmax().item())
            name = self.class_names[index] if index < len(self.class_names) else None
            return name, float(probabilities[index].item())


def score_files_for_triage(
    paths: Sequence[tuple[int, str]],
    checkpoint_path: str | Path,
    pipeline_config: dict[str, Any],
    metadata_rows: dict[int, dict[str, Any]] | None = None,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> dict[int, float]:
    """Score whole files with a binary checkpoint. Returns ``{file_id: probability}``."""
    scorer = CheckpointScorer(checkpoint_path)
    crop_config = CropConfig.from_config(pipeline_config)
    from callisto_trainer.core.crops import whole_file_box

    scores: dict[int, float] = {}
    total = len(paths)
    for index, (file_id, path) in enumerate(paths):
        if progress is not None and progress(index, total, Path(path).name) is False:
            break
        try:
            spectrum, _ = read_fits_spectrum(path)
            normalized = normalize_full_spectrum(spectrum, pipeline_config)
            tensor = crop_from_normalized(
                normalized, whole_file_box(normalized.shape), crop_config, apply_margin=False
            )
            _, probability = scorer.score(
                tensor, (metadata_rows or {}).get(file_id)
            )
        except Exception as exc:
            LOGGER.warning("Could not score %s: %r", path, exc)
            continue
        scores[file_id] = probability
    return scores


def score_files_with_unified(
    paths: Sequence[tuple[int, str]],
    checkpoint_path: str | Path,
    pipeline_config: dict[str, Any],
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> dict[int, float]:
    """Score whole files by a unified model's strongest region. ``{file_id: evidence}``.

    Exactly the number Predict thresholds, so sorting by it puts the files the
    model would flag -- rightly or wrongly -- at the top of the review queue.
    """
    from callisto_trainer.core.inference import CascadePredictor

    predictor = CascadePredictor(pipeline_config, unified_checkpoint=checkpoint_path)
    scores: dict[int, float] = {}
    total = len(paths)
    for index, (file_id, path) in enumerate(paths):
        if progress is not None and progress(index, total, Path(path).name) is False:
            break
        result = predictor.predict_file(path)
        if result.error is None and result.burst_probability is not None:
            scores[file_id] = float(result.burst_probability)
    return scores


def propose_from_normalized(
    normalized: np.ndarray,
    pipeline_config: dict[str, Any],
    scorer: CheckpointScorer | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Proposal]:
    """Propose regions from an already-normalized spectrum, optionally typing them.

    Takes the array rather than a path because the labelling canvas already holds
    the decoded spectrum; re-reading the gzip would be the slowest part by far.
    """
    proposals = find_candidate_regions(
        normalized, threshold=threshold, min_area=min_area, max_candidates=max_candidates
    )
    if not proposals or scorer is None:
        return proposals

    crop_config = CropConfig.from_config(pipeline_config)
    for proposal in proposals:
        try:
            tensor = crop_from_normalized(normalized, proposal.as_box(), crop_config)
        except ValueError:
            continue
        proposal.burst_type, proposal.probability = scorer.score(tensor)
    return proposals


def propose_boxes_for_file(
    path: str | Path,
    pipeline_config: dict[str, Any],
    type_checkpoint: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    scorer: CheckpointScorer | None = None,
) -> tuple[list[Proposal], np.ndarray]:
    """Read a file, find candidate regions and, when possible, classify them."""
    spectrum, _ = read_fits_spectrum(path)
    normalized = normalize_full_spectrum(spectrum, pipeline_config)
    if scorer is None and type_checkpoint is not None:
        scorer = CheckpointScorer(type_checkpoint)
    proposals = propose_from_normalized(
        normalized,
        pipeline_config,
        scorer=scorer,
        threshold=threshold,
        min_area=min_area,
        max_candidates=max_candidates,
    )
    return proposals, normalized


def find_latest_checkpoint(outputs_dir: str | Path, task: str) -> Path | None:
    """Newest ``best.pt`` produced for ``task`` ("type" or "binary"), if any."""
    root = Path(outputs_dir)
    if not root.exists():
        return None
    candidates = [
        path / "checkpoints" / "best.pt"
        for path in sorted(root.iterdir(), reverse=True)
        if path.is_dir() and path.name.startswith(f"{task}_")
    ]
    return next((path for path in candidates if path.exists()), None)
