"""Region-based inference on unlabelled files.

## The unified model (default)

One model over regions. Every candidate region the finder proposes is encoded
exactly as the exporter encoded the training samples (see
:mod:`callisto_trainer.core.region_inputs`), and the model returns a probability
for background, RFI and each burst type. A region is a burst when its **burst
evidence** -- ``1 - P(No_Burst) - P(RFI)`` -- reaches the checkpoint's
**calibrated threshold**, tuned after training so that at most a chosen share of
held-out quiet files is flagged (see :mod:`callisto_trainer.core.file_eval`). A
file is a burst if any region is. Regions the model assigns to RFI are reported
as RFI, never as detections.

Before calibration existed the decision was the argmax, and nothing tied it to
how many quiet files it would flag; a checkpoint without a calibrated threshold
still falls back to that.

Which *type* a burst is called is corrected for how often each type really
occurs (see :mod:`callisto_trainer.core.type_priors`): the type probabilities
are shifted from the training mix toward the observed frequencies with the
strength the checkpoint was calibrated with. The shift only moves probability
between burst types, so burst evidence and the threshold are unaffected.

## Why this is not just ``predict_file``

The vendored :mod:`callisto_trainer.core.predict` runs the type model on the
whole-file tensor. That was correct when the type model was trained on
whole-file spectra sorted into folders. It is **not** correct for a type model
trained by this app, which learns from *crops of burst regions* -- a 224x224
view of a few tens of seconds and a few MHz, not a whole recording squeezed into
the same 224x224. Feeding it a whole file is out of distribution, and it would
answer confidently anyway. So the legacy cascade here mirrors training:

1. the binary model scores the **whole file** (as it was trained);
2. if it says burst, candidate regions are located in the normalized spectrum;
3. each candidate is cropped **exactly as the exporter crops a drawn box**;
4. the type model classifies each crop (as it was trained).

## The binary gate is not optional for the legacy cascade

Measured on this archive, the region finder produced at least one candidate in
**10 of 10 quiet files at every threshold tried between 0.22 and 0.45**, with a
median of 12 regions -- *more* than the 5-9 median found in burst files. Bright
connected regions are ubiquitous: receiver interference, carrier lines and
calibration artifacts all qualify. A type model that only ever saw real bursts
will still assign one of its classes to each, so ``CascadePredictor`` allows a
type model without the binary gate (inspecting a known-burst file that way is
useful), but the UI warns whenever the gate is absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from callisto_trainer.core.coords import SpectrumAxes, box_to_physical
from callisto_trainer.core.crops import (
    CropConfig,
    PixelBox,
    crop_from_normalized,
    normalize_full_spectrum,
    whole_file_box,
)
from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.predict import probability_to_alert_level

# The finder and its settings live in core/region_finder.py; they are re-exported
# here because this is where callers have always found them.
from callisto_trainer.core.region_finder import (  # noqa: F401
    ADAPTIVE_FLOOR,
    ADAPTIVE_PERCENTILE,
    DEFAULT_HYSTERESIS,
    DEFAULT_MAX_REGIONS,
    DEFAULT_MIN_AREA,
    DEFAULT_REGION_THRESHOLD,
    find_candidate_regions,
    resolve_threshold,
)

# Sentinel: take the finder's hysteresis from the unified checkpoint, which
# records the setting its training regions were mined and its threshold was
# calibrated with, and fall back to the module default.
FROM_CHECKPOINT = object()
from callisto_trainer.core.taxonomy import NO_BURST, NON_BURST_LABELS, RFI

LOGGER = get_logger(__name__)

# The unified model's background class. A region assigned this is discarded.
NO_BURST_LABEL = NO_BURST

# Regions are classified in batches of this many crops.
BATCH_SIZE = 32


@dataclass
class RegionResult:
    """One located region and what the model made of it."""

    row0: int
    row1: int
    col0: int
    col1: int
    area: int
    peak: float
    burst_type: str | None = None
    type_confidence: float | None = None
    type_probabilities: dict[str, float] = field(default_factory=dict)
    # Unified model only: 1 - P(No_Burst) - P(RFI), i.e. how strongly this
    # region is a burst of any type. Distinct from type_confidence, which is
    # confidence in the specific class chosen.
    burst_evidence: float | None = None
    freq_lo_mhz: float | None = None
    freq_hi_mhz: float | None = None
    t_start_s: float | None = None
    t_end_s: float | None = None
    # Drift rate for this region. For a region called Type II or Type III it is
    # the burst-parameter definition, from the region's own frequency range and
    # duration (burst_physics.box_parameters); otherwise the pixel measurement
    # the model was given, when it uses physics.
    drift_mhz_per_s: float | None = None
    physics_confidence: str | None = None
    burst_count: int | None = None
    # Plain-language reasons the region looks like interference, when any.
    interference_hints: list[str] = field(default_factory=list)

    @property
    def is_rfi(self) -> bool:
        return self.burst_type == RFI

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileResult:
    """The full prediction for one file."""

    file_path: str
    file_name: str
    predicted_label: str | None = None
    burst_probability: float | None = None
    decision_threshold: float = 0.5
    confidence: float | None = None
    alert_level: str | None = None
    station: str | None = None
    obs_date: str | None = None
    obs_time: str | None = None
    regions: list[RegionResult] = field(default_factory=list)
    dominant_type: str | None = None
    error: str | None = None
    # Unified model only: how many candidate regions were examined, how many
    # the model itself rejected, and which of those it called RFI.
    regions_examined: int = 0
    regions_rejected: int = 0
    rfi_regions: list[RegionResult] = field(default_factory=list)
    # Whether the decision threshold was calibrated to a false-alarm budget.
    threshold_calibrated: bool = False
    # Strength of the type-frequency correction the types were decided with.
    type_prior_strength: float | None = None

    @property
    def is_burst(self) -> bool:
        return self.predicted_label == "Burst"

    @property
    def region_summary(self) -> str:
        if self.error:
            return "error"
        if not self.is_burst:
            return f"RFI x{len(self.rfi_regions)}" if self.rfi_regions else "-"
        if not self.regions:
            return "no region located"
        counts: dict[str, int] = {}
        for region in self.regions:
            if region.burst_type:
                counts[region.burst_type] = counts.get(region.burst_type, 0) + 1
        return "  ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = [region.as_dict() for region in self.regions]
        payload["rfi_regions"] = [region.as_dict() for region in self.rfi_regions]
        return payload


def _dominant_type(regions: Sequence[RegionResult]) -> str | None:
    """The type of the largest confidently-typed region.

    Area rather than confidence: a big region is more likely to be the actual
    event, while a small bright speck is more likely to be interference that the
    classifier still had to put in some class.
    """
    typed = [region for region in regions if region.burst_type]
    if not typed:
        return None
    return max(typed, key=lambda region: region.area).burst_type


def _relative_alert(probability: float, threshold: float | None) -> str:
    """Alert wording relative to the decision threshold, not to a fixed 0.5.

    A calibrated threshold can sit at 0.3 or 0.8; describing evidence of 0.45 as
    "No alert" when the threshold is 0.3 would contradict the verdict beside it.
    Evidence is rescaled so the threshold maps to 0.5 before the usual bands
    apply.
    """
    if threshold is None or not 0.0 < threshold < 1.0:
        return probability_to_alert_level(probability)
    if probability >= threshold:
        scaled = 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)
    else:
        scaled = 0.5 * probability / threshold
    return probability_to_alert_level(float(scaled))


class CascadePredictor:
    """Loads the model(s) once and predicts many files.

    Three modes, in order of preference:

    * ``unified_checkpoint`` -- a single model over regions. It answers
      everything at once and, crucially, decides for itself whether a candidate
      region is background or interference, so no separate gate is needed.
    * ``binary_checkpoint`` + ``type_checkpoint`` -- the older two-model cascade.
    * either alone -- burst/no-burst only, or typed regions with no gate.

    ``burst_threshold`` overrides the unified checkpoint's calibrated threshold;
    left at ``None`` the calibrated one is used, and argmax when there is none.
    ``type_prior_strength`` likewise overrides the calibrated strength of the
    type-frequency correction (0 decides types as trained).
    """

    def __init__(
        self,
        pipeline_config: dict[str, Any],
        binary_checkpoint: str | Path | None = None,
        type_checkpoint: str | Path | None = None,
        unified_checkpoint: str | Path | None = None,
        device: str | None = None,
        region_threshold: float = DEFAULT_REGION_THRESHOLD,
        min_area: int = DEFAULT_MIN_AREA,
        max_regions: int = DEFAULT_MAX_REGIONS,
        adaptive_threshold: bool = True,
        burst_threshold: float | None = None,
        hysteresis: Any = FROM_CHECKPOINT,
        type_prior_strength: float | None = None,
    ) -> None:
        import torch

        if binary_checkpoint is None and type_checkpoint is None and unified_checkpoint is None:
            raise ValueError("At least one checkpoint is required.")

        self.config = pipeline_config
        self.crop_config = CropConfig.from_config(pipeline_config)
        self.region_threshold = float(region_threshold)
        self.min_area = int(min_area)
        self.max_regions = int(max_regions)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.hysteresis = None if hysteresis is FROM_CHECKPOINT else hysteresis
        hysteresis_from_checkpoint = hysteresis is FROM_CHECKPOINT
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch = torch

        self.binary_model = None
        self.binary_config: dict[str, Any] = {}
        self.decision_threshold = 0.5
        if binary_checkpoint is not None:
            from callisto_trainer.core.predict import load_model_for_inference

            self.binary_model, self.binary_config, _ = load_model_for_inference(
                binary_checkpoint, pipeline_config, self.device
            )
            self.decision_threshold = float(
                self.binary_config.get("training", {}).get("threshold", 0.5)
            )

        self.type_model = None
        self.type_class_names: list[str] = []
        if type_checkpoint is not None:
            from callisto_trainer.core.predict import load_type_model_for_inference

            (
                self.type_model,
                _,
                _,
                self.type_class_names,
            ) = load_type_model_for_inference(type_checkpoint, pipeline_config, self.device)

        # The unified model reuses the multiclass loader; its classes include
        # No_Burst, and it brings its own input contract (views, features) and
        # its calibrated threshold in its config.
        self.unified_model = None
        self.unified_class_names: list[str] = []
        self.unified_uses_physics = False
        self.unified_config: dict[str, Any] = {}
        self.encoder = None
        self.calibrated_threshold: float | None = None
        self.burst_threshold: float | None = None
        # Type-frequency correction: log(observed / training share) per burst
        # class, and how strongly to apply it.
        self.type_adjustment: dict[str, float] = {}
        self.calibrated_type_strength: float | None = None
        self.type_strength = 0.0
        if unified_checkpoint is not None:
            from callisto_trainer.core.predict import load_type_model_for_inference
            from callisto_trainer.core.region_inputs import RegionEncoder

            (
                self.unified_model,
                self.unified_config,
                _,
                self.unified_class_names,
            ) = load_type_model_for_inference(unified_checkpoint, pipeline_config, self.device)
            if NO_BURST_LABEL not in self.unified_class_names:
                raise ValueError(
                    f"{unified_checkpoint} is not a unified model: its classes are "
                    f"{self.unified_class_names}, with no '{NO_BURST_LABEL}'. Select it as "
                    "the burst-type model instead."
                )
            # Crop geometry and inputs exactly as this model was trained with.
            encoder_config = dict(pipeline_config)
            encoder_config.update(
                {key: self.unified_config[key] for key in ("crops", "data", "model")
                 if key in self.unified_config}
            )
            self.encoder = RegionEncoder(encoder_config)
            self.unified_uses_physics = self.encoder.spec.feature_set is not None
            inference = self.unified_config.get("inference", {}) or {}
            calibrated = inference.get("burst_threshold")
            self.calibrated_threshold = None if calibrated is None else float(calibrated)
            self.burst_threshold = (
                float(burst_threshold) if burst_threshold is not None else self.calibrated_threshold
            )
            priors = inference.get("type_priors") or {}
            self.type_adjustment = {
                str(name): float(value) for name, value in (priors.get("adjustment") or {}).items()
            }
            if self.type_adjustment and isinstance(priors.get("strength"), (int, float)):
                self.calibrated_type_strength = float(priors["strength"])
            self.type_strength = (
                float(type_prior_strength) if type_prior_strength is not None
                else (self.calibrated_type_strength or 0.0)
            )
            finder = inference.get("region_finder") or {}
            if hysteresis_from_checkpoint and "hysteresis" in finder:
                # A checkpoint from before hysteresis records no key and gets
                # the default; one that recorded None keeps plain thresholding.
                self.hysteresis = finder["hysteresis"]
                hysteresis_from_checkpoint = False
        if hysteresis_from_checkpoint:
            self.hysteresis = DEFAULT_HYSTERESIS

    @property
    def is_unified(self) -> bool:
        return self.unified_model is not None

    @property
    def needs_quiet(self) -> bool:
        """Whether the unified model takes the quiet-background view."""
        return self.encoder is not None and self.encoder.spec.needs_quiet

    def quiet_for(self, spectrum: np.ndarray | None) -> np.ndarray | None:
        """The quiet-background array a raw spectrum gives this model, or None.

        Callers holding a raw spectrum pass the result on to ``examine`` and
        friends; it is only computed when the model actually uses it.
        """
        return self.encoder.quiet(spectrum) if self.encoder is not None else None

    def region_finder_settings(self) -> dict[str, Any]:
        """The finder settings in effect, recorded beside any calibration."""
        return {
            "threshold": self.region_threshold,
            "adaptive": self.adaptive_threshold,
            "min_area": self.min_area,
            "max_regions": self.max_regions,
            "hysteresis": self.hysteresis,
        }

    # -- single file -------------------------------------------------------

    def predict_file(self, path: str | Path) -> FileResult:
        """Run the cascade on one raw FITS file."""
        path = Path(path)
        result = FileResult(file_path=str(path), file_name=path.name)
        try:
            spectrum, metadata = read_fits_spectrum_and_axes(path)
            normalized = normalize_full_spectrum(spectrum, self.config)
        except Exception as exc:
            result.error = str(exc)
            LOGGER.warning("Could not read %s: %r", path, exc)
            return result

        result.station = metadata.get("station")
        result.obs_date = metadata.get("date")
        result.obs_time = metadata.get("start_time")
        axes = SpectrumAxes.from_metadata(metadata)
        return self.predict_normalized(
            normalized, axes, result, metadata, quiet=self.quiet_for(spectrum)
        )

    def predict_normalized(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None,
        result: FileResult,
        metadata: dict[str, Any] | None = None,
        quiet: np.ndarray | None = None,
    ) -> FileResult:
        """Run the cascade on an already-normalized spectrum.

        ``quiet`` is the file's quiet-background array (see :meth:`quiet_for`),
        required by a model that uses the quiet-background view.
        """
        if self.is_unified:
            rfi_channels = (metadata or {}).get("rfi_channels_mhz")
            return self._predict_unified(normalized, axes, result, rfi_channels, quiet)

        if self.binary_model is not None:
            probability = self._score_binary(normalized, metadata or {})
            result.burst_probability = probability
            result.decision_threshold = self.decision_threshold
            result.predicted_label = (
                "Burst" if probability >= self.decision_threshold else "No_Burst"
            )
            result.confidence = probability if result.is_burst else 1.0 - probability
            result.alert_level = probability_to_alert_level(probability)

        # Stage 2 runs when the file is a burst, or when there is no binary model
        # to gate on and the caller is asking purely for typed regions.
        gate_open = self.binary_model is None or result.is_burst
        if self.type_model is not None and gate_open:
            result.regions = self._typed_regions(normalized, axes)
            result.dominant_type = _dominant_type(result.regions)
        return result

    # -- unified -----------------------------------------------------------

    def examine(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None,
        rfi_channels: Any = None,
        quiet: np.ndarray | None = None,
    ) -> list[RegionResult]:
        """Every candidate region with the unified model's probabilities.

        No decision is applied: each region carries its burst evidence and full
        probability vector, which is what calibration needs. ``decide`` turns
        these into verdicts. The probabilities already carry the type-frequency
        correction, which leaves burst evidence as it was.
        """
        proposals = self._proposals(normalized)
        boxes = [PixelBox(p.row0, p.row1, p.col0, p.col1) for p in proposals]
        probabilities, encoded = self._unified_probabilities(
            normalized, axes, boxes, rfi_channels, quiet=quiet
        )
        if len(boxes):
            from callisto_trainer.core.type_priors import adjust_probabilities

            probabilities = adjust_probabilities(
                probabilities, self.unified_class_names, self.type_adjustment, self.type_strength
            )

        from callisto_trainer.core.region_features import describe_region
        from callisto_trainer.core.unified_metrics import burst_evidence

        evidence = burst_evidence(probabilities, self.unified_class_names) if len(boxes) else []
        regions: list[RegionResult] = []
        for index, proposal in enumerate(proposals):
            region = RegionResult(
                row0=proposal.row0,
                row1=proposal.row1,
                col0=proposal.col0,
                col1=proposal.col1,
                area=proposal.area,
                peak=proposal.peak,
                type_probabilities={
                    name: float(probabilities[index, i])
                    for i, name in enumerate(self.unified_class_names)
                },
                burst_evidence=float(evidence[index]),
            )
            sample = encoded[index]
            if sample.physics is not None:
                region.drift_mhz_per_s = sample.physics.drift_mhz_per_s
                region.physics_confidence = sample.physics.confidence
                region.burst_count = sample.physics.burst_count
            region.interference_hints = describe_region(sample.region)
            self._attach_physical(region, axes)
            regions.append(region)
        return regions

    def burst_evidence_for(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None,
        boxes: Sequence[PixelBox],
        rfi_channels: Any = None,
        quiet: np.ndarray | None = None,
    ) -> np.ndarray:
        """Burst evidence for arbitrary regions of one file (hard-negative mining)."""
        from callisto_trainer.core.unified_metrics import burst_evidence

        if not boxes:
            return np.zeros(0)
        probabilities, _ = self._unified_probabilities(
            normalized, axes, list(boxes), rfi_channels, quiet=quiet
        )
        return burst_evidence(probabilities, self.unified_class_names)

    def _unified_probabilities(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None,
        boxes: Sequence[PixelBox],
        rfi_channels: Any = None,
        quiet: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[Any]]:
        """``(probabilities [N, K], encoded regions)`` for ``boxes``, batched."""
        torch = self.torch
        if not boxes:
            return np.zeros((0, len(self.unified_class_names))), []
        context = self.encoder.context(normalized, axes, rfi_channels)
        encoded = [
            self.encoder.encode(normalized, box, axes, context, quiet=quiet) for box in boxes
        ]

        chunks: list[np.ndarray] = []
        for start in range(0, len(encoded), BATCH_SIZE):
            batch = encoded[start:start + BATCH_SIZE]
            inputs = [
                torch.from_numpy(np.stack([item.image for item in batch])).float().to(self.device)
            ]
            if self.unified_uses_physics:
                inputs.append(
                    torch.from_numpy(np.stack([item.features for item in batch]))
                    .float()
                    .to(self.device)
                )
            with torch.no_grad():
                logits = self.unified_model(*inputs).reshape(len(batch), -1)
                chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(chunks), encoded

    def decide(self, region: RegionResult) -> bool:
        """Label a region from its probabilities. Returns whether it is a burst."""
        names = self.unified_class_names
        probabilities = region.type_probabilities
        evidence = float(region.burst_evidence or 0.0)
        burst_names = [name for name in names if name not in NON_BURST_LABELS]
        other_names = [name for name in names if name in NON_BURST_LABELS]

        if self.burst_threshold is None:
            is_burst = max(names, key=lambda name: probabilities.get(name, 0.0)) in burst_names
        else:
            is_burst = evidence >= self.burst_threshold

        pool, mass = (burst_names, evidence) if is_burst else (other_names, 1.0 - evidence)
        best = max(pool, key=lambda name: probabilities.get(name, 0.0))
        region.burst_type = best
        # Confidence in the type *given* the burst / not-burst decision.
        region.type_confidence = float(probabilities.get(best, 0.0) / max(mass, 1e-9))
        return is_burst

    def _predict_unified(
        self,
        normalized: np.ndarray,
        axes: SpectrumAxes | None,
        result: FileResult,
        rfi_channels: Any = None,
        quiet: np.ndarray | None = None,
    ) -> FileResult:
        """One model over regions: detection, typing and location at once.

        Each candidate is classified into background, RFI or a burst type, and
        regions that are not bursts are discarded from the findings -- that
        rejection is the thing a brightness threshold could never do. The file's
        burst probability is the strongest burst evidence of any region.
        """
        from callisto_trainer.core.burst_physics import BOX_DRIFT_TYPES, box_parameters

        regions = self.examine(normalized, axes, rfi_channels, quiet=quiet)
        bursts: list[RegionResult] = []
        rfi_regions: list[RegionResult] = []
        for region in regions:
            if self.decide(region):
                bursts.append(region)
                if region.burst_type in BOX_DRIFT_TYPES and axes is not None:
                    parameters = box_parameters(
                        axes, region.row0, region.row1, region.col0, region.col1,
                        region.burst_type,
                    )
                    region.drift_mhz_per_s = parameters.drift_mhz_per_s
                    region.physics_confidence = parameters.confidence
            elif region.is_rfi:
                rfi_regions.append(region)

        result.regions = bursts
        result.rfi_regions = rfi_regions
        result.regions_examined = len(regions)
        result.regions_rejected = len(regions) - len(bursts)

        probability = max((float(region.burst_evidence or 0.0) for region in regions), default=0.0)
        result.burst_probability = probability
        result.decision_threshold = (
            self.burst_threshold if self.burst_threshold is not None else 0.5
        )
        result.threshold_calibrated = (
            self.burst_threshold is not None and self.burst_threshold == self.calibrated_threshold
        )
        result.predicted_label = "Burst" if bursts else "No_Burst"
        result.confidence = probability if bursts else 1.0 - probability
        result.alert_level = _relative_alert(probability, self.burst_threshold)
        result.dominant_type = _dominant_type(bursts)
        result.type_prior_strength = self.type_strength if self.type_adjustment else None
        return result

    # -- stages ------------------------------------------------------------

    def _proposals(self, normalized: np.ndarray):
        return find_candidate_regions(
            normalized,
            threshold=resolve_threshold(normalized, self.region_threshold, self.adaptive_threshold),
            min_area=self.min_area,
            max_candidates=self.max_regions,
            hysteresis=self.hysteresis,
        )

    def _attach_physical(self, region: RegionResult, axes: SpectrumAxes | None) -> None:
        if axes is None:
            return
        physical = box_to_physical(axes, region.row0, region.row1, region.col0, region.col1)
        region.freq_lo_mhz = physical["freq_lo_mhz"]
        region.freq_hi_mhz = physical["freq_hi_mhz"]
        region.t_start_s = physical["t_start_s"]
        region.t_end_s = physical["t_end_s"]

    def _score_binary(self, normalized: np.ndarray, metadata: dict[str, Any]) -> float:
        """Whole-file probability, matching how the binary model was trained."""
        torch = self.torch
        tensor = crop_from_normalized(
            normalized, whole_file_box(normalized.shape), self.crop_config, apply_margin=False
        )
        inputs = [torch.from_numpy(tensor[np.newaxis]).float().to(self.device)]

        if bool(self.binary_config.get("model", {}).get("use_metadata", False)):
            from callisto_trainer.core.metadata_features import row_to_meta_vector

            vocab = self.binary_config["model"].get("station_vocab", {}) or {}
            vector = row_to_meta_vector(metadata, vocab)
            inputs.append(torch.from_numpy(vector[np.newaxis]).float().to(self.device))

        with torch.no_grad():
            logits = self.binary_model(*inputs).reshape(-1)
            return float(torch.sigmoid(logits)[0].item())

    def _typed_regions(
        self, normalized: np.ndarray, axes: SpectrumAxes | None
    ) -> list[RegionResult]:
        """Legacy type model: locate candidate regions and classify each crop."""
        regions: list[RegionResult] = []
        for proposal in self._proposals(normalized):
            region = RegionResult(
                row0=proposal.row0,
                row1=proposal.row1,
                col0=proposal.col0,
                col1=proposal.col1,
                area=proposal.area,
                peak=proposal.peak,
            )
            try:
                tensor = crop_from_normalized(
                    normalized,
                    PixelBox(region.row0, region.row1, region.col0, region.col1),
                    self.crop_config,
                )
            except ValueError:
                continue

            burst_type, confidence, probabilities = self._classify_crop(tensor)
            region.burst_type = burst_type
            region.type_confidence = confidence
            region.type_probabilities = probabilities
            self._attach_physical(region, axes)
            regions.append(region)
        return regions

    def _classify_crop(
        self, tensor: np.ndarray, unified: bool = False, physics: Any = None
    ) -> tuple[str, float, dict[str, float]]:
        """Legacy type model on one ``[1, H, W]`` crop."""
        torch = self.torch
        image = torch.from_numpy(tensor[np.newaxis]).float().to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.type_model(image).reshape(1, -1), dim=1)[0]
            values = probabilities.cpu().numpy()
        names = self.type_class_names
        best = int(values.argmax())
        return (
            names[best] if best < len(names) else str(best),
            float(values[best]),
            {name: float(values[i]) for i, name in enumerate(names)},
        )


def predict_paths(
    paths: Sequence[str | Path],
    pipeline_config: dict[str, Any],
    binary_checkpoint: str | Path | None = None,
    type_checkpoint: str | Path | None = None,
    unified_checkpoint: str | Path | None = None,
    region_threshold: float = DEFAULT_REGION_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    max_regions: int = DEFAULT_MAX_REGIONS,
    adaptive_threshold: bool = True,
    progress: Callable[[int, int, str], bool | None] | None = None,
    burst_threshold: float | None = None,
    hysteresis: Any = FROM_CHECKPOINT,
    type_prior_strength: float | None = None,
) -> list[FileResult]:
    """Run the cascade over many files, loading each model once."""
    predictor = CascadePredictor(
        pipeline_config,
        binary_checkpoint=binary_checkpoint,
        type_checkpoint=type_checkpoint,
        unified_checkpoint=unified_checkpoint,
        region_threshold=region_threshold,
        min_area=min_area,
        max_regions=max_regions,
        adaptive_threshold=adaptive_threshold,
        burst_threshold=burst_threshold,
        hysteresis=hysteresis,
        type_prior_strength=type_prior_strength,
    )

    results: list[FileResult] = []
    total = len(paths)
    for index, path in enumerate(paths):
        if progress is not None and progress(index, total, Path(path).name) is False:
            break
        results.append(predictor.predict_file(path))
    return results


def checkpoint_inference_settings(checkpoint_path: str | Path) -> dict[str, Any]:
    """The ``inference`` block a unified checkpoint was calibrated with, or ``{}``."""
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        LOGGER.warning("Could not read %s: %r", checkpoint_path, exc)
        return {}
    return dict((checkpoint.get("config", {}) or {}).get("inference", {}) or {})


# -- reports ---------------------------------------------------------------

CSV_COLUMNS = [
    "file_name",
    "file_path",
    "station",
    "obs_date",
    "obs_time",
    "predicted_label",
    "burst_probability",
    "decision_threshold",
    "confidence",
    "alert_level",
    "dominant_type",
    "region_count",
    "regions",
    "rfi_region_count",
    "error",
]


def write_csv(results: Sequence[FileResult], output_path: str | Path) -> Path:
    """One row per file; regions are JSON in a single cell to keep the CSV flat."""
    import csv
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = result.as_dict()
            row["region_count"] = len(result.regions)
            row["rfi_region_count"] = len(result.rfi_regions)
            row["regions"] = json.dumps(
                [
                    {
                        "burst_type": region.burst_type,
                        "confidence": region.type_confidence,
                        "burst_evidence": region.burst_evidence,
                        "pixel_box": [region.row0, region.row1, region.col0, region.col1],
                        "freq_mhz": [region.freq_lo_mhz, region.freq_hi_mhz],
                        "time_s": [region.t_start_s, region.t_end_s],
                    }
                    for region in result.regions
                ]
            )
            writer.writerow(row)
    return output_path


def write_json(results: Sequence[FileResult], output_path: str | Path) -> Path:
    """Full nested results, including every per-class probability."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "region_finder": {
            "method": "brightness threshold + 8-connected components",
            "note": "regions are located heuristically, not by a trained detector",
        },
        "files": [result.as_dict() for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
