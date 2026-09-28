"""Inference for trained e-CALLISTO burst classifiers."""

# NOTE: Vendored from H:\Burst Identifier (src/inference/predict.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from callisto_trainer.core.manifest import read_manifest
from callisto_trainer.core.metadata_features import NUM_NUMERIC, row_to_meta_vector
from callisto_trainer.core.models.model_factory import create_model
from callisto_trainer.core.preprocess import preprocess_file
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _metadata_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    if not bool(model_config.get("model", {}).get("use_metadata", False)):
        return {}
    vocab = model_config["model"].get("station_vocab", {})
    return dict(
        use_metadata=True,
        num_stations=len(vocab) + 1,
        num_numeric=NUM_NUMERIC,
        station_emb_dim=int(model_config["model"].get("station_emb_dim", 8)),
    )


def _meta_vector_for(model_config: dict[str, Any], metadata: dict[str, Any]) -> np.ndarray | None:
    """Build the metadata feature vector for one file, or None if disabled."""
    if not bool(model_config.get("model", {}).get("use_metadata", False)):
        return None
    vocab = model_config["model"].get("station_vocab", {})
    return row_to_meta_vector(metadata, vocab)


def probability_to_alert_level(probability: float) -> str:
    """Map burst probability to alert text for the FITS Analyzer."""
    if probability < 0.50:
        return "No alert"
    if probability < 0.80:
        return "Possible burst"
    if probability < 0.90:
        return "Likely burst"
    return "High-confidence burst"


def load_model_for_inference(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    """Load model and return the config saved inside the checkpoint when present."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load(checkpoint_path, device)
    model_config = checkpoint.get("config", config)
    model = create_model(
        model_config["model"]["name"],
        in_channels=int(model_config["model"]["in_channels"]),
        dropout=float(model_config["model"].get("dropout", 0.25)),
        **_metadata_kwargs(model_config),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_config, device


# --- Cascade stage 2: burst-type classification (Type II / Type III / Other) ---

# Fallback ordering used only if a type checkpoint somehow lacks its class map.
_TYPE_CLASS_NAMES_DEFAULT = ["Type II", "Type III", "Other"]


def _type_class_names(type_config: dict[str, Any]) -> list[str]:
    """Class names ordered by label id (index 0 -> id 0)."""
    classes = type_config.get("data", {}).get("classes")
    if classes:
        return [name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))]
    return list(_TYPE_CLASS_NAMES_DEFAULT)


def load_type_model_for_inference(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device, list[str]]:
    """Load a multiclass softmax model: the 3-class type model or the unified one.

    Trainer addition: the architecture is rebuilt from the checkpoint's own
    config, including the physics branch when it has one. Building the bare image
    model for a physics-conditioned checkpoint fails in ``load_state_dict`` with a
    wall of missing ``backbone.*`` keys, because the fused model nests the
    backbone one level down.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load(checkpoint_path, device)
    model_config = checkpoint.get("config", config)
    class_names = _type_class_names(model_config)

    from callisto_trainer.core.models.model_factory import model_kwargs_from_config

    extra_kwargs: dict[str, Any] = {}
    if bool(model_config.get("model", {}).get("use_physics", False)):
        # Views and feature count come from the checkpoint's own config.
        extra_kwargs = model_kwargs_from_config(model_config["model"])

    model = create_model(
        model_config["model"]["name"],
        in_channels=int(model_config["model"]["in_channels"]),
        dropout=float(model_config["model"].get("dropout", 0.25)),
        num_classes=len(class_names),
        **extra_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_config, device, class_names


def _predict_type(
    model: torch.nn.Module,
    tensor: np.ndarray,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    """Return (burst_type, confidence, per-class probabilities) for one spectrum."""
    image = torch.from_numpy(tensor.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image).reshape(1, -1)
        probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
    best = int(probs.argmax())
    type_probabilities = {name: float(probs[i]) for i, name in enumerate(class_names)}
    return class_names[best], float(probs[best]), type_probabilities


def _add_type_fields(
    record: dict[str, Any],
    type_model: torch.nn.Module | None,
    tensor: np.ndarray,
    device: torch.device,
    class_names: list[str] | None,
) -> dict[str, Any]:
    """Enrich a prediction record with burst-type fields when a burst is detected.

    When no type model is available, or the file is not a burst, the type fields
    are set to ``None`` so the record schema stays stable across callers.
    """
    if type_model is not None and class_names and record.get("predicted_label") == "Burst":
        burst_type, confidence, probabilities = _predict_type(
            type_model, tensor, device, class_names
        )
        record["burst_type"] = burst_type
        record["type_confidence"] = confidence
        record["type_probabilities"] = probabilities
    else:
        record["burst_type"] = None
        record["type_confidence"] = None
        record["type_probabilities"] = None
    return record


def _predict_tensor(
    model: torch.nn.Module,
    tensor: np.ndarray,
    device: torch.device,
    meta: np.ndarray | None = None,
) -> float:
    image = torch.from_numpy(tensor.astype(np.float32)).unsqueeze(0).to(device)
    inputs: list[torch.Tensor] = [image]
    if meta is not None:
        meta_tensor = torch.from_numpy(np.asarray(meta, dtype=np.float32)).unsqueeze(0).to(device)
        inputs.append(meta_tensor)
    with torch.no_grad():
        logits = model(*inputs)
        logits = logits.reshape(-1)
        probability = torch.sigmoid(logits).cpu().item()
    return float(probability)


def _prediction_record(
    file_path: str | Path,
    probability: float,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    predicted_label = "Burst" if probability >= decision_threshold else "No_Burst"
    confidence = probability if predicted_label == "Burst" else 1.0 - probability
    return {
        "file_name": Path(file_path).name,
        "file_path": str(file_path),
        "predicted_label": predicted_label,
        "burst_probability": float(probability),
        "decision_threshold": float(decision_threshold),
        "confidence": float(confidence),
        "alert_level": probability_to_alert_level(probability),
    }


def predict_file(
    file_path: str | Path,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    type_checkpoint_path: str | Path | None = None,
    type_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preprocess one raw FITS file and return an Analyzer-ready prediction.

    When ``type_checkpoint_path`` is given and the file is classified as a burst,
    the cascade's stage-2 type model is run on the same preprocessed tensor and
    the record gains ``burst_type`` / ``type_confidence`` / ``type_probabilities``.
    """
    model, model_config, device = load_model_for_inference(checkpoint_path, config)
    type_model, class_names = None, None
    if type_checkpoint_path is not None:
        type_model, _, _, class_names = load_type_model_for_inference(
            type_checkpoint_path, type_config or config, device
        )

    tensor, metadata = preprocess_file(file_path, output_path=None, config=model_config)
    meta = _meta_vector_for(model_config, metadata)
    probability = _predict_tensor(model, tensor, device, meta)
    threshold = float(model_config["training"].get("threshold", 0.5))
    record = _prediction_record(file_path, probability, threshold)
    if type_checkpoint_path is not None:
        _add_type_fields(record, type_model, tensor, device, class_names)
    return record


def predict_files(
    file_paths: list[str | Path],
    checkpoint_path: str | Path,
    config: dict[str, Any],
    progress: bool = True,
    type_checkpoint_path: str | Path | None = None,
    type_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict many raw FITS files, loading the model(s) and weights only once.

    Pass ``type_checkpoint_path`` to also infer the burst type (Type II / III /
    Other) for every file the binary model flags as a burst.
    """
    model, model_config, device = load_model_for_inference(checkpoint_path, config)
    threshold = float(model_config["training"].get("threshold", 0.5))
    type_model, class_names = None, None
    if type_checkpoint_path is not None:
        type_model, _, _, class_names = load_type_model_for_inference(
            type_checkpoint_path, type_config or config, device
        )

    records: list[dict[str, Any]] = []
    total = len(file_paths)
    for index, file_path in enumerate(file_paths, start=1):
        try:
            tensor, metadata = preprocess_file(file_path, output_path=None, config=model_config)
            meta = _meta_vector_for(model_config, metadata)
            probability = _predict_tensor(model, tensor, device, meta)
            record = _prediction_record(file_path, probability, threshold)
            if type_checkpoint_path is not None:
                _add_type_fields(record, type_model, tensor, device, class_names)
            records.append(record)
            if progress:
                rec = records[-1]
                LOGGER.info(
                    "[%d/%d] %s -> %s (p=%.4f, %s)%s",
                    index, total, Path(file_path).name,
                    rec["predicted_label"], rec["burst_probability"], rec["alert_level"],
                    f" [{rec['burst_type']}]" if rec.get("burst_type") else "",
                )
        except Exception as exc:  # noqa: BLE001 - keep batch going on a bad file
            LOGGER.error("Failed to predict %s: %s", file_path, exc)
    return records


def predict_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    output_path: str | Path,
    limit: int | None = None,
    type_checkpoint_path: str | Path | None = None,
    type_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict every file in a manifest and save a CSV."""
    rows = read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    model, model_config, device = load_model_for_inference(checkpoint_path, config)
    type_model, class_names = None, None
    if type_checkpoint_path is not None:
        type_model, _, _, class_names = load_type_model_for_inference(
            type_checkpoint_path, type_config or config, device
        )
    predictions: list[dict[str, Any]] = []

    for row in rows:
        processed_path = Path(row.get("processed_path", ""))
        if processed_path.exists():
            with np.load(processed_path, allow_pickle=False) as loaded:
                tensor = loaded["spectrum"].astype(np.float32)
        else:
            tensor, _ = preprocess_file(row["file_path"], output_path=None, config=model_config)

        meta = _meta_vector_for(model_config, row)
        probability = _predict_tensor(model, tensor, device, meta)
        threshold = float(model_config["training"].get("threshold", 0.5))
        record = _prediction_record(row["file_path"], probability, threshold)
        if type_checkpoint_path is not None:
            _add_type_fields(record, type_model, tensor, device, class_names)
        record.update(
            {
                "true_label": row.get("label", ""),
                "true_label_id": row.get("label_id", ""),
                "split": row.get("split", ""),
            }
        )
        predictions.append(record)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "file_path",
        "true_label",
        "true_label_id",
        "split",
        "predicted_label",
        "burst_probability",
        "decision_threshold",
        "confidence",
        "alert_level",
    ]
    if type_checkpoint_path is not None:
        fieldnames += ["burst_type", "type_confidence", "type_probabilities"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        # ``type_probabilities`` is a dict; serialise it as JSON so the CSV cell
        # stays a single readable field. extrasaction="ignore" keeps the writer
        # robust if a record carries type fields the header does not include.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in predictions:
            row_out = dict(record)
            if isinstance(row_out.get("type_probabilities"), dict):
                row_out["type_probabilities"] = json.dumps(row_out["type_probabilities"])
            writer.writerow(row_out)

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run burst/no-burst inference")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--file", default=None, help="Single raw .fit.gz file to predict")
    parser.add_argument("--manifest", default=None, help="Manifest CSV for batch prediction")
    parser.add_argument(
        "--output",
        default="outputs/reports/inference_predictions.csv",
        help="Output CSV for manifest prediction",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional manifest row limit")
    parser.add_argument(
        "--type-checkpoint",
        default=None,
        help="Optional burst-type model checkpoint. When set, bursts also get a "
        "Type II / Type III / Other prediction (cascade stage 2).",
    )
    parser.add_argument(
        "--type-config",
        default=None,
        help="YAML config for the type model (defaults to the type checkpoint's saved config).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    type_config = load_config(args.type_config) if args.type_config else None

    if args.file:
        prediction = predict_file(
            args.file,
            args.checkpoint,
            config,
            type_checkpoint_path=args.type_checkpoint,
            type_config=type_config,
        )
        print(json.dumps(prediction, indent=2))
        return

    if args.manifest:
        predictions = predict_manifest(
            args.manifest,
            args.checkpoint,
            config,
            output_path=args.output,
            limit=args.limit,
            type_checkpoint_path=args.type_checkpoint,
            type_config=type_config,
        )
        LOGGER.info("Wrote %d predictions to %s", len(predictions), args.output)
        return

    parser.error("Provide either --file or --manifest")


if __name__ == "__main__":
    main()
