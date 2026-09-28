"""Evaluate a trained burst-type classifier and save reports."""

# NOTE: Vendored from H:\Burst Identifier (src/evaluation/evaluate_type.py).
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
from torch.utils.data import DataLoader

from callisto_trainer.core.dataset import CallistoBurstDataset
from callisto_trainer.core.type_metrics import compute_multiclass_metrics
from callisto_trainer.core.models.model_factory import create_model
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _ordered_class_names(config: dict[str, Any]) -> list[str]:
    classes = config["data"]["classes"]
    return [name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))]


def load_trained_type_model(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    """Load a checkpointed type model plus its saved config and class names."""
    checkpoint = _torch_load(checkpoint_path, device)
    model_config = checkpoint.get("config", config)
    class_names = _ordered_class_names(model_config)
    # Trainer addition: rebuild the physics branch when the checkpoint used one,
    # or load_state_dict would reject the extra parameters.
    physics_kwargs: dict[str, Any] = {}
    if bool(model_config["model"].get("use_physics", False)):
        from callisto_trainer.core.models.model_factory import model_kwargs_from_config

        physics_kwargs = model_kwargs_from_config(model_config["model"])

    model = create_model(
        model_config["model"]["name"],
        in_channels=int(model_config["model"]["in_channels"]),
        dropout=float(model_config["model"].get("dropout", 0.25)),
        num_classes=len(class_names),
        **physics_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_config, class_names


def predict_probabilities(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, softmax probabilities [N, K])`` over a dataloader."""
    y_true: list[int] = []
    chunks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            spectra, labels = batch[0], batch[1]
            inputs = [spectra.to(device, non_blocking=True)]
            if len(batch) > 2:  # physics features from the dataset
                inputs.append(batch[2].to(device, non_blocking=True))
            logits = model(*inputs)
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy().astype(int).tolist())

    probabilities = np.concatenate(chunks) if chunks else np.zeros((0, 0))
    return np.asarray(y_true, dtype=int), probabilities


def _predict_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, probabilities = predict_probabilities(model, dataloader, device)
    y_pred = probabilities.argmax(axis=1) if probabilities.size else np.zeros(0, dtype=int)
    return y_true, np.asarray(y_pred, dtype=int)


def split_dataloader(checkpoint_config: dict[str, Any], split: str) -> DataLoader:
    """The manifest split a checkpoint was trained on, as its model reads it."""
    dataset = CallistoBurstDataset(
        checkpoint_config["paths"]["manifest_path"],
        split=split,
        return_physics_features=bool(checkpoint_config["model"].get("use_physics", False)),
        feature_set=checkpoint_config["model"].get("feature_set"),
    )
    return DataLoader(
        dataset,
        batch_size=int(checkpoint_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(checkpoint_config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def predict_split(
    config: dict[str, Any], checkpoint_path: str | Path, split: str, with_rows: bool = False
) -> tuple:
    """``(y_true, probabilities, class_names, checkpoint_config)`` for one split.

    With ``with_rows`` the manifest rows of the samples, in the same order, are
    appended as a fifth item.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_config, class_names = load_trained_type_model(
        checkpoint_path, config, device
    )
    dataloader = split_dataloader(checkpoint_config, split)
    y_true, probabilities = predict_probabilities(model, dataloader, device)
    if with_rows:
        return y_true, probabilities, class_names, checkpoint_config, list(dataloader.dataset.rows)
    return y_true, probabilities, class_names, checkpoint_config


def save_confusion_matrix(
    metrics: dict[str, Any], class_names: list[str], output_path: str | Path
) -> None:
    """Save the K x K confusion-matrix counts as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [f"pred_{name}" for name in class_names])
        for name, row in zip(class_names, cm):
            writer.writerow([f"true_{name}"] + list(row))


def save_misclassified_files(
    rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> int:
    """Save a CSV of files where the predicted type differs from the true type."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "label",
        "label_id",
        "predicted_label",
        "predicted_label_id",
        "station",
        "date",
        "start_time",
    ]

    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_label, predicted_label in zip(rows, y_true, y_pred):
            if int(true_label) == int(predicted_label):
                continue
            writer.writerow(
                {
                    "file_path": row["file_path"],
                    "label": row.get("label", class_names[int(true_label)]),
                    "label_id": row.get("label_id", int(true_label)),
                    "predicted_label": class_names[int(predicted_label)],
                    "predicted_label_id": int(predicted_label),
                    "station": row.get("station", ""),
                    "date": row.get("date", ""),
                    "start_time": row.get("start_time", ""),
                }
            )
            count += 1
    return count


def _save_confusion_plot(
    metrics: dict[str, Any], class_names: list[str], figures_dir: str | Path
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "type_confusion_matrix.png", dpi=150)
    plt.close(fig)


def evaluate_type_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a type checkpoint on a manifest split and save reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_config, class_names = load_trained_type_model(
        checkpoint_path, config, device
    )

    dataloader = split_dataloader(checkpoint_config, split)
    dataset = dataloader.dataset

    y_true, y_pred = _predict_dataset(model, dataloader, device)
    metrics = compute_multiclass_metrics(y_true, y_pred, class_names)

    reports_dir = Path(checkpoint_config["paths"]["reports_dir"])
    figures_dir = Path(checkpoint_config["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / f"{split}_type_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_confusion_matrix(metrics, class_names, reports_dir / f"{split}_type_confusion_matrix.csv")
    misclassified_count = save_misclassified_files(
        dataset.rows,
        y_true,
        y_pred,
        class_names,
        reports_dir / f"{split}_type_misclassified_files.csv",
    )
    _save_confusion_plot(metrics, class_names, figures_dir)

    LOGGER.info(
        "Type evaluation: accuracy=%.4f macro_f1=%.4f (misclassified=%d)",
        metrics["accuracy"],
        metrics["macro_f1"],
        misclassified_count,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an e-CALLISTO burst-type classifier")
    parser.add_argument(
        "--config",
        default="configs/type_resnet18_rtx5060_8gb.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/type_resnet18/checkpoints/best.pt",
        help="Path to type model checkpoint",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_type_model(config, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
