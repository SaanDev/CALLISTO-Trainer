"""Evaluate a trained burst classifier and save reports."""

# NOTE: Vendored from H:\Burst Identifier (src/evaluation/evaluate.py).
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
from callisto_trainer.core.metadata_features import NUM_NUMERIC
from callisto_trainer.core.metrics import compute_binary_metrics
from callisto_trainer.core.models.model_factory import create_model
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


def load_trained_model(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load a checkpointed model for evaluation."""
    model, _ = load_trained_model_and_config(checkpoint_path, config, device)
    return model


def load_trained_model_and_config(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpointed model and the config saved in that checkpoint."""
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
    return model, model_config


def _predict_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[float] = []
    y_prob: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                spectra, labels, meta = batch
                inputs = [spectra.to(device, non_blocking=True), meta.to(device, non_blocking=True)]
            else:
                spectra, labels = batch
                inputs = [spectra.to(device, non_blocking=True)]
            logits = model(*inputs)
            logits = logits.reshape(labels.shape)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


def save_confusion_matrix(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save confusion-matrix counts as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "pred_no_burst", "pred_burst"])
        writer.writerow(["true_no_burst", metrics["tn"], metrics["fp"]])
        writer.writerow(["true_burst", metrics["fn"], metrics["tp"]])


def save_misclassified_files(
    rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_path: str | Path,
) -> int:
    """Save a CSV of files where predicted class differs from true label."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    fieldnames = [
        "file_path",
        "label",
        "label_id",
        "predicted_label",
        "predicted_label_id",
        "burst_probability",
        "station",
        "date",
        "start_time",
    ]

    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_label, predicted_label, probability in zip(rows, y_true, y_pred, y_prob):
            if int(true_label) == int(predicted_label):
                continue
            writer.writerow(
                {
                    "file_path": row["file_path"],
                    "label": row["label"],
                    "label_id": row["label_id"],
                    "predicted_label": "Burst" if predicted_label else "No_Burst",
                    "predicted_label_id": int(predicted_label),
                    "burst_probability": float(probability),
                    "station": row.get("station", ""),
                    "date": row.get("date", ""),
                    "start_time": row.get("start_time", ""),
                }
            )
            count += 1
    return count


def _save_metric_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict[str, Any],
    figures_dir: str | Path,
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No_Burst", "Burst"])
    ax.set_yticks([0, 1], labels=["No_Burst", "Burst"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"PR-AUC={metrics['pr_auc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "pr_curve.png", dpi=150)
        plt.close(fig)


def evaluate_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a checkpoint on a manifest split and save reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_config = load_trained_model_and_config(checkpoint_path, config, device)

    use_metadata = bool(checkpoint_config.get("model", {}).get("use_metadata", False))
    metadata_vocab = checkpoint_config["model"].get("station_vocab") if use_metadata else None
    dataset = CallistoBurstDataset(
        checkpoint_config["paths"]["manifest_path"],
        split=split,
        metadata_vocab=metadata_vocab,
        return_metadata_features=use_metadata,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(checkpoint_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(checkpoint_config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    y_true, y_prob = _predict_dataset(model, dataloader, device)
    threshold = float(checkpoint_config["training"].get("threshold", config["training"]["threshold"]))
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    reports_dir = Path(checkpoint_config["paths"]["reports_dir"])
    figures_dir = Path(checkpoint_config["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_confusion_matrix(metrics, reports_dir / f"{split}_confusion_matrix.csv")
    misclassified_count = save_misclassified_files(
        dataset.rows,
        y_true,
        y_prob,
        threshold,
        reports_dir / f"{split}_misclassified_files.csv",
    )
    _save_metric_plots(y_true, y_prob, metrics, figures_dir)

    LOGGER.info("Evaluation metrics: %s", metrics)
    LOGGER.info("Misclassified files: %d", misclassified_count)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an e-CALLISTO burst classifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_model(config, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
